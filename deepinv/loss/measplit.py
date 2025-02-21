from __future__ import annotations
from typing import Optional, Tuple, Union
from copy import deepcopy
from warnings import warn
import torch
from deepinv.physics import Inpainting, Physics
from deepinv.loss.loss import Loss
from deepinv.loss.metric.metric import Metric
from deepinv.physics.generator import (
    PhysicsGenerator,
    BernoulliSplittingMaskGenerator,
    Phase2PhaseSplittingMaskGenerator,
    Artifact2ArtifactSplittingMaskGenerator,
)
from deepinv.models.dynamic import TimeAveragingNet
from deepinv.physics.time import TimeMixin
from deepinv.models.base import Reconstructor


class SplittingLoss(Loss):
    r"""
    Measurement splitting loss.

    Implements measurement splitting loss from `Yaman et al. <https://pubmed.ncbi.nlm.nih.gov/32614100/>`_ (SSDU) for MRI,
    `Hendriksen et al. <https://arxiv.org/abs/2001.11801>`_ (Noise2Inverse) for CT,
    `Acar et al. <https://link.springer.com/chapter/10.1007/978-3-030-88552-6_4>`_ dynamic MRI. Also see :class:`deepinv.loss.Artifact2ArtifactLoss`, :class:`deepinv.loss.Phase2PhaseLoss` for similar.

    Splits the measurement and forward operator :math:`\forw{}` (of size :math:`m`)
    into two smaller pairs  :math:`(y_1,A_1)` (of size :math:`m_1`) and  :math:`(y_2,A_2)` (of size :math:`m_2`) ,
    to compute the self-supervised loss:

    .. math::

        \frac{m}{m_2}\| y_2 - A_2 \inversef{y_1}{A_1}\|^2

    where :math:`R` is the trainable network, :math:`A_1 = M_1 \forw{}, A_2 = M_2 \forw{}`, and :math:`M_i` are randomly
    generated masks (i.e. diagonal matrices) such that :math:`M_1+M_2=\mathbb{I}_m`.

    See :ref:`sphx_glr_auto_examples_self-supervised-learning_demo_splitting_loss.py` for usage example.

    .. note::

        If the forward operator has its own subsampling mask :math:`M_{\forw{}}`, e.g. :class:`deepinv.physics.Inpainting`
        or :class:`deepinv.physics.MRI`,
        the splitting masks will be subsets of the physics' mask such that :math:`M_1+M_2=M_{\forw{}}`

    This loss was used in SSDU for MRI in `Yaman et al. Self-supervised learning of physics-guided reconstruction neural
    networks without fully sampled reference data <https://pubmed.ncbi.nlm.nih.gov/32614100/>`_

    By default, the error is computed using the MSE metric, however any appropriate metric can be used.

    .. warning::

        The model should be adapted before training using the method :func:`adapt_model <deepinv.loss.SplittingLoss.adapt_model>`
        to include the splitting mechanism at the input.

    .. note::

        To obtain the best test performance, the trained model should be averaged at test time
        over multiple realizations of the splitting, i.e.
        :math:`\hat{x} = \frac{1}{N}\sum_{i=1}^N \inversef{y_1^{(i)}}{A_1^{(i)}}`. To disable this, set ``eval_n_samples=1``.

    .. note::

        To disable measurement splitting (and use the full input) at evaluation time, set ``eval_split_input=False``. This is done in `SSDU <https://pubmed.ncbi.nlm.nih.gov/32614100/>`_.

    :param Metric, torch.nn.Module metric: metric used for computing data consistency,
        which is set as the mean squared error by default.
    :param float split_ratio: splitting ratio, should be between 0 and 1. The size of :math:`y_1` increases
        with the splitting ratio. Ignored if ``mask_generator`` passed.
    :param deepinv.physics.generator.PhysicsGenerator, None mask_generator: function to generate the mask. If
        None, the :class:`deepinv.physics.generator.BernoulliSplittingMaskGenerator` is used, with the parameters ``split_ratio`` and ``pixelwise``.
    :param int eval_n_samples: Number of samples used for averaging at evaluation time. Must be greater than 0.
    :param bool eval_split_input: if True, perform input measurement splitting during evaluation. If False, use full measurement at eval (no MC samples are performed and eval_split_output will have no effect)
    :param bool eval_split_output: at evaluation time, pass the output through the output mask too.
        i.e. :math:`(\sum_{j=1}^N M_2^{(j)})^{-1} \sum_{i=1}^N M_2^{(i)} \inversef{y_1^{(i)}}{A_1^{(i)}}`.
        Only valid when :math:`y` is same domain (and dimension) as :math:`x`. Although better results may be observed on small datasets, more samples must be used for bigger images. Defaults to ``False``.
    :param bool pixelwise: if ``True``, create pixelwise splitting masks i.e. zero all channels simultaneously. Ignored if ``mask_generator`` passed.

    |sep|

    :Example:

    >>> import torch
    >>> import deepinv as dinv
    >>> physics = dinv.physics.Inpainting(tensor_size=(1, 8, 8), mask=0.5)
    >>> model = dinv.models.MedianFilter()
    >>> loss = dinv.loss.SplittingLoss(split_ratio=0.9, eval_n_samples=2)
    >>> model = loss.adapt_model(model) # important step!
    >>> x = torch.ones((1, 1, 8, 8))
    >>> y = physics(x)
    >>> x_net = model(y, physics, update_parameters=True) # save random mask in forward pass
    >>> l = loss(x_net, y, physics, model)
    >>> print(l.item() > 0)
    True


    """

    def __init__(
        self,
        metric: Union[Metric, torch.nn.Module] = torch.nn.MSELoss(),
        split_ratio: float = 0.9,
        mask_generator: Optional[PhysicsGenerator] = None,
        eval_n_samples=5,
        eval_split_input=True,
        eval_split_output=False,
        pixelwise=True,
    ):
        super().__init__()
        self.name = "ms"
        self.metric = metric
        self.mask_generator = mask_generator
        self.split_ratio = split_ratio
        self.eval_n_samples = eval_n_samples
        self.eval_split_input = eval_split_input
        self.eval_split_output = eval_split_output
        self.pixelwise = pixelwise

    @staticmethod
    def split(mask: torch.Tensor, y: torch.Tensor, physics: Optional[Physics] = None):
        r"""Perform splitting given mask

        :param torch.Tensor mask: splitting mask
        :param torch.Tensor y: input data
        :param deepinv.physics.Physics physics: physics to split, retaining its original noise model. If ``None``, only :math:`y` is split.
        """
        inp = Inpainting(y.size()[1:], mask=mask, device=y.device)

        # divide measurements y_i = M_i * y
        y_split = inp.A(y)

        # concatenate operators A_i = M_i * A
        if physics is None:
            return y_split

        physics_split = inp * physics
        physics_split.noise_model = physics.noise_model

        return y_split, physics_split

    def forward(self, x_net, y, physics, model, **kwargs):
        r"""
        Computes the measurement splitting loss

        :param torch.Tensor y: Measurements.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction function.
        :return: (:class:`torch.Tensor`) loss.
        """
        # Get splitting mask and make sure it is subsampled from physics mask, if it exists
        mask = model.get_mask() * getattr(physics, "mask", 1.0)

        # Create output mask M_2 = I - M_1
        mask2 = getattr(physics, "mask", 1.0) - mask
        y2, physics2 = self.split(mask2, y, physics)

        loss_ms = self.metric(physics2.A(x_net), y2)

        return loss_ms / mask2.mean()  # normalize loss

    def adapt_model(
        self, model: torch.nn.Module, eval_n_samples=None
    ) -> SplittingModel:
        r"""
        Apply random splitting to input.

        This method modifies a reconstruction
        model :math:`R` to include the splitting mechanism at the input:

        .. math::

            \hat{R}(y, A) = \frac{1}{N}\sum_{i=1}^N \inversef{y_1^{(i)}}{A_1^{(i)}}

        where :math:`N\geq 1` is the number of Monte Carlo samples,
        and :math:`y_1^{(i)}` and :math:`A_1^{(i)}` are obtained by
        randomly splitting the measurements :math:`y` and operator :math:`A`.
        During training (i.e. when ``model.train()``), we use only one sample, i.e. :math:`N=1`
        for computational efficiency, whereas at test time, we use multiple samples for better performance.
        For other parameters that control how splitting is applied, see the class parameters.

        :param torch.nn.Module model: Reconstruction model.
        :param int eval_n_samples: deprecated. Pass ``eval_n_samples`` at class initialisation instead.
        :return: (:class:`torch.nn.Module`) Model modified for evaluation.
        """
        if eval_n_samples is not None:
            warn(
                "eval_n_samples parameter is deprecated. Pass eval_n_samples at init: SplittingLoss(eval_n_samples=...)"
            )

        if isinstance(model, self.SplittingModel):
            return model
        else:
            return self.SplittingModel(
                model,
                split_ratio=self.split_ratio,
                mask_generator=self.mask_generator,
                eval_n_samples=self.eval_n_samples,
                eval_split_input=self.eval_split_input,
                eval_split_output=self.eval_split_output,
                pixelwise=self.pixelwise,
            )

    class SplittingModel(Reconstructor):
        """
        Model wrapper when using SplittingLoss.

        Performs input splitting during forward pass. At evaluation,
        perform forward passes for multiple realisations of splitting mask and average.

        :param deepinv.models.Reconstructor model: base model
        :param float split_ratio: splitting ratio, should be between 0 and 1. The size of :math:`y_1` increases
            with the splitting ratio. Ignored if ``mask_generator`` passed.
        :param deepinv.physics.generator.PhysicsGenerator, None mask_generator: function to generate the mask. If
            None, the :class:`deepinv.physics.generator.BernoulliSplittingMaskGenerator` is used, with the parameters ``split_ratio`` and ``pixelwise``.
        :param int eval_n_samples: Number of samples used for averaging at evaluation time. Must be greater than 0.
        :param bool eval_split_input: if True, perform input measurement splitting during evaluation. If False, use full measurement at eval (no MC samples are performed and eval_split_output will have no effect)
        :param bool eval_split_output: at evaluation time, pass the output through the output mask too.
            i.e. :math:`(\sum_{j=1}^N M_2^{(j)})^{-1} \sum_{i=1}^N M_2^{(i)} \inversef{y_1^{(i)}}{A_1^{(i)}}`.
            Only valid when :math:`y` is same domain (and dimension) as :math:`x`. Although better results may be observed on small datasets, more samples must be used for bigger images. Defaults to ``False``.
        :param bool pixelwise: if ``True``, create pixelwise splitting masks i.e. zero all channels simultaneously. Ignored if ``mask_generator`` passed.

        """

        def __init__(
            self,
            model,
            split_ratio,
            mask_generator,
            eval_n_samples,
            eval_split_input,
            eval_split_output,
            pixelwise,
        ):
            super().__init__()
            self.model = model
            self.split_ratio = split_ratio
            self.eval_n_samples = eval_n_samples
            self.mask = 0
            self.mask_generator = mask_generator
            self.eval_split_input = eval_split_input
            self.eval_split_output = eval_split_output
            self.pixelwise = pixelwise

        @staticmethod
        def split(mask, y, physics):
            return SplittingLoss.split(mask, y, physics)

        def forward(
            self, y: torch.Tensor, physics: Physics, update_parameters: bool = False
        ):
            """
            Adapted model forward pass for input splitting. During training, only one splitting realisation is performed for computational efficiency.
            """

            if (
                self.mask_generator is None
                or self.mask_generator.tensor_size != y.size()[1:]
            ):
                self.mask_generator = BernoulliSplittingMaskGenerator(
                    tensor_size=y.size()[1:],
                    split_ratio=self.split_ratio,
                    pixelwise=self.pixelwise,
                    device=y.device,
                )

            with torch.set_grad_enabled(self.training):
                if not self.eval_split_input and not self.training:
                    # No splitting
                    return self.model(y, physics)
                elif (
                    self.eval_split_output
                    and self.eval_split_input
                    and not self.training
                ):
                    return self._forward_split_input_output(y, physics)
                else:
                    return self._forward_split_input(
                        y, physics, update_parameters=update_parameters
                    )

        def _forward_split_input(
            self, y: torch.Tensor, physics: Physics, update_parameters: bool = False
        ):
            eval_n_samples = 1 if self.training else self.eval_n_samples
            out = 0

            for _ in range(eval_n_samples):
                # Perform input masking
                mask = self.mask_generator.step(
                    y.size(0), input_mask=getattr(physics, "mask", None)
                )["mask"]
                y1, physics1 = self.split(mask, y, physics)

                # Forward pass
                out += self.model(y1, physics1) / eval_n_samples

            if self.training and update_parameters:
                self.mask = mask.clone()

            return out

        def _forward_split_input_output(self, y: torch.Tensor, physics: Physics):
            """
            Perform splitting at model output too, only at eval time
            """
            out = 0
            normaliser = torch.zeros_like(y)

            for _ in range(self.eval_n_samples):
                # Perform input masking
                mask = self.mask_generator.step(
                    y.size(0), input_mask=getattr(physics, "mask", None)
                )
                y1, physics1 = self.split(mask, y, physics)

                # Forward pass
                x_hat = self.model(y1, physics1)

                # Output masking
                mask2 = getattr(physics, "mask", 1.0) - mask["mask"]
                out += self.split(mask2, x_hat)
                normaliser += mask2

            out[normaliser != 0] /= normaliser[normaliser != 0]

            return out

        def get_mask(self):
            if not isinstance(self.mask, torch.Tensor):
                raise ValueError(
                    "Mask not generated during forward pass - use model(y, physics, update_parameters=True)"
                )
            return self.mask


class Phase2PhaseLoss(SplittingLoss):
    r"""
    Phase2Phase loss for dynamic data.

    Implements dynamic measurement splitting loss from `Phase2Phase: Respiratory Motion-Resolved Reconstruction of Free-Breathing Magnetic Resonance Imaging Using Deep Learning Without a Ground Truth for Improved Liver Imaging <https://journals.lww.com/investigativeradiology/abstract/2021/12000/phase2phase__respiratory_motion_resolved.4.aspx>`_
    for free-breathing MRI.
    This is a special (temporal) case of the generic splitting loss: see :class:`deepinv.loss.SplittingLoss` for more details.

    Splits the dynamic measurements into even time frames ("phases") at model input and odd phases to use for constructing the loss.
    Equally, the physics mask (if it exists) is split as well: the even phases are used for the model (e.g. for data consistency in an unrolled network) and odd phases are used for the reference.
    At test time, the full input is passed through the network.

    .. warning::

        The model should be adapted before training using the method :func:`adapt_model <deepinv.loss.SplittingLoss.adapt_model>`
        to include the splitting mechanism at the input.

    .. warning::

        Must only be used for dynamic or sequential measurements, i.e. where data :math:`y` and ``physics.mask`` (if it exists) are of 5D shape (B, C, T, H, W).

    .. note::

        Phase2Phase can be used to reconstruct video sequences by setting ``dynamic_model=True`` and using physics :class:`deepinv.physics.DynamicMRI`.
        It can also be used to reconstructs **static** images, where the k-space measurements is a time-sequence,
        where each time step (phase) consists of sampled spokes such that the whole measurement is a set of non-overlapping spokes.
        To do this, set ``dynamic_model=False`` and use physics :class:`deepinv.physics.SequentialMRI`. See below for example or :ref:`sphx_glr_auto_examples_self-supervised-learning_demo_artifact2artifact.py` for full MRI example.


    By default, the error is computed using the MSE metric, however any appropriate metric can be used.

    :param tuple[int] tensor_size: size of the tensor to be masked without batch dimension of shape (C, T, H, W)
    :param bool dynamic_model: set ``True`` if using with a model that inputs and outputs time-data i.e. ``x`` of shape (B,C,T,H,W). Set ``False`` if ``x`` are static images (B,C,H,W).

    :param Metric, torch.nn.Module metric: metric used for computing data consistency, which is set as the mean squared error by default.
    :param str, torch.device device: torch device.

    |sep|

    :Example:

        Dynamic MRI with Phase2Phase with a video network:

        >>> import torch
        >>> from deepinv.models import AutoEncoder, TimeAgnosticNet
        >>> from deepinv.physics import DynamicMRI, SequentialMRI
        >>> from deepinv.loss import Phase2PhaseLoss
        >>>
        >>> x = torch.rand((1, 2, 4, 4, 4)) # B, C, T, H, W
        >>> mask = torch.zeros((1, 2, 4, 4, 4))
        >>> mask[:, :, torch.arange(4), torch.arange(4) % 4, :] = 1 # Create time-varying mask
        >>>
        >>> physics = DynamicMRI(mask=mask)
        >>> loss = Phase2PhaseLoss((2, 4, 4, 4))
        >>> model = TimeAgnosticNet(AutoEncoder(32, 2, 2)) # Example video network
        >>> model = loss.adapt_model(model) # Adapt model to perform Phase2Phase
        >>>
        >>> y = physics(x)
        >>> x_net = model(y, physics, update_parameters=True) # save random mask in forward pass
        >>> l = loss(x_net, y, physics, model)
        >>> print(l.item() > 0)
        True

        Free-breathing MRI with Phase2Phase with an image network and sequential measurements:

        >>> physics = SequentialMRI(mask=mask) # mask is B, C, T, H, W
        >>> loss = Phase2PhaseLoss((2, 4, 4, 4), dynamic_model=False) # Process static images x
        >>>
        >>> model = AutoEncoder(32, 2, 2) # Example image reconstruction network
        >>> model = loss.adapt_model(model) # Adapt model to perform Phase2Phase
        >>>
        >>> x = torch.rand((1, 2, 4, 4)) # B, C, H, W
        >>> y = physics(x) # B, C, T, H, W
        >>> x_net = model(y, physics, update_parameters=True)
        >>> l = loss(x_net, y, physics, model)
        >>> print(l.item() > 0)
        True

    """

    def __init__(
        self,
        tensor_size: Tuple[int],
        dynamic_model: bool = True,
        metric: Union[Metric, torch.nn.Module] = torch.nn.MSELoss(),
        device="cpu",
    ):
        super().__init__()
        self.name = "phase2phase"
        self.tensor_size = tensor_size
        self.dynamic_model = dynamic_model
        self.metric = metric
        self.device = device
        self.mask_generator = Phase2PhaseSplittingMaskGenerator(
            tensor_size=self.tensor_size, device=self.device
        )
        if not self.dynamic_model:
            # Metric wrapper to flatten dynamic inputs
            class TimeAveragingMetric(TimeMixin, torch.nn.Module):
                def __init__(self, metric: torch.nn.Module):
                    super().__init__()
                    self.metric = metric

                def forward(self, estimate, target):
                    assert estimate.shape == target.shape
                    return self.metric.forward(
                        self.average(estimate), self.average(target)
                    )

            self.metric = TimeAveragingMetric(self.metric)

    @staticmethod
    def split(mask: torch.Tensor, y: torch.Tensor, physics: Optional[Physics] = None):
        r"""Override splitting to actually remove masked pixels. In Phase2Phase, this corresponds to masked phases (i.e. time steps).

        :param torch.Tensor mask: Phase2Phase mask
        :param torch.Tensor y: input data
        :param deepinv.physics.Physics physics: forward physics
        """
        y_split, physics_split = SplittingLoss.split(mask, y, physics=physics)

        if len(mask.shape) < 5 or len(y.shape) < 5 or len(physics_split.mask.shape) < 5:
            raise ValueError(
                "mask, y and physics.mask must be of shape (B, C, T, H, W)"
            )

        def remove_zeros(arr, mask):
            reducer = (
                (mask != 0)[:, [0]]
                .view(mask.shape[0], 1, mask.shape[2], -1)
                .any(dim=3, keepdim=True)
                .unsqueeze(-1)
                .expand_as(mask)
            )  # assume pixelwise i.e. no channel dim
            return arr[reducer].view(
                mask.shape[0], mask.shape[1], -1, mask.shape[3], mask.shape[4]
            )

        y_split_reduced = remove_zeros(y_split, mask)

        if physics is None:
            return y_split_reduced

        physics_split_reduced = deepcopy(physics_split)
        physics_split_reduced.update_parameters(
            mask=remove_zeros(physics_split.mask, mask)
        )

        return y_split_reduced, physics_split_reduced

    def adapt_model(
        self, model: Reconstructor, **kwargs
    ) -> SplittingLoss.SplittingModel:
        r"""
        Apply Phase2Phase splitting to model input. Also perform time-averaging if a static model is used.

        :param deepinv.models.Reconstructor, torch.nn.Module model: Reconstruction model.
        :return: (:class:`deepinv.loss.SplittingLoss.SplittingModel`) Model modified for evaluation.
        """

        class Phase2PhaseModel(self.SplittingModel):
            @staticmethod
            def split(
                mask: torch.Tensor, y: torch.Tensor, physics: Optional[Physics] = None
            ):
                return Phase2PhaseLoss.split(mask, y, physics)

        if any(isinstance(module, self.SplittingModel) for module in model.modules()):
            return model

        if not self.dynamic_model:
            model = TimeAveragingNet(model)

        model = Phase2PhaseModel(
            model,
            mask_generator=self.mask_generator,
            split_ratio=None,
            eval_n_samples=0,
            eval_split_input=False,
            eval_split_output=False,
            pixelwise=False,
        )

        return model


class Artifact2ArtifactLoss(Phase2PhaseLoss):
    r"""
    Artifact2Artifact loss for dynamic data.

    Implements dynamic measurement splitting loss from `RARE: Image Reconstruction using Deep Priors Learned without Ground Truth <https://arxiv.org/abs/1912.05854>`_
    for free-breathing MRI.
    This is a special case of the generic splitting loss: see :class:`deepinv.loss.SplittingLoss` for more details.

    At model input, choose a random time-chunk from the dynamic measurements ("Artifact..."), and another random chunk for constructing the loss ("...2Artifact").
    Equally, the physics mask (if it exists) is split as well: the input chunk is used for the model (e.g. for data consistency in an unrolled network) and the output chunk is used as the reference.
    At test time, the full input is passed through the network.
    Note this implementation performs a Monte-Carlo-style version where the network output is only compared to one other chunk per iteration.

    .. warning::

        The model should be adapted before training using the method :func:`adapt_model <deepinv.loss.SplittingLoss.adapt_model>`
        to include the splitting mechanism at the input.

    .. warning::

        Must only be used for dynamic or sequential measurements, i.e. where data :math:`y` and ``physics.mask`` (if it exists) are of 5D shape (B, C, T, H, W).

    .. note::

        Artifact2Artifact can be used to reconstruct video sequences by setting ``dynamic_model=True`` and using physics :class:`deepinv.physics.DynamicMRI`.
        It can also be used to reconstructs **static** images, where the k-space measurements is a time-sequence,
        where each time step (phase) consists of sampled spokes such that the whole measurement is a set of non-overlapping spokes.
        To do this, set ``dynamic_model=False`` and use physics :class:`deepinv.physics.SequentialMRI`. See below for example or :ref:`sphx_glr_auto_examples_self-supervised-learning_demo_artifact2artifact.py` for full MRI example.

    By default, the error is computed using the MSE metric, however any appropriate metric can be used.

    :param tuple[int] tensor_size: size of the tensor to be masked without batch dimension of shape (C, T, H, W)
    :param int, tuple[int] split_size: time-length of chunk. Must divide ``tensor_size[1]`` exactly. If ``tuple``, one is randomly selected each time.
    :param bool dynamic_model: set True if using with a model that inputs and outputs time-data i.e. x of shape (B,C,T,H,W). Set False if x are static images (B,C,H,W).
    :param Metric, torch.nn.Module metric: metric used for computing data consistency, which is set as the mean squared error by default.
    :param str, torch.device device: torch device.

    |sep|

    :Example:

        Dynamic MRI with Artifact2Artifact with a video network:

        >>> import torch
        >>> from deepinv.models import AutoEncoder, TimeAgnosticNet
        >>> from deepinv.physics import DynamicMRI, SequentialMRI
        >>> from deepinv.loss import Artifact2ArtifactLoss
        >>>
        >>> x = torch.rand((1, 2, 4, 4, 4)) # B, C, T, H, W
        >>> mask = torch.zeros((1, 2, 4, 4, 4))
        >>> mask[:, :, torch.arange(4), torch.arange(4) % 4, :] = 1 # Create time-varying mask
        >>>
        >>> physics = DynamicMRI(mask=mask)
        >>> loss = Artifact2ArtifactLoss((2, 4, 4, 4))
        >>> model = TimeAgnosticNet(AutoEncoder(32, 2, 2)) # Example video network
        >>> model = loss.adapt_model(model) # Adapt model to perform Artifact2Artifact
        >>>
        >>> y = physics(x)
        >>> x_net = model(y, physics, update_parameters=True) # save random mask in forward pass
        >>> l = loss(x_net, y, physics, model)
        >>> print(l.item() > 0)
        True

        Free-breathing MRI with Artifact2Artifact with an image network and sequential measurements:

        >>> physics = SequentialMRI(mask=mask) # mask is B, C, T, H, W
        >>> loss = Artifact2ArtifactLoss((2, 4, 4, 4), dynamic_model=False) # Process static images x
        >>>
        >>> model = AutoEncoder(32, 2, 2) # Example image reconstruction network
        >>> model = loss.adapt_model(model) # Adapt model to perform Artifact2Artifact
        >>>
        >>> x = torch.rand((1, 2, 4, 4)) # B, C, H, W
        >>> y = physics(x) # B, C, T, H, W
        >>> x_net = model(y, physics, update_parameters=True)
        >>> l = loss(x_net, y, physics, model)
        >>> print(l.item() > 0)
        True

    """

    def __init__(
        self,
        tensor_size: Tuple[int],
        split_size: Union[int, Tuple[int]] = 2,
        dynamic_model: bool = True,
        metric: Union[Metric, torch.nn.Module] = torch.nn.MSELoss(),
        device="cpu",
    ):
        super().__init__(
            tensor_size=tensor_size,
            dynamic_model=dynamic_model,
            metric=metric,
            device=device,
        )
        self.name = "artifact2artifact"
        self.mask_generator = Artifact2ArtifactSplittingMaskGenerator(
            tensor_size=self.tensor_size, split_size=split_size, device=self.device
        )

    def forward(self, x_net, y, physics, model, **kwargs):
        mask = model.get_mask() * getattr(physics, "mask", 1.0)

        # Create output mask by re-splitting leftover samples
        mask2 = self.mask_generator.step(
            y.size(0),
            input_mask=getattr(physics, "mask", 1.0) - mask,
            persist_prev=True,
        )["mask"]

        y2, physics2 = self.split(mask2, y, physics)

        loss_ms = self.metric(physics2.A(x_net), y2)

        return loss_ms / mask2.mean()


class Neighbor2Neighbor(Loss):
    r"""
    Neighbor2Neighbor loss.

    Splits the noisy measurements using two masks :math:`A_1` and :math:`A_2`, each choosing a different neighboring
    map (see details in `"Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images"
    <https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_Neighbor2Neighbor_Self-Supervised_Denoising_From_Single_Noisy_Images_CVPR_2021_paper.pdf>`_).

    The self-supervised loss is computed as:

    .. math::

        \| A_2 y - R(A_1 y)\|^2 + \gamma \| A_2 y - R(A_1 y) - (A_2 R(y) - A_1 R(y))\|^2

    where :math:`R` is the trainable denoiser network, :math:`\gamma>0` is a regularization parameter
    and no gradient is propagated when computing :math:`R(y)`.

    By default, the error is computed using the MSE metric, however any other metric (e.g., :math:`\ell_1`)
    can be used as well.

    The code has been adapted from the repository https://github.com/TaoHuang2018/Neighbor2Neighbor.

    :param Metric, torch.nn.Module metric: metric used for computing data consistency,
        which is set as the mean squared error by default.
    :param float gamma: regularization parameter :math:`\gamma`.
    """

    def __init__(
        self, metric: Union[Metric, torch.nn.Module] = torch.nn.MSELoss(), gamma=2.0
    ):
        super().__init__()
        self.name = "neigh2neigh"
        self.metric = metric
        self.gamma = gamma

    def space_to_depth(self, x, block_size):
        n, c, h, w = x.size()
        unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
        return unfolded_x.view(n, c * block_size**2, h // block_size, w // block_size)

    def generate_mask_pair(self, img):
        # prepare masks (N x C x H/2 x W/2)
        n, c, h, w = img.shape
        mask1 = torch.zeros(
            size=(n * h // 2 * w // 2 * 4,), dtype=torch.bool, device=img.device
        )
        mask2 = torch.zeros(
            size=(n * h // 2 * w // 2 * 4,), dtype=torch.bool, device=img.device
        )
        # prepare random mask pairs
        idx_pair = torch.tensor(
            [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
            dtype=torch.int64,
            device=img.device,
        )
        rd_idx = torch.zeros(
            size=(n * h // 2 * w // 2,), dtype=torch.int64, device=img.device
        )
        torch.randint(low=0, high=8, size=(n * h // 2 * w // 2,), out=rd_idx)
        rd_pair_idx = idx_pair[rd_idx]
        rd_pair_idx += torch.arange(
            start=0,
            end=n * h // 2 * w // 2 * 4,
            step=4,
            dtype=torch.int64,
            device=img.device,
        ).reshape(-1, 1)
        # get masks
        mask1[rd_pair_idx[:, 0]] = 1
        mask2[rd_pair_idx[:, 1]] = 1
        return mask1, mask2

    def generate_subimages(self, img, mask):
        n, c, h, w = img.shape
        subimage = torch.zeros(
            n, c, h // 2, w // 2, dtype=img.dtype, layout=img.layout, device=img.device
        )
        # per channel
        for i in range(c):
            img_per_channel = self.space_to_depth(img[:, i : i + 1, :, :], block_size=2)
            img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
            subimage[:, i : i + 1, :, :] = (
                img_per_channel[mask].reshape(n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
            )
        return subimage

    def forward(self, y, physics, model, **kwargs):
        r"""
        Computes the neighbor2neighbor loss.


        :param torch.Tensor y: Measurements.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction function.
        :return: (:class:`torch.Tensor`) loss.
        """

        assert len(y.shape) == 4, "Input measurements should be images"
        assert (
            y.shape[2] % 2 == 0 and y.shape[3] % 2 == 0
        ), "Image dimensions should be even"

        mask1, mask2 = self.generate_mask_pair(y)

        y1 = self.generate_subimages(y, mask1)
        xhat1 = model(y1, physics)
        y2 = self.generate_subimages(y, mask2)

        xhat = model(y, physics).detach()
        y1_hat = self.generate_subimages(xhat, mask1)
        y2_hat = self.generate_subimages(xhat, mask2)

        loss_n2n = self.metric(xhat1, y2) + self.gamma * self.metric(
            xhat1 - y1_hat, y2 - y2_hat
        )

        return loss_n2n


if __name__ == "__main__":
    import deepinv as dinv
    import torch
    import numpy as np

    sigma = 0.1
    physics = dinv.physics.Denoising()
    physics.noise_model = dinv.physics.GaussianNoise(sigma)
    # choose a reconstruction architecture
    backbone = dinv.models.MedianFilter()
    f = dinv.models.ArtifactRemoval(backbone)
    # choose training losses
    split_ratio = 0.9
    loss = SplittingLoss(split_ratio=split_ratio)
    f = loss.adapt_model(f, eval_n_samples=2)  # important step!

    batch_size = 1
    imsize = (3, 128, 128)
    device = "cuda"

    x = torch.ones((batch_size,) + imsize, device=device)
    y = physics(x)

    x_net = f(y, physics)
    mse = dinv.metric.MSE()(physics.A(x), physics.A(x_net))
    split_loss = loss(y=y, x_net=x_net, physics=physics, model=f)

    print(
        f"split_ratio:{split_ratio:.2f}  mse: {mse:.2e}, split-loss: {split_loss:.2e}"
    )
    rel_error = (split_loss - mse).abs() / mse
    print(f"rel_error: {rel_error:.2f}")
