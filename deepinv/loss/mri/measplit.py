from __future__ import annotations
from typing import Optional, Union
import torch
from deepinv.physics.forward import Physics
from deepinv.physics.noise import GaussianNoise
from deepinv.loss.metric.metric import Metric
from deepinv.physics.generator import (
    BaseMaskGenerator,
    BernoulliSplittingMaskGenerator,
    Phase2PhaseSplittingMaskGenerator,
    Artifact2ArtifactSplittingMaskGenerator,
)
from deepinv.models.dynamic import TimeAveragingNet
from deepinv.physics.time import TimeMixin
from deepinv.models.base import Reconstructor
from deepinv.loss.measplit import SplittingLoss
from deepinv.utils.decorators import _deprecated_alias


class WeightedSplittingLoss(SplittingLoss):
    r"""
    K-Weighted Splitting Loss

    Implements the K-weighted Noisier2Noise-SSDU loss from :footcite:t:`millard2023theoretical`.
    The loss is designed for problems where measurements are observed as :math:`y_i=M_iAx`,
    where :math:`M_i` is a random mask, such as in :class:`MRI <deepinv.physics.MRI>` where `A` is the Fourier transform.
    The loss is defined as follows, using notation from :class:`deepinv.loss.SplittingLoss`:

    .. math::

        \frac{m}{m_2}\| (1-\mathbf{K})^{-1/2} (y_2 - A_2 \inversef{y_1}{A_1})\|^2

    where :math:`\mathbf{K}` is derived from the probability density function (pdf) of the (original) acceleration mask and (further) splitting mask:

    .. math::

        \mathbf{K}=(\mathbb{I}_n-\tilde{\mathbf{P}}\mathbf{P})^{-1}(\mathbb{I}_n-\mathbf{P})

    and :math:`\mathbf{P}=\mathbb{E}[\mathbf{M}_i],\tilde{\mathbf{P}}=\mathbb{E}[\mathbf{M}_1]` i.e. the average imaging mask and splitting mask, respectively.
    At inference, the original whole measurement :math:`y` is used as input.

    .. note::

        To match the original paper, the loss should be used with the splitting mask :class:`deepinv.physics.generator.MultiplicativeSplittingMaskGenerator`
        where the input additional subsampling mask should be the same type as that used to generate the measurements.

        Note the method was originally proposed for accelerated MRI problems (where the measurements are generated via a mask generator).

        Note also that we assume that all masks are 1D mask in the image width dimension repeated in all other dimensions.

    :param deepinv.physics.generator.BernoulliSplittingMaskGenerator mask_generator: splitting mask generator for further subsampling.
    :param deepinv.physics.generator.BaseMaskGenerator physics_generator: original mask generator used to generate the measurements.
    :param float eps: small value to avoid division by zero.
    :param Metric, torch.nn.Module metric: metric used for computing data consistency, which is set as the mean squared error by default.

    |sep|

    :Example:

    >>> import torch
    >>> from deepinv.physics.generator import GaussianMaskGenerator, MultiplicativeSplittingMaskGenerator
    >>> from deepinv.loss.mri import WeightedSplittingLoss
    >>> physics_generator = GaussianMaskGenerator((128, 128), acceleration=4)
    >>> split_generator = GaussianMaskGenerator((128, 128), acceleration=2)
    >>> mask_generator = MultiplicativeSplittingMaskGenerator((1, 128, 128), split_generator)
    >>> loss = WeightedSplittingLoss(mask_generator, physics_generator)

    """

    class WeightedMetric(torch.nn.Module):
        """Wraps metric to apply weight on inputs

        :param torch.Tensor: loss weight.
        :param Metric, torch.nn.Module metric: loss metric.
        :param bool expand: whether expand weight to input dims
        """

        def __init__(
            self,
            weight: torch.Tensor,
            metric: Union[Metric, torch.nn.Module],
            expand: bool = True,
        ):
            super().__init__()
            self.weight = weight
            self.metric = metric
            self.expand = lambda w, y: w.expand_as(y) if expand else w

        def forward(self, y1, y2):
            """Weighted metric forward pass."""
            return self.metric(
                self.expand(self.weight, y1) * y1, self.expand(self.weight, y2) * y2
            )

    def __init__(
        self,
        mask_generator: BernoulliSplittingMaskGenerator,
        physics_generator: BaseMaskGenerator,
        eps: float = 1e-9,
        metric: Union[Metric, torch.nn.Module] = torch.nn.MSELoss(),
    ):

        super().__init__(eval_split_input=False, pixelwise=True)
        self.mask_generator = mask_generator
        self.physics_generator = physics_generator
        self.name = "WeightedSplitting"
        self.k = self.compute_k(eps=eps)
        self.weight = (1 - self.k).clamp(min=eps) ** (-0.5)
        self.metric = self.WeightedMetric(self.weight, metric)
        self.normalize_loss = False

    def compute_k(self, eps: float = 1e-9) -> torch.Tensor:
        """
        Compute K for K-weighted splitting loss where K is a diagonal matrix of shape (H, W).

        Estimates the 1D PDFs of the mask generators empirically.

        :param float eps: small value to avoid division by zero.
        """

        P = self.physics_generator.average()["mask"]
        P_tilde = self.mask_generator.average()["mask"]

        if P.shape != P_tilde.shape:
            raise ValueError(
                "physics_generator and mask_generator should produce same size masks."
            )

        # Reduce to 1D PDF in W dimension
        while len(P.shape) > 1:
            P, P_tilde = P[0], P_tilde[0]

        # makes sure P_tilde < 1
        P_tilde[P_tilde > (1 - eps)] = 1 - eps

        diag_1_minus_PtP = 1 - P_tilde * P
        diag_1_minus_PtP = diag_1_minus_PtP.clamp(min=eps)
        inv_diag_1_minus_PtP = 1 / diag_1_minus_PtP
        diag_1_minus_P = 1 - P

        # element-wise multiplication to get K
        k_weight = inv_diag_1_minus_PtP * diag_1_minus_P
        return k_weight.unsqueeze(0)


class RobustSplittingLoss(WeightedSplittingLoss):
    r"""
    Robust Weighted Splitting Loss

    Implements the Robust-SSDU loss from :footcite:t:`millard2024clean`.
    The loss is designed for problems where measurements are observed as :math:`y_i=M_iAx+\epsilon`,
    where :math:`M_i` is a random mask, such as in :class:`MRI <deepinv.physics.MRI>` where `A` is the Fourier transform,
    and :math:`\epsilon` is Gaussian noise.
    The loss is related to the :class:`deepinv.loss.mri.WeightedSplittingLoss` as follows:

    .. math::

        \mathcal{L}_\text{Robust-SSDU}=\mathcal{L}_\text{Weighted-SSDU}(\tilde{y};y) + \lVert(1+\frac{1}{\alpha^2}) M_1 M (\forw{\inverse{\tilde{y},A} - y}\rVert_2^2

    where :math:`\tilde{y}\sim\mathcal{N}(y,\alpha^2\sigma^2\mathbf{I})` is further noised (i.e. "noisier") measurement, and :math:`\alpha` is a hyperparameter.
    This is derived from Eqs. 34 & 35 of the paper :footcite:`millard2024clean`.
    At inference, the original measurement :math:`y` is used as input.

    .. note::

        See :class:`deepinv.loss.mri.WeightedSplittingLoss` on what is expected of the input measurements, and the `mask_generator`.

    :param deepinv.physics.generator.BernoulliSplittingMaskGenerator mask_generator: splitting mask generator for further subsampling.
    :param deepinv.physics.generator.BaseMaskGenerator physics_generator: original mask generator used to generate the measurements.
    :param deepinv.physics.NoiseModel noise_model: noise model for adding further noise, must be of same type as original measurement noise.
        Note this loss only supports :class:`deepinv.physics.GaussianNoise`.
    :param float alpha: hyperparameter controlling further noise std.
    :param float eps: small value to avoid division by zero.
    :param Metric, torch.nn.Module metric: metric used for computing data consistency, which is set as the mean squared error by default.
    """

    def __init__(
        self,
        mask_generator: BernoulliSplittingMaskGenerator,
        physics_generator: BaseMaskGenerator,
        noise_model: GaussianNoise = GaussianNoise(sigma=0.1),
        alpha: float = 0.75,
        eps: float = 1e-9,
        metric: Union[Metric, torch.nn.Module] = torch.nn.MSELoss(),
    ):
        super().__init__(mask_generator, physics_generator, eps=eps, metric=metric)
        self.alpha = alpha
        self.noise_model = noise_model
        self.noise_model.update_parameters(sigma=noise_model.sigma * alpha)

    def forward(self, x_net, y, physics, model, **kwargs):
        recon_loss = super().forward(x_net, y, physics, model, **kwargs)

        mask = model.get_mask() * getattr(physics, "mask", 1.0)  # M_\lambda\cap\omega
        n2n_metric = self.WeightedMetric(
            (1 + 1 / (self.alpha**2)) * mask, self.metric.metric, expand=False
        )

        return recon_loss + n2n_metric(physics.A(x_net), y)

    def adapt_model(self, model: torch.nn.Module) -> RobustSplittingModel:
        return (
            model
            if isinstance(model, self.RobustSplittingModel)
            else self.RobustSplittingModel(
                model, mask_generator=self.mask_generator, noise_model=self.noise_model
            )
        )

    class RobustSplittingModel(SplittingLoss.SplittingModel):
        def __init__(self, model, mask_generator, noise_model):
            super().__init__(
                model,
                split_ratio=None,
                mask_generator=mask_generator,
                eval_n_samples=1,
                eval_split_input=False,
                eval_split_output=False,
                pixelwise=True,
            )
            self.noise_model = noise_model

        def split(self, mask, y, physics=None):
            y1, physics1 = SplittingLoss.split(mask, y, physics)
            return (mask * self.noise_model(y1) if self.training else y1), physics1


class Phase2PhaseLoss(SplittingLoss):
    r"""
    Phase2Phase loss for dynamic data.

    Implements dynamic measurement splitting loss from :footcite:t:`eldeniz2021phase2phase` for free-breathing MRI.
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

    :param tuple[int] img_size: size of the tensor to be masked without batch dimension of shape (C, T, H, W)
    :param bool dynamic_model: set ``True`` if using with a model that inputs and outputs time-data i.e. ``x`` of shape (B,C,T,H,W). Set ``False`` if ``x`` are static images (B,C,H,W).

    :param Metric, torch.nn.Module metric: metric used for computing data consistency, which is set as the mean squared error by default.
    :param str, torch.device device: torch device.

    |sep|

    :Example:

        Dynamic MRI with Phase2Phase with a video network:

        >>> import torch
        >>> from deepinv.models import AutoEncoder, TimeAgnosticNet
        >>> from deepinv.physics import DynamicMRI, SequentialMRI
        >>> from deepinv.loss.mri import Phase2PhaseLoss
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

    @_deprecated_alias(tensor_size="img_size")
    def __init__(
        self,
        img_size: tuple[int],
        dynamic_model: bool = True,
        metric: Union[Metric, torch.nn.Module] = torch.nn.MSELoss(),
        device="cpu",
    ):
        super().__init__()
        self.name = "phase2phase"
        self.img_size = img_size
        self.dynamic_model = dynamic_model
        self.metric = metric
        self.device = device
        self.mask_generator = Phase2PhaseSplittingMaskGenerator(
            img_size=self.img_size, device=self.device
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

        physics_split_reduced = physics_split.clone()
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

    Implements dynamic measurement splitting loss from :footcite:t:`liu2020rare` for free-breathing MRI.
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

    :param tuple[int] img_size: size of the tensor to be masked without batch dimension of shape (C, T, H, W)
    :param int, tuple[int] split_size: time-length of chunk. Must divide ``img_size[1]`` exactly. If ``tuple``, one is randomly selected each time.
    :param bool dynamic_model: set True if using with a model that inputs and outputs time-data i.e. x of shape (B,C,T,H,W). Set False if x are static images (B,C,H,W).
    :param Metric, torch.nn.Module metric: metric used for computing data consistency, which is set as the mean squared error by default.
    :param str, torch.device device: torch device.

    |sep|

    :Example:

        Dynamic MRI with Artifact2Artifact with a video network:

        >>> import torch
        >>> from deepinv.models import AutoEncoder, TimeAgnosticNet
        >>> from deepinv.physics import DynamicMRI, SequentialMRI
        >>> from deepinv.loss.mri import Artifact2ArtifactLoss
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

    @_deprecated_alias(tensor_size="img_size")
    def __init__(
        self,
        img_size: tuple[int],
        split_size: Union[int, tuple[int]] = 2,
        dynamic_model: bool = True,
        metric: Union[Metric, torch.nn.Module] = torch.nn.MSELoss(),
        device="cpu",
    ):
        super().__init__(
            img_size=img_size,
            dynamic_model=dynamic_model,
            metric=metric,
            device=device,
        )
        self.name = "artifact2artifact"
        self.mask_generator = Artifact2ArtifactSplittingMaskGenerator(
            img_size=self.img_size, split_size=split_size, device=self.device
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
