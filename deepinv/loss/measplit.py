from __future__ import annotations
from typing import Optional, Union
from warnings import warn
import torch
from deepinv.physics import Inpainting, Physics
from deepinv.loss.loss import Loss
from deepinv.loss.metric.metric import Metric
from deepinv.physics.generator import BernoulliSplittingMaskGenerator
from deepinv.models.base import Reconstructor


class SplittingLoss(Loss):
    r"""
    Measurement splitting loss.

    Implements measurement splitting loss. Splits the measurement and forward operator :math:`A` (of size :math:`m`)
    into two smaller pairs  :math:`(y_1,A_1)` (of size :math:`m_1`) and  :math:`(y_2,A_2)` (of size :math:`m_2`) ,
    to compute the self-supervised loss:

    .. math::

        \frac{m}{m_2}\| y_2 - A_2 \inversef{y_1}{A_1}\|^2

    where :math:`R` is the trainable network, :math:`A_1 = M_1 A, A_2 = M_2 A`, and :math:`M_i` are randomly
    generated masks (i.e. diagonal matrices) such that :math:`M_1+M_2=\mathbb{I}_m`.

    See :ref:`sphx_glr_auto_examples_self-supervised-learning_demo_splitting_loss.py` for usage example.

    .. note::

        If the forward operator has its own subsampling mask :math:`M_{A}`, e.g. :class:`deepinv.physics.Inpainting`
        or :class:`deepinv.physics.MRI`,
        the splitting masks will be subsets of the physics' mask such that :math:`M_1+M_2=M_{A}`

    This loss was used for MRI in SSDU :footcite:t:`yaman2020self` for MRI, Noise2Inverse :footcite:t:`hendriksen2020noise2inverse` for CT, as well as numerous other papers.
    Note we implement the multi-mask strategy proposed by :footcite:t:`yaman2020self`.


    By default, the error is computed using the MSE metric, however any appropriate metric can be used.

    .. warning::

        The model should be adapted before training using the method :func:`adapt_model <deepinv.loss.SplittingLoss.adapt_model>`
        to include the splitting mechanism at the input.

    .. note::

        To obtain the best test performance, the trained model should be averaged at test time
        over multiple realizations of the splitting, i.e.
        :math:`\hat{x} = \frac{1}{N}\sum_{i=1}^N \inversef{y_1^{(i)}}{A_1^{(i)}}`. To disable this, set ``eval_n_samples=1``.

    .. note::

        To disable measurement splitting (and use the full input) at evaluation time, set ``eval_split_input=False``. This is done in SSDU :footcite:t:`yaman2020self`.

    .. seealso::

        :class:`deepinv.loss.mri.Artifact2ArtifactLoss`, :class:`deepinv.loss.mri.Phase2PhaseLoss`, :class:`deepinv.loss.mri.WeightedSplittingLoss`, :class:`deepinv.loss.mri.RobustSplittingLoss`
            Specialised splitting losses and their extensions for MRI applications.

    :param Metric, torch.nn.Module metric: metric used for computing data consistency, which is set as the mean squared error by default.
    :param float split_ratio: splitting ratio, should be between 0 and 1. The size of :math:`y_1` increases
        with the splitting ratio. Ignored if ``mask_generator`` passed.
    :param deepinv.physics.generator.BernoulliSplittingMaskGenerator, None mask_generator: function to generate the mask. If
        None, the :class:`deepinv.physics.generator.BernoulliSplittingMaskGenerator` is used, with the parameters ``split_ratio`` and ``pixelwise``.
    :param int eval_n_samples: Number of samples used for averaging at evaluation time. Must be greater than 0.
    :param bool eval_split_input: if True, perform input measurement splitting during evaluation. If False, use full measurement at eval (no MC samples are performed and eval_split_output will have no effect)
    :param bool eval_split_output: at evaluation time, pass the output through the output mask too.
        i.e. :math:`(\sum_{j=1}^N M_2^{(j)})^{-1} \sum_{i=1}^N M_2^{(i)} \inversef{y_1^{(i)}}{A_1^{(i)}}`.
        Only valid when :math:`y` is same domain (and dimension) as :math:`x`. Although better results may be observed on small datasets, more samples must be used for bigger images. Defaults to ``False``.
    :param bool pixelwise: if ``True``, create pixelwise splitting masks i.e. zero all channels simultaneously. Ignored if ``mask_generator`` passed.
    :param bool normalize_loss: whether to normalize loss by the target size

    |sep|

    :Example:

    >>> import torch
    >>> import deepinv as dinv
    >>> physics = dinv.physics.Inpainting(img_size=(1, 8, 8), mask=0.5)
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
        mask_generator: Optional[BernoulliSplittingMaskGenerator] = None,
        eval_n_samples: int = 5,
        eval_split_input: bool = True,
        eval_split_output: bool = False,
        pixelwise: bool = True,
        normalize_loss: bool = True,
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
        self.normalize_loss = normalize_loss

    @staticmethod
    def split(mask: torch.Tensor, y: torch.Tensor, physics: Optional[Physics] = None):
        r"""Perform splitting given mask

        :param torch.Tensor mask: splitting mask of shape (B,C,H,W)
        :param torch.Tensor y: input data of shape (B,C,...,H,W)
        :param deepinv.physics.Physics physics: physics to split, retaining its original noise model. If ``None``, only :math:`y` is split.
        """
        if y.shape[-2:] != mask.shape[-2:]:
            raise ValueError(
                f"y and mask must have same shape in last 2 dimensions, but y has {y.shape} and mask has {mask.shape}"
            )

        inp = Inpainting(
            y.size()[1:],
            mask=mask.view(
                *mask.shape[:2], *([1] * (y.ndim - mask.ndim)), *mask.shape[2:]
            ),
            device=y.device,
        )

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

        :param torch.Tensor x_net: reconstructions.
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

        l = self.metric(physics2.A(x_net), y2)

        return l / mask2.mean() if self.normalize_loss else l

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
        r"""
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
        def split(mask, y, physics=None):
            r"""Perform splitting given mask

            :param torch.Tensor mask: splitting mask
            :param torch.Tensor y: input data
            :param deepinv.physics.Physics physics: physics to split, retaining its original noise model. If ``None``, only :math:`y` is split.
            """
            return SplittingLoss.split(mask, y, physics)

        def forward(
            self, y: torch.Tensor, physics: Physics, update_parameters: bool = False
        ):
            """
            Adapted model forward pass for input splitting. During training, only one splitting realisation is performed for computational efficiency.
            """

            if self.mask_generator is None:
                warn("Mask generator not defined. Using new Bernoulli mask generator.")
                self.mask_generator = BernoulliSplittingMaskGenerator(
                    img_size=y.shape[1:],
                    split_ratio=self.split_ratio,
                    pixelwise=self.pixelwise,
                    device=y.device,
                )

            if self.mask_generator.img_size[-2:] != y.shape[-2:]:
                raise ValueError(
                    f"Mask generator should be same shape as y in last 2 dims, but mask has {self.mask_generator.img_size[-2:]} and y has {y.shape[-2:]}"
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
                )["mask"]
                y1, physics1 = self.split(mask, y, physics)

                # Forward pass
                x_hat = self.model(y1, physics1)

                # Output masking
                mask2 = getattr(physics, "mask", 1.0) - mask
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


class Neighbor2Neighbor(Loss):
    r"""
    Neighbor2Neighbor loss.

    Implements the self-supervised Neighbor2Neighbor loss :footcite:t:`huang2021neighbor2neighbor`.

    Splits the noisy measurements using two masks :math:`A_1` and :math:`A_2`, each choosing a different neighboring
    map (see details in :footcite:t:`huang2021neighbor2neighbor`). The self-supervised loss is computed as:

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
