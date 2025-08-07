from __future__ import annotations
from typing import TYPE_CHECKING
from functools import partial

import torch
from torch import Tensor
from torchmetrics.functional.image import (
    structural_similarity_index_measure,
    multiscale_structural_similarity_index_measure,
    spectral_angle_mapper,
    error_relative_global_dimensionless_synthesis,
)

from deepinv.loss.metric.metric import Metric
from deepinv.loss.metric.functional import cal_mse, cal_psnr, cal_mae

if TYPE_CHECKING:
    from deepinv.physics.remote_sensing import Pansharpen
    from deepinv.utils.tensorlist import TensorList


class MAE(Metric):
    r"""
    Mean Absolute Error metric.

    Calculates :math:`\text{MAE}(\hat{x},x)` where :math:`\hat{x}=\inverse{y}`.

    .. note::

        By default, no reduction is performed in the batch dimension.

    .. note::

        :class:`deepinv.loss.metric.MAE` is functionally equivalent to :class:`torch.nn.L1Loss` when ``reduction='mean'`` or ``reduction='sum'``,
        but when ``reduction=None`` our MAE reduces over all dims except batch dim (same behaviour as ``torchmetrics``) whereas ``L1Loss`` does not perform any reduction.

    :Example:

    >>> import torch
    >>> from deepinv.loss.metric import MAE
    >>> m = MAE()
    >>> x_net = x = torch.ones(3, 2, 8, 8) # B,C,H,W
    >>> m(x_net, x)
    tensor([0., 0., 0.])

    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2``normalizes by L2 spatial norm, ``min_max`` normalizes by min and max of each input.
    """

    def metric(self, x_net, x, *args, **kwargs):
        return cal_mae(x_net, x)


class MSE(Metric):
    r"""
    Mean Squared Error metric.

    Calculates :math:`\text{MSE}(\hat{x},x)` where :math:`\hat{x}=\inverse{y}`.

    .. note::

        By default, no reduction is performed in the batch dimension.

    .. note::

        :class:`deepinv.loss.metric.MSE` is functionally equivalent to :class:`torch.nn.MSELoss` when ``reduction='mean'`` or ``reduction='sum'``,
        but when ``reduction=None`` our MSE reduces over all dims except batch dim (same behaviour as ``torchmetrics``) whereas ``MSELoss`` does not perform any reduction.

    :Example:

    >>> import torch
    >>> from deepinv.loss.metric import MSE
    >>> m = MSE()
    >>> x_net = x = torch.ones(3, 2, 8, 8) # B,C,H,W
    >>> m(x_net, x)
    tensor([0., 0., 0.])

    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2``normalizes by L2 spatial norm, ``min_max`` normalizes by min and max of each input.
    """

    def metric(self, x_net, x, *args, **kwargs):
        return cal_mse(x_net, x)


class NMSE(MSE):
    r"""
    Normalised Mean Squared Error metric.

    Calculates :math:`\text{NMSE}(\hat{x},x)` where :math:`\hat{x}=\inverse{y}`.
    Normalises MSE by the L2 norm of the ground truth ``x``.

    .. note::

        By default, no reduction is performed in the batch dimension.

    :Example:

    >>> import torch
    >>> from deepinv.loss.metric import NMSE
    >>> m = NMSE()
    >>> x_net = x = torch.ones(3, 2, 8, 8) # B,C,H,W
    >>> m(x_net, x)
    tensor([0., 0., 0.])

    :param str method: normalisation method. Currently only supports ``l2``.
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2``normalizes by L2 spatial norm, ``min_max`` normalizes by min and max of each input.
    """

    def __init__(self, method="l2", **kwargs):
        super().__init__(**kwargs)
        self.method = method
        if self.method not in ("l2",):
            raise ValueError("method must be l2.")

    def metric(self, x_net, x, *args, **kwargs):
        if self.method == "l2":
            norm = cal_mse(x, 0)
        return cal_mse(x_net, x) / norm


class SSIM(Metric):
    r"""
    Structural Similarity Index (SSIM) metric using torchmetrics.

    Calculates :math:`\text{SSIM}(\hat{x},x)` where :math:`\hat{x}=\inverse{y}`.
    See https://en.wikipedia.org/wiki/Structural_similarity for more information.

    To set the max pixel on the fly (as is the case in `fastMRI evaluation code <https://github.com/facebookresearch/fastMRI/blob/main/banding_removal/fastmri/common/evaluate.py>`_), set ``max_pixel=None``.

    .. note::

        By default, no reduction is performed in the batch dimension.

    :Example:

    >>> import torch
    >>> from deepinv.loss.metric import SSIM
    >>> m = SSIM()
    >>> x_net = x = torch.ones(3, 2, 32, 32) # B,C,H,W
    >>> m(x_net, x)
    tensor([1., 1., 1.])

    :param bool multiscale: if ``True``, computes the multiscale SSIM. Default: ``False``.
    :param float max_pixel: maximum pixel value. If None, uses max pixel value of the ground truth image x.
    :param float min_pixel: minimum pixel value. If None, uses min pixel value of the ground truth image x.
    :param dict torchmetric_kwargs: kwargs for torchmetrics SSIM as dict. See https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param bool train_loss: use metric as a training loss, by returning one minus the metric.
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2``normalizes by L2 spatial norm, ``min_max`` normalizes by min and max of each input.
    """

    def __init__(
        self,
        multiscale=False,
        max_pixel=1.0,
        min_pixel=0.0,
        torchmetric_kwargs: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ssim = (
            multiscale_structural_similarity_index_measure
            if multiscale
            else structural_similarity_index_measure
        )
        self.torchmetric_kwargs = torchmetric_kwargs
        self.max_pixel = max_pixel
        self.min_pixel = min_pixel
        self.lower_better = False

    def invert_metric(self, m):
        return 1.0 - m

    def metric(self, x_net, x, *args, **kwargs):
        max_pixel = (
            self.max_pixel
            if self.max_pixel is not None
            else x.amax(dim=tuple(range(1, x.ndim)))
        )
        min_pixel = (
            self.min_pixel
            if self.min_pixel is not None
            else x.amin(dim=tuple(range(1, x.ndim)))
        )
        data_range = max_pixel - min_pixel
        if self.max_pixel is None or self.min_pixel is None:
            return torch.cat(
                [
                    self.ssim(
                        x_net[i : i + 1],
                        x[i : i + 1],
                        data_range=data_range[i].item(),
                        reduction="none",
                        **self.torchmetric_kwargs,
                    )
                    for i in range(x.shape[0])
                ],
                0,
            )
        return self.ssim(
            x_net,
            x,
            data_range=data_range,
            reduction="none",
            **self.torchmetric_kwargs,
        )


class PSNR(Metric):
    r"""
    Peak Signal-to-Noise Ratio (PSNR) metric.

    Calculates :math:`\text{PSNR}(\hat{x},x)` where :math:`\hat{x}=\inverse{y}`.
    If the tensors have size ``(B, C, H, W)``, then the PSNR is computed as

    .. math::
        \text{PSNR} = \frac{20}{B} \log_{10} \frac{\text{MAX}_I}{\sqrt{\|\hat{x}-x\|^2_2 / (CHW) }}

    where :math:`\text{MAX}_I` is the maximum possible pixel value of the image (e.g. 1.0 for a
    normalized image).

    To set the max pixel on the fly (as is the case in `fastMRI evaluation code <https://github.com/facebookresearch/fastMRI/blob/main/banding_removal/fastmri/common/evaluate.py>`_), set ``max_pixel=None``.

    .. note::

        By default, no reduction is performed in the batch dimension.

    :Example:

    >>> import torch
    >>> from deepinv.loss.metric import PSNR
    >>> m = PSNR()
    >>> x_net = x = torch.ones(3, 2, 8, 8) # B,C,H,W
    >>> m(x_net, x)
    tensor([80., 80., 80.])

    :param float max_pixel: maximum pixel value. If None, uses max pixel value of the ground truth image x.
    :param float min_pixel: minimum pixel value. If None, uses min pixel value of the ground truth image x.
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2``normalizes by L2 spatial norm, ``min_max`` normalizes by min and max of each input.
    """

    def __init__(self, max_pixel=1, min_pixel=0, **kwargs):
        super().__init__(**kwargs)
        self.max_pixel = max_pixel
        self.min_pixel = min_pixel
        self.lower_better = False

    def metric(self, x_net, x, *args, **kwargs):
        max_pixel = (
            self.max_pixel
            if self.max_pixel is not None
            else x.amax(dim=tuple(range(1, x.ndim)))
        )
        min_pixel = (
            self.min_pixel
            if self.min_pixel is not None
            else x.amin(dim=tuple(range(1, x.ndim)))
        )
        return cal_psnr(x_net, x, max_pixel=max_pixel, min_pixel=min_pixel)


class L1L2(Metric):
    r"""
    Combined L2 and L1 metric.

    Calculates L2 distance (i.e. MSE) + L1 (i.e. MAE) distance,
    :math:`\alpha L_1(\hat{x},x)+(1-\alpha)L_2(\hat{x},x)` where :math:`\hat{x}=\inverse{y}`.

    .. note::

        By default, no reduction is performed in the batch dimension.

    :Example:

    >>> import torch
    >>> from deepinv.loss.metric import L1L2
    >>> m = L1L2()
    >>> x_net = x = torch.ones(3, 2, 8, 8) # B,C,H,W
    >>> m(x_net, x)
    tensor([0., 0., 0.])

    :param float alpha: Weight between L2 and L1. Defaults to 0.5.
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2``normalizes by L2 spatial norm, ``min_max`` normalizes by min and max of each input.
    """

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.l1 = MAE().metric
        self.l2 = MSE().metric

    def metric(self, x_net, x, *args, **kwargs):
        l1 = self.l1(x_net, x)
        l2 = self.l2(x_net, x)
        return self.alpha * l1 + (1 - self.alpha) * l2


class LpNorm(Metric):
    r"""
    :math:`\ell_p` metric for :math:`p>0`.

    Calculates :math:`L_p(\hat{x},x)` where :math:`\hat{x}=\inverse{y}`.

    If ``onesided=False`` then the metric is defined as
    :math:`d(x,y)=\|x-y\|_p^p`.

    Otherwise, it is the one-sided error :footcite:t:`jacques2013robust`, defined as
    :math:`d(x,y)= \|\max(x\circ y) \|_p^p`. where :math:`\circ` denotes element-wise multiplication.

    .. note::

        By default, no reduction is performed in the batch dimension.

    :Example:

    >>> import torch
    >>> from deepinv.loss.metric import LpNorm
    >>> m = LpNorm(p=3) # L3 norm
    >>> x_net = x = torch.ones(3, 2, 8, 8) # B,C,H,W
    >>> m(x_net, x)
    tensor([0., 0., 0.])

    :param int p: order p of the Lp norm
    :param bool onesided: whether one-sided metric.
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2``normalizes by L2 spatial norm, ``min_max`` normalizes by min and max of each input.

    """

    def __init__(self, p=2, onesided=False, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.onesided = onesided

    def metric(self, x_net, x, *args, **kwargs):
        if self.onesided:
            diff = torch.maximum(x_net, x)
        else:
            diff = x_net - x

        return torch.norm(diff.view(diff.size(0), -1), p=self.p, dim=1).pow(self.p)


class QNR(Metric):
    r"""
    Quality with No Reference (QNR) metric for pansharpening.

    Calculates the no-reference :math:`\text{QNR}(\hat{x})` where :math:`\hat{x}=\inverse{y}`.

    QNR was proposed in Alparone et al., "Multispectral and Panchromatic Data Fusion Assessment Without Reference".

    Note we don't use the torchmetrics implementation.

    .. note::

        By default, no reduction is performed in the batch dimension.

    :Example:

    >>> import torch
    >>> from deepinv.loss.metric import QNR
    >>> from deepinv.physics import Pansharpen
    >>> m = QNR()
    >>> x = x_net = torch.rand(1, 3, 64, 64) # B,C,H,W
    >>> physics = Pansharpen((3, 64, 64))
    >>> y = physics(x) #[BCH'W', B1HW]
    >>> m(x_net=x_net, y=y, physics=physics) # doctest: +ELLIPSIS
    tensor([...])

    :param float alpha: weight for spectral quality, defaults to 1
    :param float beta: weight for structural quality, defaults to 1
    :param float p: power exponent for spectral D, defaults to 1
    :param float q: power exponent for structural D, defaults to 1
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param bool train_loss: use metric as a training loss, by returning one minus the metric.
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2``normalizes by L2 spatial norm, ``min_max`` normalizes by min and max of each input.
    """

    def __init__(
        self, alpha: float = 1, beta: float = 1, p: float = 1, q: float = 1, **kwargs
    ):
        super().__init__(**kwargs)
        self.alpha, self.beta, self.p, self.q = alpha, beta, p, q
        self.Q = partial(
            structural_similarity_index_measure, reduction="none"
        )  # Wang-Bovik
        self.lower_better = False

    def invert_metric(self, m):
        return 1.0 - m

    def D_lambda(self, hrms: Tensor, lrms: Tensor) -> float:
        """Calculate spectral distortion index."""
        _, n_bands, _, _ = hrms.shape
        out = 0
        for b in range(n_bands):
            for c in range(n_bands):
                out += (
                    abs(
                        self.Q(hrms[:, [b], :, :], hrms[:, [c], :, :])
                        - self.Q(lrms[:, [b], :, :], lrms[:, [c], :, :])
                    )
                    ** self.p
                )
        return (out / (n_bands * (n_bands - 1))) ** (1 / self.p)

    def D_s(self, hrms: Tensor, lrms: Tensor, pan: Tensor, pan_lr: Tensor) -> float:
        """Calculate spatial (or structural) distortion index."""
        _, n_bands, _, _ = hrms.shape
        out = 0
        for b in range(n_bands):
            out += (
                abs(
                    self.Q(hrms[:, [b], :, :], pan) - self.Q(lrms[:, [b], :, :], pan_lr)
                )
                ** self.q
            )
        return (out / n_bands) ** (1 / self.q)

    def metric(
        self,
        x_net: Tensor,
        x: None,
        y: TensorList,
        physics: Pansharpen,
        *args,
        **kwargs,
    ):
        r"""Calculate QNR on data.

        .. note::

            Note this does not require knowledge of ``x``, but it is included here as a placeholder.

        :param torch.Tensor x_net: Reconstructed high-res multispectral image :math:`\inverse{y}` of shape ``(B,C,H,W)``.
        :param torch.Tensor x: Placeholder, does nothing.
        :param deepinv.utils.TensorList y: pansharpening measurements generated from
            :class:`deepinv.physics.Pansharpen`, where y[0] is the low-res multispectral image of shape ``(B,C,H',W')``
            and y[1] is the high-res noisy panchromatic image of shape ``(B,1,H,W)``
        :param deepinv.physics.Pansharpen physics: pansharpening physics, used to calculate low-res pan image for QNR calculation.

        :return torch.Tensor: calculated metric, the tensor size might be ``(1,)`` or ``(B,)``.
        """

        if y is None:
            raise ValueError("QNR requires the measurements y to be passed.")

        if physics is None:
            raise ValueError("QNR requires the pansharpening physics to be passed.")

        lrms = y[0]
        pan = y[1]

        pan_lr = physics.downsampling.A(pan)

        d_lambda = self.D_lambda(x_net, lrms)
        d_s = self.D_s(x_net, lrms, pan, pan_lr)
        qnr = (1 - d_lambda) ** self.alpha * (1 - d_s) ** self.beta

        return qnr


class SpectralAngleMapper(Metric):
    r"""
    Spectral Angle Mapper (SAM).

    Calculates spectral similarity between estimated and target multispectral images.

    Wraps the ``torchmetrics`` `Spectral Angle Mapper <https://lightning.ai/docs/torchmetrics/stable/image/spectral_angle_mapper.html>`_ function.
    Note that our ``reduction`` parameter follows our uniform convention (see below).

    .. note::

        By default, no reduction is performed in the batch dimension.

    :Example:

    >>> import torch
    >>> from deepinv.loss.metric import SpectralAngleMapper
    >>> m = SpectralAngleMapper()
    >>> x_net = x = torch.ones(3, 2, 8, 8) # B,C,H,W
    >>> m(x_net, x)
    tensor([0., 0., 0.])

    :param bool train_loss: use metric as a training loss, by returning one minus the metric.
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2``normalizes by L2 spatial norm, ``min_max`` normalizes by min and max of each input.
    """

    def metric(self, x_net, x, *args, **kwargs):
        return spectral_angle_mapper(x_net, x, reduction="none").mean(
            dim=tuple(range(1, x.ndim - 1)), keepdim=False
        )


class ERGAS(Metric):
    r"""
    Error relative global dimensionless synthesis metric.

    Calculates the ERGAS metric on a multispectral image and a target.
    ERGAS is a popular metric for pan-sharpening of multispectral images.

    Wraps the ``torchmetrics`` `ERGAS <https://lightning.ai/docs/torchmetrics/stable/image/error_relative_global_dimensionless_synthesis.html>`_ function.
    Note that our ``reduction`` parameter follows our uniform convention (see below).

    .. note::

        By default, no reduction is performed in the batch dimension.

    :Example:

    >>> import torch
    >>> from deepinv.loss.metric import ERGAS
    >>> m = ERGAS(factor=4)
    >>> x_net = x = torch.ones(3, 2, 8, 8) # B,C,H,W
    >>> m(x_net, x)
    tensor([0., 0., 0.])

    :param int factor: pansharpening factor.
    :param bool train_loss: use metric as a training loss, by returning one minus the metric.
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2``normalizes by L2 spatial norm, ``min_max`` normalizes by min and max of each input.
    """

    def __init__(self, factor: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metric = self._metric = (
            lambda x_hat, x, *args, **kwargs: error_relative_global_dimensionless_synthesis(
                x_hat, x, ratio=factor, reduction="none"
            )
        )


class HaarPSI(Metric):
    r"""HaarPSI metric with tuned parameters.

    The metric was proposed by `Reisenhofer et al. <https://arxiv.org/abs/1607.06140>`_ and the parameters are taken from `Karner et al. <https://arxiv.org/abs/2410.24098>`_.
    The metric computes similarities in the Haar wavelet domain and it is shown to closely match human evaluation. See original papers for more details.
    The metric range is :math:`[0,1]`. The higher the metric, the better.

    Code is adapted from `this implementation <https://github.com/ideal-iqa/haarpsi-pytorch>`_ by Sören Dittmer, Clemens Karner and Anna Breger, adapted from David Neumann, adapted from Rafael Reisenhofer.

    .. note::

        Images must be scaled to :math:`[0,1]`. You can use `norm_inputs = clip` or `min_max` to achieve this.

    The parameters should be set as follows depending on the image domain:

    - **Natural images**: :math:`C=30,\alpha=4.2`.
    - **Medical images**: :math:`C=5,\alpha=4.9`.

    .. note::

        By default, no reduction is performed in the batch dimension.

    :Example:

    >>> import torch
    >>> from deepinv.loss.metric import HaarPSI
    >>> m = HaarPSI(norm_inputs="clip")
    >>> x_net = x = torch.ones(3, 1, 8, 8) # B,C,H,W
    >>> m(x_net, x)
    tensor([1.0000, 1.0000, 1.0000])

    :param float C: metric parameter :math:`C\in[5, 100]`.
    :param float alpha: metric paramter :math:`\alpha\in[2, 8]`.
    :param bool preprocess_with_subsampling: Determines if subsampling is performed.
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2``normalizes by L2 spatial norm, ``min_max`` normalizes by min and max of each input.
    """

    def __init__(
        self,
        C: float = 5.0,
        alpha: float = 4.9,
        preprocess_with_subsampling: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.C = C
        self.alpha = alpha
        self.lower_better = False
        self.preprocess_with_subsampling = preprocess_with_subsampling

    def metric(self, x_net: Tensor = None, x: Tensor = None, *args, **kwargs) -> Tensor:

        if (
            x.shape != x_net.shape
            or x.dtype != torch.float32
            or x.dtype != x_net.dtype
            or x.shape[1] not in {1, 3}
        ):
            raise ValueError(
                "x and x_net must be of same shape, of torch.float32 dtype, and with either 1 or 3 channels."
            )

        if (
            not torch.all(x_net <= 1)
            or not torch.all(0 <= x_net)
            or not torch.all(x <= 1)
            or not torch.all(0 <= x)
        ):
            raise ValueError("x and x_net must be in the range [0, 1]")

        x = 255 * x
        x_net = 255 * x_net

        is_color_image = x_net.shape[1] == 3

        if is_color_image:
            ref_y = (
                0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
            )
            deg_y = (
                0.299 * x_net[:, 0, :, :]
                + 0.587 * x_net[:, 1, :, :]
                + 0.114 * x_net[:, 2, :, :]
            )
            ref_i = (
                0.596 * x[:, 0, :, :] - 0.274 * x[:, 1, :, :] - 0.322 * x[:, 2, :, :]
            )
            deg_i = (
                0.596 * x_net[:, 0, :, :]
                - 0.274 * x_net[:, 1, :, :]
                - 0.322 * x_net[:, 2, :, :]
            )
            ref_q = (
                0.211 * x[:, 0, :, :] - 0.523 * x[:, 1, :, :] + 0.312 * x[:, 2, :, :]
            )
            deg_q = (
                0.211 * x_net[:, 0, :, :]
                - 0.523 * x_net[:, 1, :, :]
                + 0.312 * x_net[:, 2, :, :]
            )
            ref_y = ref_y.unsqueeze(1)
            deg_y = deg_y.unsqueeze(1)
            ref_i = ref_i.unsqueeze(1)
            deg_i = deg_i.unsqueeze(1)
            ref_q = ref_q.unsqueeze(1)
            deg_q = deg_q.unsqueeze(1)
        else:
            ref_y = x
            deg_y = x_net

        if self.preprocess_with_subsampling:
            ref_y = self._subsample(ref_y)
            deg_y = self._subsample(deg_y)
            if is_color_image:
                ref_i = self._subsample(ref_i)
                deg_i = self._subsample(deg_i)
                ref_q = self._subsample(ref_q)
                deg_q = self._subsample(deg_q)

        n_scales = 3
        coeffs_ref_y = self._haar_wavelet_decompose(
            ref_y, n_scales
        )  # n_scales x B x 1 x H x W
        coeffs_deg_y = self._haar_wavelet_decompose(deg_y, n_scales)
        if is_color_image:
            coefficients_ref_i = torch.abs(
                self._convolve2d(
                    ref_i,
                    torch.ones((2, 2), device=ref_i.device, dtype=ref_i.dtype) / 4.0,
                )
            )
            coefficients_deg_i = torch.abs(
                self._convolve2d(
                    deg_i,
                    torch.ones((2, 2), device=deg_i.device, dtype=deg_i.dtype) / 4.0,
                )
            )
            coefficients_ref_q = torch.abs(
                self._convolve2d(
                    ref_q,
                    torch.ones((2, 2), device=ref_q.device, dtype=ref_q.dtype) / 4.0,
                )
            )
            coefficients_deg_q = torch.abs(
                self._convolve2d(
                    deg_q,
                    torch.ones((2, 2), device=deg_q.device, dtype=deg_q.dtype) / 4.0,
                )
            )

        B, _, H, W = ref_y.shape
        n_channels = 3 if is_color_image else 2

        local_similarities = torch.zeros(n_channels, B, 1, H, W, device=ref_y.device)
        weights = torch.zeros(n_channels, B, 1, H, W, device=ref_y.device)

        for orientation in [0, 1]:
            weights[orientation] = self._get_weights_for_orientation(
                coeffs_deg_y, coeffs_ref_y, n_scales, orientation
            )
            local_similarities[orientation] = (
                self._get_local_similarity_for_orientation(
                    self.C, coeffs_deg_y, coeffs_ref_y, n_scales, orientation
                )
            )

        if is_color_image:
            similarity_i = (2 * coefficients_ref_i * coefficients_deg_i + self.C) / (
                coefficients_ref_i**2 + coefficients_deg_i**2 + self.C
            )
            similarity_q = (2 * coefficients_ref_q * coefficients_deg_q + self.C) / (
                coefficients_ref_q**2 + coefficients_deg_q**2 + self.C
            )
            local_similarities[2, :, :, :, :] = (similarity_i + similarity_q) / 2
            weights[2, :, :, :, :] = (
                weights[0, :, :, :, :] + weights[1, :, :, :, :]
            ) / 2

        pre_logit = torch.sum(
            torch.sigmoid(self.alpha * local_similarities) * weights, dim=(0, 3, 4)
        ) / torch.sum(weights, dim=(0, 3, 4))

        logit = lambda value, alpha: torch.log(value / (1 - value)) / alpha
        similarity = logit(pre_logit, self.alpha) ** 2
        return similarity[:, 0]

    def _get_local_similarity_for_orientation(
        self,
        C: float,
        coeffs_deg_y: Tensor,
        coeffs_ref_y: Tensor,
        n_scales: int,
        orientation: int,
    ) -> Tensor:
        """Helper function to get local similarity.

        :param float C: C parameter
        :param Tensor coeffs_deg_y: x_net wavelet coefficients.
        :param Tensor coeffs_ref_y: x wavelet coefficients.
        :param int n_scales: number of scales
        :param int orientation: orientation, 0 or 1
        :return: torch Tensor local similarity of shape B,1,H,W
        """
        coeffs_ref_y_magnitude = coeffs_ref_y.abs()[
            (orientation * n_scales, 1 + orientation * n_scales), :, :
        ]
        coeffs_deg_y_magnitude = coeffs_deg_y.abs()[
            (orientation * n_scales, 1 + orientation * n_scales), :, :
        ]

        frac = (2 * coeffs_ref_y_magnitude * coeffs_deg_y_magnitude + C) / (
            coeffs_ref_y_magnitude**2 + coeffs_deg_y_magnitude**2 + C
        )  # 2,B,1,H,W
        return (frac[0] + frac[1]) / 2

    def _get_weights_for_orientation(
        self,
        coeffs_deg_y: Tensor,
        coeffs_ref_y: Tensor,
        n_scales: int,
        orientation: int,
    ) -> Tensor:
        """Helper function to get weights.

        :param Tensor coeffs_deg_y: x_net wavelet coefficients.
        :param Tensor coeffs_ref_y: x wavelet coefficients.
        :param int n_scales: number of scales
        :param int orientation: orientation, 0 or 1
        :return: torch Tensor weights of shape B,1,H,W
        """
        return torch.maximum(
            coeffs_ref_y[2 + orientation * n_scales].abs(),
            coeffs_deg_y[2 + orientation * n_scales].abs(),
        )

    def _subsample(self, image: Tensor, factor: int = 2) -> Tensor:
        """Helper function to subsample image.

        :param Tensor image: input image
        :param int factor: downsampling factor, defaults to 2
        :return Tensor: subsampled image
        """
        kernel = torch.ones(factor, factor, device=image.device) / (factor**2)
        return self._convolve2d(image, kernel)[:, :, ::factor, ::factor]

    def _convolve2d(self, data: Tensor, kernel: Tensor) -> Tensor:
        """Helper function to perform 2D convolution

        :param Tensor data: input data
        :param Tensor kernel: convolution kernel
        :return Tensor: output image
        """
        res = torch.nn.functional.conv2d(
            torch.rot90(data, 2, [2, 3]),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=(kernel.shape[0] // 2, kernel.shape[1] // 2),
        )
        res = torch.nn.functional.interpolate(
            res, data.shape[-2:], mode="nearest", align_corners=None
        )
        res = torch.rot90(res, 2, [2, 3])
        return res

    def _get_haar_filter(self, scale: int, device: torch.device) -> Tensor:
        """Helper function to get Haar filter.

        :param int scale: filter scale
        :param torch.device device: torch device
        :return Tensor: Haar filter
        """
        haar_filter = 2**-scale * torch.ones(2**scale, 2**scale, device=device)
        haar_filter[: haar_filter.shape[0] // 2, :] = -haar_filter[
            : haar_filter.shape[0] // 2, :
        ]
        return haar_filter

    def _haar_wavelet_decompose(self, image: Tensor, number_of_scales: int) -> Tensor:
        """Decompose image in wavelet domain

        :param Tensor image: input image
        :param int number_of_scales: number of scales.
        :return Tensor: wavelet coefficients.
        """
        coefficients = torch.zeros(
            2 * number_of_scales, *image.shape, device=image.device
        )
        for scale in range(1, number_of_scales + 1):
            haar_filter = self._get_haar_filter(scale, image.device)
            coefficients[scale - 1] = self._convolve2d(image, haar_filter)
            coefficients[scale + number_of_scales - 1] = self._convolve2d(
                image, haar_filter.t()
            )
        return coefficients
