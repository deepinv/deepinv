from __future__ import annotations
from typing import TYPE_CHECKING
from functools import partial

import torch
from torch import Tensor
from torch.nn import MSELoss, L1Loss
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
    :param float max_pixel: maximum pixel value. If None, uses max pixel value of x.
    :param dict torchmetric_kwargs: kwargs for torchmetrics SSIM as dict. See https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param bool train_loss: use metric as a training loss, by returning one minus the metric.
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2``normalizes by L2 spatial norm, ``min_max`` normalizes by min and max of each input.
    """

    def __init__(
        self, multiscale=False, max_pixel=1.0, torchmetric_kwargs: dict = {}, **kwargs
    ):
        super().__init__(**kwargs)
        self.ssim = (
            multiscale_structural_similarity_index_measure
            if multiscale
            else structural_similarity_index_measure
        )
        self.torchmetric_kwargs = torchmetric_kwargs
        self.max_pixel = max_pixel
        self.lower_better = False

    def invert_metric(self, m):
        return 1.0 - m

    def metric(self, x_net, x, *args, **kwargs):
        max_pixel = self.max_pixel if self.max_pixel is not None else x.max()
        return self.ssim(
            x_net, x, data_range=max_pixel, reduction="none", **self.torchmetric_kwargs
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

    :param float max_pixel: maximum pixel value. If None, uses max pixel value of x.
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2``normalizes by L2 spatial norm, ``min_max`` normalizes by min and max of each input.
    """

    def __init__(self, max_pixel=1, **kwargs):
        super().__init__(**kwargs)
        self.max_pixel = max_pixel
        self.lower_better = False

    def metric(self, x_net, x, *args, **kwargs):
        max_pixel = self.max_pixel if self.max_pixel is not None else x.max()
        return cal_psnr(x_net, x, max_pixel=max_pixel)


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

    Otherwise, it is the one-sided error https://ieeexplore.ieee.org/abstract/document/6418031/, defined as
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
            QNR requires knowledge of ``y`` and ``physics``, which is not standard. In order to use QNR with
            :class:`deepinv.Trainer`, you will have to override the ``compute_metrics`` method to
            pass ``y,physics`` into the metric.

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
