import torch
from torch.nn import MSELoss, L1Loss
from torchmetrics.functional import (
    structural_similarity_index_measure,
    multiscale_structural_similarity_index_measure,
)

from deepinv.loss.metric.metric import Metric
from deepinv.loss.metric.functional import cal_mse, cal_psnr, cal_mae


class MAE(Metric):
    r"""
    Mean Absolute Error metric.

    See docs for ``forward()`` below for more details.

    .. note::

        :class:`deepinv.metric.MAE` is functionally equivalent to :class:`torch.nn.L1Loss` when ``reduction='mean'`` or ``reduction='sum'``,
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
    :param bool train_loss: use metric as a training loss, by returning one minus the metric. If lower is better, does nothing.
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2``normalizes by L2 spatial norm, ``min_max`` normalizes by min and max of each input.
    """

    def metric(self, x_net, x, *args, **kwargs):
        return cal_mae(x_net, x)


class MSE(Metric):
    r"""
    Mean Squared Error metric.

    See docs for ``forward()`` below for more details.

    .. note::

        :class:`deepinv.metric.MSE` is functionally equivalent to :class:`torch.nn.MSELoss` when ``reduction='mean'`` or ``reduction='sum'``,
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
    :param bool train_loss: use metric as a training loss, by returning one minus the metric. If lower is better, does nothing.
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2``normalizes by L2 spatial norm, ``min_max`` normalizes by min and max of each input.
    """

    def metric(self, x_net, x, *args, **kwargs):
        return cal_mse(x_net, x)


class NMSE(MSE):
    r"""
    Normalised Mean Squared Error metric.

    Normalises MSE by the L2 norm of the ground truth ``x``.

    See docs for ``forward()`` below for more details.

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

    See https://en.wikipedia.org/wiki/Structural_similarity for more information.

    To set the max pixel on the fly (as is the case in `fastMRI evaluation code <https://github.com/facebookresearch/fastMRI/blob/main/banding_removal/fastmri/common/evaluate.py>`_), set ``max_pixel=None``.

    See docs for ``forward()`` below for more details.

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

    If the tensors have size (N, C, H, W), then the PSNR is computed as

    .. math::
        \text{PSNR} = \frac{20}{N} \log_{10} \frac{\text{MAX}_I}{\sqrt{\|\hat{x}-x\|^2_2 / (CHW) }}

    where :math:`\text{MAX}_I` is the maximum possible pixel value of the image (e.g. 1.0 for a
    normalized image), and :math:`\hat{x}` and :math:`x` are the estimate (``x_net``) and reference images (``x``).

    To set the max pixel on the fly (as is the case in `fastMRI evaluation code <https://github.com/facebookresearch/fastMRI/blob/main/banding_removal/fastmri/common/evaluate.py>`_), set ``max_pixel=None``.

    See docs for ``forward()`` below for more details.

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

    def __init__(self, max_pixel=1, normalize=False, **kwargs):
        super().__init__(**kwargs)
        self.max_pixel = max_pixel
        self.lower_better = False

    def metric(self, x_net, x, *args, **kwargs):
        max_pixel = self.max_pixel if self.max_pixel is not None else x.max()
        return cal_psnr(x_net, x, max_pixel=max_pixel)


class L1L2(Metric):
    r"""
    Combined L2 and L1 metric.

    Calculates L2 distance (i.e. MSE) + L1 (i.e. MAE) distance.

    See docs for ``forward()`` below for more details.

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

    If ``onesided=False`` then the metric is defined as
    :math:`d(x,y)=\|x-y\|_p^p`.

    Otherwise, it is the one-sided error https://ieeexplore.ieee.org/abstract/document/6418031/, defined as
    :math:`d(x,y)= \|\max(x\circ y) \|_p^p`. where :math:`\circ` denotes element-wise multiplication.

    See docs for ``forward()`` below for more details.

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
