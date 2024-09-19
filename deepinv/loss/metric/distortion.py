import torch
from torch.nn import MSELoss, L1Loss
from torchmetrics.functional import (
    structural_similarity_index_measure,
    multiscale_structural_similarity_index_measure,
)

from deepinv.loss.metric.metric import Metric
from deepinv.loss.metric.functional import cal_mse, cal_psnr


class MSE(Metric):
    r"""
    Mean Squared Error metric.
    """

    def metric(self, x_net, x, *args, **kwargs):
        return cal_mse(x_net, x)


class NMSE(MSE):
    r"""
    Normalised Mean Squared Error metric.

    Normalises MSE by the L2 norm of the ground truth ``x``.

    TODO add various other normalisation methods from torchmetrics. See [https://github.com/Lightning-AI/torchmetrics/pull/2442](https://github.com/Lightning-AI/torchmetrics/pull/2442)

    :param str method: normalisation method. Currently only supports ``l2``.
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

    :param bool multiscale: if ``True``, computes the multiscale SSIM. Default: ``False``.
    :param float max_pixel: maximum pixel value. If None, uses max pixel value of x.
    :param dict torchmetric_kwargs: kwargs for torchmetrics SSIM as dict. See https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html
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

    def metric(self, x_net, x, *args, **kwargs):
        max_pixel = self.max_pixel if self.max_pixel is not None else x.max()
        return self.ssim(x_net, x, data_range=max_pixel, **self.torchmetric_kwargs)


class PSNR(Metric):
    r"""
    Peak Signal-to-Noise Ratio (PSNR) metric.

    If the tensors have size (N, C, H, W), then the PSNR is computed as

    .. math::
        \text{PSNR} = \frac{20}{N} \log_{10} \frac{\text{MAX}_I}{\sqrt{\|a- b\|^2_2 / (CHW) }}

    where :math:`\text{MAX}_I` is the maximum possible pixel value of the image (e.g. 1.0 for a
    normalized image), and :math:`a` and :math:`b` are the estimate and reference images.

    To set the max pixel on the fly (as is the case in `fastMRI evaluation code <https://github.com/facebookresearch/fastMRI/blob/main/banding_removal/fastmri/common/evaluate.py>`_), set ``max_pixel=None``.

    :param float max_pixel: maximum pixel value. If None, uses max pixel value of x.
    :param bool normalize: if ``True``, the estimate is normalized to have the same norm as the reference.
    """

    def __init__(self, max_pixel=1, normalize=False, **kwargs):
        super().__init__(**kwargs)
        self.max_pixel = max_pixel
        self.normalize = normalize

    def metric(self, x_net, x, *args, **kwargs):
        r"""
        Computes the PSNR metric.

        :param torch.Tensor x: reference image.
        :param torch.Tensor x_net: reconstructed image.
        :return: torch.Tensor size (batch_size,).
        """
        max_pixel = self.max_pixel if self.max_pixel is not None else x.max()
        return cal_psnr(
            x_net, x, max_pixel, self.normalize, mean_batch=False, to_numpy=False
        )


class L1L2(Metric):
    r"""
    Combined L2 and L1 metric.

    Calculates L2 distance (i.e. MSE) + L1 (i.e. MAE) distance.

    :param float alpha: Weight between L2 and L1. Defaults to 0.5.
    """

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.l1 = L1Loss()
        self.l2 = MSELoss()

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

    :param int p: order p of the Lp norm
    :param bool onesided: whether one-sided metric.
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
