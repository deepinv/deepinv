import math
import torch
import torch.nn as nn
from torch import autograd as autograd
from deepinv.loss.loss import Loss
from deepinv.utils import cal_psnr


try:
    import pyiqa
except:
    pyiqa = ImportError("The pyiqa package is not installed.")


def check_pyiqa():
    if isinstance(pyiqa, ImportError):
        raise ImportError(
            "Metric not available. Please install the pyiqa package with `pip install pyiqa`."
        ) from pyiqa


class NIQE(Loss):
    r"""
    Natural Image Quality Evaluator (NIQE) metric.

    It is a no-reference image quality metric that estimates the quality of images.

    :param str device: device to use for the metric computation. Default: 'cpu'.
    """

    def __init__(self, device="cpu"):
        super().__init__()
        check_pyiqa()
        self.metric = pyiqa.create_metric("niqe").to(device)

    def forward(self, x_net, **kwargs):
        r"""
        Computes the NIQE metric (no reference).

        :param torch.Tensor x_net: input tensor.
        :return: torch.Tensor size (batch_size,).
        """
        return self.metric(x_net)


class LPIPS(Loss):
    r"""
    Learned Perceptual Image Patch Similarity (LPIPS) metric.

    Computes the perceptual similarity between two images, based on a pre-trained deep neural network.

    :param bool train: if ``True``, the metric is used for training. Default: ``False``.
    :param str device: device to use for the metric computation. Default: 'cpu'.
    """

    def __init__(self, train=False, device="cpu"):
        super().__init__()
        check_pyiqa()
        self.metric = pyiqa.create_metric("lpips").to(device)
        self.train = train

    def forward(self, x, x_net, **kwargs):
        r"""
        Computes the LPIPS metric.

        :param torch.Tensor x: reference image.
        :param torch.Tensor x_net: reconstructed image.
        :return: torch.Tensor size (batch_size,).
        """
        loss = self.metric(x, x_net)
        return (1 - loss) if self.train else loss


class SSIM(Loss):
    r"""
    Structural Similarity Index (SSIM) metric.

    See https://en.wikipedia.org/wiki/Structural_similarity for more information.

    :param bool multiscale: if ``True``, computes the multiscale SSIM. Default: ``False``.
    :param bool train: if ``True``, the metric is used for training. Default: ``False``.
    :param str device: device to use for the metric computation. Default: 'cpu'.
    """

    def __init__(self, multiscale=False, train=False, device="cpu"):
        super().__init__()
        check_pyiqa()
        if multiscale:
            self.metric = pyiqa.create_metric("ms-ssim").to(device)
        else:
            self.metric = pyiqa.create_metric("ssim").to(device)
        self.train = train

    def forward(self, x, x_net, **kwargs):
        r"""
        Computes the SSIM metric.

        :param torch.Tensor x: reference image.
        :param torch.Tensor x_net: reconstructed image.
        :return: torch.Tensor size (batch_size,).
        """
        loss = self.metric(x, x_net)
        return (1 - loss) if self.train else loss


class PSNR(Loss):
    r"""
    Peak Signal-to-Noise Ratio (PSNR) metric.

    If the tensors have size (N, C, H, W), then the PSNR is computed as

    .. math::
        \text{PSNR} = \frac{20}{N} \log_{10} \frac{\text{MAX}_I}{\sqrt{\|a- b\|^2_2 / (CHW) }}

    where :math:`\text{MAX}_I` is the maximum possible pixel value of the image (e.g. 1.0 for a
    normalized image), and :math:`a` and :math:`b` are the estimate and reference images.

    :param float max_pixel: maximum pixel value
    :param bool normalize: if ``True``, the estimate is normalized to have the same norm as the reference.
    """

    def __init__(self, max_pixel=1, normalize=False):
        super(PSNR, self).__init__()
        self.max_pixel = max_pixel
        self.normalize = normalize

    def forward(self, x_net, x, **kwargs):
        r"""
        Computes the PSNR metric.

        :param torch.Tensor x: reference image.
        :param torch.Tensor x_net: reconstructed image.
        :return: torch.Tensor size (batch_size,).
        """
        return cal_psnr(
            x_net, x, self.max_pixel, self.normalize, mean_batch=False, to_numpy=False
        )


class LpNorm(torch.nn.Module):
    r"""
    :math:`\ell_p` metric for :math:`p>0`.


    If ``onesided=False`` then the metric is defined as
    :math:`d(x,y)=\|x-y\|_p^p`.

    Otherwise, it is the one-sided error https://ieeexplore.ieee.org/abstract/document/6418031/, defined as
    :math:`d(x,y)= \|\max(x\circ y) \|_p^p`. where :math:`\circ` denotes element-wise multiplication.

    """

    def __init__(self, p=2, onesided=False):
        super().__init__()
        self.p = p
        self.onesided = onesided

    def forward(self, x_net, x, **kwargs):
        if self.onesided:
            return torch.nn.functional.relu(-x * x).flatten().pow(self.p).mean()
        else:
            return (x - x).flatten().abs().pow(self.p).mean()


def mse():
    return nn.MSELoss()


def l1():
    return nn.L1Loss()


def r1_penalty(real_pred, real_img):
    """R1 regularization for discriminator. The core idea is to
    penalize the gradient on real data alone: when the
    generator distribution produces the true data distribution
    and the discriminator is equal to 0 on the data manifold, the
    gradient penalty ensures that the discriminator cannot create
    a non-zero gradient orthogonal to the data manifold without
    suffering a loss in the GAN game.
    Ref:
    Eq. 9 in Which training methods for GANs do actually converge.
    """
    grad_real = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()


def gradient_penalty_loss(discriminator, real_data, fake_data, weight=None):
    """Calculate gradient penalty for wgan-gp.
    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        weight (Tensor): Weight tensor. Default: None.
    Returns:
        Tensor: A tensor for gradient penalty.
    """

    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

    # interpolate between real_data and fake_data
    interpolates = alpha * real_data + (1.0 - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    if weight is not None:
        gradients = gradients * weight

    gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    if weight is not None:
        gradients_penalty /= torch.mean(weight)

    return gradients_penalty
