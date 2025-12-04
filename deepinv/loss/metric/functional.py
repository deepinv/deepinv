import torch


def norm(a):
    """Computes the L2 norm i.e. root sum of squares"""
    return torch.linalg.vector_norm(a, dim=(-1, -2), keepdim=True)


def cal_psnr(
    a: torch.Tensor,
    b: torch.Tensor,
    max_pixel: float = 1.0,
    min_pixel: float = 0.0,
):
    r"""
    Computes the peak signal-to-noise ratio (PSNR).

    :param torch.Tensor a: tensor estimate
    :param torch.Tensor b: tensor reference
    :param float max_pixel: maximum pixel value
    :param float min_pixel: minimum pixel value
    """
    with torch.no_grad():
        psnr = -10.0 * torch.log10(cal_mse(a, b) / (max_pixel - min_pixel) ** 2 + 1e-8)
    return psnr


def signal_noise_ratio(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r"""
    Compute the signal-to-noise ratio (SNR)

    For a reference signal :math:`x` corrupted by noise :math:`\varepsilon`

    .. math::

    y = x + \varepsilon,

    the signal-to-noise ratio expressed in dB is defiend as

    .. math::

        \mathrm{SNR} = 10 \log_{10} \left( \frac{\sum_{i=1}^n |x_i|^2}{\sum_{i=1}^n |x_i - y_i|^2} \right).

    .. note::

        The input is assumed to be batched and the SNR is computed for each element independently.

    :param torch.Tensor preds: The noisy signal.
    :param torch.Tensor target: The reference signal.
    :return: (torch.Tensor) The SNR value in decibels (dB).
    """
    noise = preds - target
    signal_power = target.abs().pow(2).flatten(1, -1).mean(dim=1)
    noise_power = noise.abs().pow(2).flatten(1, -1).mean(dim=1)
    snr = signal_power / noise_power
    return 10 * torch.log10(snr)


def cal_mse(a, b):
    """Computes the mean squared error (MSE)"""
    return (a - b).abs().pow(2).mean(dim=tuple(range(1, a.ndim)), keepdim=False)


def cal_mae(a, b):
    """Computes the mean absolute error (MAE)"""
    return (a - b).abs().mean(dim=tuple(range(1, a.ndim)), keepdim=False)
