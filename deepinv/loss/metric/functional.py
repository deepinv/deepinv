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


def signal_noise_ratio(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""
    Compute the signal-to-noise ratio (SNR)

    The signal-to-noise ratio (in dB) associated to a ground truth signal :math:`x` and a noisy estimate :math:`\hat{x} = \inverse{y}` is defined by

    .. math::

        \mathrm{SNR} = 10 \log_{10} \left( \frac{\|x\|_2^2}{\|x - y\|_2^2} \right).

    .. note::

        The input is assumed to be batched and the SNR is computed for each element independently.

    :param torch.Tensor preds: The noisy signal.
    :param torch.Tensor target: The reference signal.
    :return: (torch.Tensor) The SNR value in decibels (dB).
    """
    noise = x_hat - x
    # For a more efficient implementation, we compute the SNR from the signal
    # and noise L2 norms instead of their powers.
    signal_norm = torch.linalg.vector_norm(x.flatten(1, -1), ord=2, dim=1)
    noise_norm = torch.linalg.vector_norm(noise.flatten(1, -1), ord=2, dim=1)
    sqrt_snr = signal_norm / noise_norm
    # The factor 20 instead of 10 comes from the square root.
    return 20 * torch.log10(sqrt_snr)


def cal_mse(a, b):
    """Computes the mean squared error (MSE)"""
    return (a - b).abs().pow(2).mean(dim=tuple(range(1, a.ndim)), keepdim=False)


def cal_mae(a, b):
    """Computes the mean absolute error (MAE)"""
    return (a - b).abs().mean(dim=tuple(range(1, a.ndim)), keepdim=False)
