import torch
import torch.nn as nn

import pywt
import ptwt

from deepinv.models.utils import patchify


class WaveletNoiseEstimator(nn.Module):
    r"""
    Wavelet noise level estimator.

    This estimator was proposed in :footcite:t:`donoho1994ideal`.

    :Examples:

        >>> import torch
        >>> from deepinv.models import WaveletNoiseEstimator
        >>> # set seed
        >>> torch.manual_seed(0)
        >>> sigma_true = 0.1
        >>> noise = sigma_true * torch.randn(1, 1, 256, 256)
        >>> noise_estimator = WaveletNoiseEstimator()
        >>> sigma_est = noise_estimator(noise)
        >>> print(sigma_est)
        tensor([0.990])
    """

    def __init__(self):
        try:
            import pywt
            import ptwt
        except ImportError:  # pragma: no cover
            raise RuntimeError(
                "WaveletNoiseEstimator requires the Pytorch Wavelets package. Please install it (pip install ptwt)"
            )
        super(WaveletNoiseEstimator, self).__init__()

    @staticmethod
    def estimate_noise(im):
        dec = ptwt.wavedec2(im, pywt.Wavelet("db8"), level=1)
        l_coeffs = [dec[1][_].reshape(dec[1][_].shape[0], -1) for _ in range(3)]
        batched_coeffs = torch.hstack(l_coeffs)
        med = torch.median(batched_coeffs.abs(), dim=-1).values
        return med / 0.6745

    def forward(self, im):
        r"""
        Forward pass.

        :param torch.Tensor im: input image
        :return: estimated noise level
        """
        return self.estimate_noise(im)


class PatchCovarianceNoiseEstimator(nn.Module):
    r"""
    Noise level estimator based on eigenvalues of the covariance matrix.

    This method was initially proposed in :footcite:t:`chen2015efficient`.

    :Examples:
    >>> import torch
    >>> from deepinv.models import PatchCovarianceNoiseEstimator
    >>> # set seed
    >>> torch.manual_seed(0)
    >>> sigma_true = 0.1
    >>> noise = sigma_true * torch.randn(1, 1, 256, 256)
    >>> noise_estimator = PatchCovarianceNoiseEstimator()
    >>> sigma_est = noise_estimator(noise)
    >>> print(sigma_est)
    tensor([0.990])
    """

    def __init__(self):
        super(PatchCovarianceNoiseEstimator, self).__init__()

    @staticmethod
    def estimate_noise(im, pch_size=8):
        """
        Estimates noise level in image im.

        :param torch.Tensor im: input image
        :param (int, int) pch_size: patch size
        """
        # image to patch
        pch = patchify(
            im, pch_size, stride=3
        )  # C x pch_size x pch_size x num_pch tensor

        B, num_pch = pch.shape[0], pch.shape[-1]
        pch = pch.reshape(B, -1, num_pch)  # d x num_pch matrix
        d = pch.shape[1]

        mu = pch.mean(dim=-1, keepdim=True)  # B x d x 1
        X = pch - mu
        sigma_X = torch.bmm(X, X.transpose(-2, -1)) / num_pch
        sig_value = torch.linalg.eigvalsh(sigma_X)
        sig_value, _ = torch.sort(sig_value)

        noise_level = None
        for ii in range(-1, -d - 1, -1):
            tau = sig_value[..., :ii].mean()
            if torch.sum(sig_value[..., :ii] > tau) == torch.sum(
                sig_value[..., :ii] < tau
            ):
                noise_level = torch.sqrt(tau)
                return noise_level

        if noise_level is None:
            raise ValueError("Noise level estimation failed.")

    def forward(self, im):
        r"""
        Forward pass.

        :param torch.Tensor im: input image
        :return: estimated noise level
        """
        return self.estimate_noise(im)
