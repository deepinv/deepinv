import torch
import torch.nn as nn

from deepinv.models.utils import patchify


class WaveletNoiseEstimator(nn.Module):
    r"""
    Wavelet Gaussian noise level estimator.

    This estimator was proposed in :footcite:t:`donoho1994ideal`. It estimates the standard
    deviation of a Gaussian noise corrupted image. More precisely, given a noisy image
    :math:`y = x + n` where :math:`n \sim \mathcal{N}(0, \sigma^2)`, the noise level estimator computes an
    estimate of :math:`\sigma` as

    .. math::
        \hat{\sigma} = \frac{\text{median}(|w|)}{0.6745}

    where :math:`w` are the wavelet coefficients of the noisy image :math:`y` at the first level of decomposition.

    .. note::

        As noted by the authors, this estimator is an upper bound on the noise level, and may overestimate the true
        noise level in some cases, in particular if the SNR is high (i.e., the noise level is low compared to the
        signal level). In such cases, the estimator may be less accurate than the :class:`PatchCovarianceNoiseEstimator`
        which is based on the eigenvalues of the covariance matrix of image patches.

    .. warning::

        This estimator assumes that the noise in the corrupted image follows a Gaussian distribution.
        It may not perform well if the noise distribution deviates significantly from Gaussian, or if the image contains
        strong edges or textures that can affect the wavelet coefficients.

    .. warning::

        This model requires Pytorch Wavelets (``ptwt``) to be installed. It can be installed with
        ``pip install ptwt``.

    |sep|

    :Examples:

        >>> import deepinv as dinv
        >>> from deepinv.models import WaveletNoiseEstimator
        >>> rng = torch.Generator('cpu').manual_seed(0) # set seed
        >>> noise = dinv.physics.GaussianNoise(0.1, rng=rng)
        >>> noise_estimator = WaveletNoiseEstimator()
        >>> sigma_est = noise_estimator(noise)
        >>> print(sigma_est)
        tensor([0.1003])
    """

    def __init__(self):
        super(WaveletNoiseEstimator, self).__init__()

    @staticmethod
    def estimate_noise(x: torch.Tensor) -> torch.Tensor:
        r"""
        Estimates noise level in image im.

        :param torch.Tensor x: input image
        :return: (:class:`torch.Tensor`) estimated noise level
        """
        try:
            import pywt
            import ptwt
        except ImportError:
            raise ImportError(
                "pywt and ptwt are required for the WaveletNoiseEstimator. "
                "Please install them using `pip install pywt ptwt`."
            )
        dec = ptwt.wavedec2(x, pywt.Wavelet("db8"), level=1)
        l_coeffs = [dec[1][_].reshape(dec[1][_].shape[0], -1) for _ in range(3)]
        batched_coeffs = torch.hstack(l_coeffs)
        med = torch.median(batched_coeffs.abs(), dim=-1).values
        return med / 0.6745

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass.

        :param torch.Tensor x: input image
        :return: (:class: `torch.Tensor`) estimated noise level
        """
        return self.estimate_noise(x)


class PatchCovarianceNoiseEstimator(nn.Module):
    r"""
    Pach Covariance Gaussian noise level estimator.

    This method was initially proposed in :footcite:t:`chen2015efficient`. Given a noisy image :math:`y = x + n` where
    :math:`n \sim \mathcal{N}(0, \sigma^2)`, this estimator computes an estimate of :math:`\sigma` based on the
    eigenvalues of the covariance matrix of image patches.

    .. warning::

        This estimator assumes that the noise in the corrupted image follows a Gaussian distribution.
        It may not perform well if the noise distribution deviates significantly from Gaussian, or if the image lacks
        sufficient homogeneous regions for reliable patch statistics.

    |sep|

    :Examples:

    >>> import deepinv as dinv
    >>> from deepinv.models import PatchCovarianceNoiseEstimator
    >>> rng = torch.Generator('cpu').manual_seed(0) # set seed
    >>> noise = dinv.physics.GaussianNoise(0.1, rng=rng)
    >>> noise_estimator = PatchCovarianceNoiseEstimator()
    >>> sigma_est = noise_estimator(noise)
    >>> print(sigma_est)
    tensor([0.0995])
    """

    def __init__(self):
        super(PatchCovarianceNoiseEstimator, self).__init__()

    @staticmethod
    def estimate_noise(x: torch.Tensor, pch_size=8) -> torch.Tensor:
        """
        Estimates noise level in image im.

        :param torch.Tensor x: input image
        :param (int, int) pch_size: patch size
        :return: (:class:`torch.Tensor`) estimated noise level
        """
        # Convert image to patches
        pch = patchify(x, pch_size, stride=3)  # C x pch_size x pch_size x num_pch
        B, num_pch = pch.shape[0], pch.shape[-1]
        pch = pch.reshape(B, -1, num_pch)  # d x num_pch matrix
        d = pch.shape[1]

        # Compute covariance matrix eigenvalues
        mu = pch.mean(dim=-1, keepdim=True)  # B x d x 1
        X = pch - mu
        sigma_X = torch.bmm(X, X.transpose(-2, -1)) / num_pch
        sig_value = torch.linalg.eigvalsh(sigma_X)
        sig_value, _ = torch.sort(sig_value)

        # Track noise level and which samples have been solved
        noise_level = torch.zeros(B, device=x.device)
        found = torch.zeros(B, dtype=torch.bool, device=x.device)

        # Find tau where eigenvalues are balanced around it
        for ii in range(-1, -d - 1, -1):
            tau = sig_value[..., :ii].mean(dim=-1)
            counts_greater = torch.sum(sig_value[..., :ii] > tau.unsqueeze(-1), dim=-1)
            counts_less = torch.sum(sig_value[..., :ii] < tau.unsqueeze(-1), dim=-1)

            # Update samples where condition is met and not yet found
            mask = (counts_greater == counts_less) & ~found
            noise_level[mask] = torch.sqrt(tau[mask])
            found = found | mask

        if not torch.all(found):
            raise RuntimeError("Noise level estimation failed.")

        return noise_level

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass.

        :param torch.Tensor x: input image
        :return: (:class:`torch.Tensor`) estimated noise level
        """
        return self.estimate_noise(x)
