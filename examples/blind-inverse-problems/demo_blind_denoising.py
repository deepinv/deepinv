r"""
Blind denoising & noise level estimation
========================================

This example focuses on blind image Gaussian denoising, i.e. the problem

.. math::

    y = x + \sigma n \quad n \sim \mathcal{N}(0, 1)

where :math:`\sigma` is unknown. In this example, we first propose to estimate the noise level with different approaches,
and then show general restoration models available in the library.

"""

import torch
import deepinv as dinv

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# Build a noisy image
# ~~~~~~~~~~~~~~~~~~~
#
# We load a noiseless image and generate a noisy (Gaussian) version of this image, with standard deviation that we will
# assume to be unknown. We set it to :math:`\sigma = 0.042` for this example.

x = dinv.utils.load_example("butterfly.png", device=device)

sigma_true = 0.042

y = x + sigma_true * torch.randn_like(x)

# %%
# A naive approach
# ~~~~~~~~~~~~~~~~
#
# A first naive approach to estimate :math:`sigma` consists in taking a patch of the image, removing the mean, and using
# the standard deviation of the resulting patch as an estimate of the noise level.

p = 50
y_patch = y[:, :, -p:, p - p // 2 : p + p // 2]  # extract a patch
std_naive = y_patch.std()

print("Naive noise level estimate: ", std_naive.item())


# %%
# Noise level estimators
# ~~~~~~~~~~~~~~~~~~~~~~
#
# A more advanced approach consists in performing the same approach as above, but in an appropriate domain.
# A good transform is the wavelet transform, where we can expect the noise to dominate high-frequency components.
# We can illustrate this as follows:

import ptwt
import pywt

coeffs = ptwt.wavedec2(y, pywt.Wavelet("db8"), mode="constant", level=1, axes=(-2, -1))

imgs = [coeffs[0], coeffs[1][0], coeffs[1][1], coeffs[1][2]]
titles = ["LF", "HF (horizontal)", "HF (vertical)", "HF (diagonal)"]

dinv.utils.plot_inset(
    img_list=imgs,
    titles=titles,
    suptitle="Wavelet decomposition of noisy image",
    extract_size=0.2,
    extract_loc=(0.7, 0.7),
    inset_size=0.5,
    figsize=(len(imgs) * 1.5, 2.5),
)

# %%
# We notice that the high-frequency components are mostly noise.
# We can thus use these components to estimate the noise level more robustly.
# This is implemented in :class:`deepinv.models.WaveletNoiseEstimator`. Under the hood, the estimator uses the
# Median Absolute Deviation (MAD) estimator on the wavelet high-frequency coefficients:
#
# .. math::
#        \qquad \hat{\sigma} = \frac{\mathrm{median}(|w|)}{0.6745},
#
# where :math:`w` are the high-frequency wavelet coefficients.
#

wavelet_estimator = dinv.models.WaveletNoiseEstimator()
sigma_wavelet = wavelet_estimator(y)
print("Wavelet-based noise level estimate: ", sigma_wavelet.item())

# %%
# We notice that this approach provides a signficantly better estimate of the noise level compared to the naive approach.
# However, it tends to slightly over-estimate the noise level in this example. As noted in the original paper, this
# is due to the presence of residual signal in the high-frequency wavelet coefficients (these are not only noise).
#
# Another approach is to use the eigenvalues of the covariance matrix of patches extracted from the noisy
# image. This is implemented in :class:`deepinv.models.PatchCovarianceNoiseEstimator`.
# The method was initially proposed in :footcite:t:`chen2015efficient`.

patch_cov_estimator = dinv.models.PatchCovarianceNoiseEstimator()
sigma_patch_cov = patch_cov_estimator(y)
print("Patch covariance-based noise level estimate: ", sigma_patch_cov.item())

# %%
# Blind denoising with estimated noise level
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Once we have estimated the noise level, we can use general denoising models available in the library.
# Here, we use the DRUNet model from :footcite:t:`zhang2021plug` that can handle a range of noise levels.

denoiser = dinv.models.DRUNet()

with torch.no_grad():
    denoised_naive = denoiser(y, sigma=std_naive)
    denoised_wavelet = denoiser(y, sigma=sigma_wavelet)
    denoised_patch_cov = denoiser(y, sigma=sigma_patch_cov)


metric = dinv.metric.PSNR()

psnr_noisy = metric(y, x).item()
psnr_naive = metric(denoised_naive, x).item()
psnr_wavelet = metric(denoised_wavelet, x).item()
psnr_patch_cov = metric(denoised_patch_cov, x).item()

dinv.utils.plot(
    {
        f"Noisy\n PSNR: {psnr_noisy:.2f} dB": y,
        f"Denoised (naive)\n PSNR: {psnr_naive:.2f} dB": denoised_naive,
        f"Denoised (wavelet)\n PSNR: {psnr_wavelet:.2f} dB": denoised_wavelet,
        f"Denoised (patch cov.)\n PSNR: {psnr_patch_cov:.2f} dB": denoised_patch_cov,
    },
    fontsize=9,
)

# %%
# Which noise level estimator is best?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This will depend on several parameters, e.g. image size, content and noise level.
# Above, the patch covariance estimator provides the best results. We can investigate the performance on the above image
# for different noise levels as follows:
#

list_sigmas = torch.logspace(-2, 0, steps=10)

estimate_errors = {
    "wavelet mean": [],
    "wavelet std": [],
    "patch_cov mean": [],
    "patch_cov std": [],
}

# run estimations for different noise levels, and average over 10 random seeds
for sigma in list_sigmas:

    sigma_wavelet, sigma_patch_cov = [], []

    for seed in range(10):
        torch.manual_seed(seed)
        y_ = x + sigma * torch.randn_like(x)

        sigma_wavelet.append(wavelet_estimator(y_))
        sigma_patch_cov.append(patch_cov_estimator(y_))

    sigma_wavelet = torch.stack(sigma_wavelet)
    sigma_patch_cov = torch.stack(sigma_patch_cov)

    estimate_errors["wavelet mean"].append((sigma_wavelet - sigma).abs().mean().item())
    estimate_errors["wavelet std"].append(sigma_wavelet.std().item())
    estimate_errors["patch_cov mean"].append(
        (sigma_patch_cov - sigma).abs().mean().item()
    )
    estimate_errors["patch_cov std"].append(sigma_patch_cov.std().item())

# next, plot the results in semilogx with error bars

import matplotlib.pyplot as plt

plt.figure()
plt.errorbar(
    list_sigmas.cpu(),
    estimate_errors["wavelet mean"],
    yerr=estimate_errors["wavelet std"],
    label="Wavelet-based estimator",
    fmt="-o",
)
plt.errorbar(
    list_sigmas.cpu(),
    estimate_errors["patch_cov mean"],
    yerr=estimate_errors["patch_cov std"],
    label="Patch covariance-based estimator",
    fmt="-o",
)
plt.xscale("log")
# plt.yscale('log')
plt.xlabel("True noise level sigma")
plt.ylabel("Absolute estimation error")
plt.title("Noise level estimation error vs true noise level")
plt.legend()
plt.show()


# %%
# Blind denoising models
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# Finally, we can also use blind denoising models that are trained to denoise images without knowing the noise level.
# For instance, we can use the SCUNet model from :footcite:t:`zhang2019scunet`.

blind_denoiser = dinv.models.SCUNet()

with torch.no_grad():
    denoised_blind = blind_denoiser(y)

psnr_blind = metric(denoised_blind, x).item()
dinv.utils.plot(
    {
        "Noisy": y,
        f"Denoised (blind SCUNet)\n PSNR: {psnr_blind:.2f} dB": denoised_blind,
    },
    fontsize=9,
)

# We note that this model provides less good results than the non-blind denoiser with estimated noise level.
# However, this model can also tackle more complex noise models than simple Gaussian noise with unknown standard deviation.
