r"""
Blind inverse problems with no reference metrics
====================================================================================================

In blind inverse problems, some parameters of the physics are unknown at test time.
Running non-blind models requires knowing these parameters, which are often hard to estimate.
For example, a denoiser needs the noise level :math:`\sigma`, a deblurring model needs the blur kernel.

*No-reference* (or blind) image quality metrics offer an approach for tuning these parameters without ground truth.
They score a single image :math:`\hat{x}` without ever seeing a reference, so they can be evaluated at test time on the
reconstruction itself. If such a metric is a good proxy for reconstruction quality, we can sweep
the unknown parameter, score each candidate reconstruction, and keep the best one, entirely
without ground truth.

We take two problems where we *do* know the right physics parameter,
hide it from the model, and check whether each no-reference metric recovers it:

1. **Denoising.** We corrupt an image with Gaussian noise of :math:`\sigma = 0.1` and reconstruct
   it with :class:`DRUNet <deepinv.models.DRUNet>` :footcite:p:`zhang2021plug`, pretending the
   noise level is unknown and sweeping :math:`\sigma \in \{0.05, 0.1, 0.15, 0.2\}`.
2. **Deblurring.** We blur an image with a Gaussian kernel of width :math:`\sigma = 2` and
   reconstruct it with the :class:`RAM <deepinv.models.RAM>` foundation model
   :footcite:p:`terris2025reconstruct`, pretending the kernel width is unknown and sweeping
   :math:`\sigma \in \{1, 2, 3, 4\}`.

We score every reconstruction with the five no-reference metrics of the library:

- :class:`deepinv.loss.metric.BRISQUE` :footcite:p:`mittal2012no`, which scores how far the image
  departs from the natural scene statistics of high quality photographs, using a regressor trained on
  human quality ratings. Lower is better.
- :class:`deepinv.loss.metric.NIMA` :footcite:p:`talebi2018nima`, which predicts the distribution
  of human opinion scores with a convolutional network. Higher is better. We use the ``technical``
  head, which is trained to rate distortion rather than aesthetic appeal.
- :class:`deepinv.loss.metric.NIQE` :footcite:p:`mittal2012making`, which measures the distance
  between the local statistics of the image and a model fitted on high quality images. Lower is better.
- :class:`deepinv.loss.metric.SharpnessIndex` :footcite:p:`blanchet2012sharpness,leclaire2015sharpness`,
  which measures sharpness through the global phase coherence of the image. Higher is better.
- :class:`deepinv.loss.metric.BlurStrength` :footcite:p:`crete2007blur`, which estimates blur from
  how much the image changes when it is deliberately blurred further. Lower is better.

"""

# %%
# Setup
# -----
# We use natural images which is what the above no-reference metrics were designed for.
#
# Every metric is also constructed with ``center_crop=-16`` to disregard edge effects.
#

import torch
import matplotlib.pyplot as plt

import deepinv as dinv
from deepinv.utils import plot

device = dinv.utils.get_device()
torch.manual_seed(0)

x = dinv.utils.load_example(
    "div2k_valid_hr_0877.png", img_size=256, resize_mode="crop"
).to(device)

crop = -16

metrics = [
    dinv.loss.metric.BRISQUE(device=device, center_crop=crop),
    dinv.loss.metric.NIMA(variant="technical", device=device, center_crop=crop),
    dinv.loss.metric.NIQE(denominator=1 / 255, device=device, center_crop=crop),
    dinv.loss.metric.SharpnessIndex(center_crop=crop),
    dinv.loss.metric.BlurStrength(center_crop=crop),
]


# %%
# Experiment 1: denoising with an unknown noise level
# ---------------------------------------------------
# We add Gaussian noise of :math:`\sigma = 0.1` and run DRUNet at four candidate noise levels.
# Under-estimating :math:`\sigma` leaves visible noise in the output, over-estimating it removes
# the noise but also the fine texture of the image.

sigma_true = 0.1
physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma=sigma_true))

torch.manual_seed(0)  # fix the noise realization so the sweep is reproducible
y = physics(x)

denoiser = dinv.models.DRUNet(pretrained="download", device=device)
sigmas = [0.05, 0.1, 0.15, 0.2]

denoised = []
for s in sigmas:
    with torch.no_grad():
        denoised.append(denoiser(y, s))

plot(
    [x, y] + denoised,
    titles=["ground truth", rf"noisy ($\sigma={sigma_true}$)"]
    + [rf"DRUNet $\sigma={s}$" for s in sigmas],
    rescale_mode="clip",
)

# %%
# Plot the no-reference metrics as a function of the assumed noise level

plt.figure(figsize=(12, 8))

with torch.no_grad():
    for i, metric in enumerate(metrics):
        scores = []
        for x_hat in denoised:
            scores.append(metric(x_hat).item())

        name = metric.__class__.__name__
        plt.subplot(2, 3, i + 1)
        plt.plot(sigmas, scores, marker="o", label=name)
        plt.title(
            f"{name} ({'lower is better' if metric.lower_better else 'higher is better'})"
        )
        plt.xlabel(rf"assumed noise $\sigma$")
        plt.axvline(x=sigma_true, color="red", linestyle="--", alpha=0.5)
        plt.ylabel(f"score")
        plt.grid()

plt.tight_layout()
plt.show()

# %%
# Experiment 2: deblurring with an unknown kernel width
# -----------------------------------------------------
# We blur the image with a Gaussian kernel of width :math:`\sigma = 2` (plus a little noise) and
# reconstruct with RAM with the 4 candidate kernels. Assuming too small a
# kernel leaves the image blurry, assuming too large a kernel makes the model over-sharpen and
# introduces ringing and other artifacts.

blur_true = 2.0

blur_physics = dinv.physics.BlurFFT(
    img_size=x.shape[1:],
    filter=dinv.physics.functional.gaussian_blur(sigma=(blur_true, blur_true)),
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=0.01),
)

# fix the noise realization so the sweep is reproducible
torch.manual_seed(0)
y = blur_physics(x)

model = dinv.models.RAM(device=device)
blur_sigmas = [1.0, 2.0, 3.0, 4.0]

with torch.no_grad():
    deblurred = []
    for s in blur_sigmas:
        # update physics to use the candidate kernel width, and run RAM
        blur_physics.update(filter=dinv.physics.functional.gaussian_blur(sigma=(s, s)))
        deblurred.append(model(y, blur_physics))

plot(
    [x, y] + deblurred,
    titles=["ground truth", rf"blurred ($\sigma={blur_true}$)"]
    + [rf"RAM $\sigma={s}$" for s in blur_sigmas],
    rescale_mode="clip",
)

# %%
# Plot the no-reference metrics as a function of the assumed blur

plt.figure(figsize=(12, 8))

with torch.no_grad():
    for i, metric in enumerate(metrics):
        scores = []
        for x_hat in denoised:
            scores.append(metric(x_hat).item())

        name = metric.__class__.__name__
        plt.subplot(2, 3, i + 1)
        plt.plot(blur_sigmas, scores, marker="o", label=name)
        plt.title(
            f"{name} ({'lower is better' if metric.lower_better else 'higher is better'})"
        )
        plt.xlabel(rf"assumed blur $\sigma$")
        plt.axvline(x=blur_true, color="red", linestyle="--", alpha=0.5)
        plt.ylabel(f"score")
        plt.grid()

plt.tight_layout()
plt.show()

# %%
# :References:
#
# .. footbibliography::
