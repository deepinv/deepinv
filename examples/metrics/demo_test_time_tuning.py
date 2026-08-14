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

We also report the oracle PSNR against the ground truth for reference.
"""

# %%
# Setup
# -----
# We use a single high-resolution natural image. All five no-reference metrics were designed for
# natural photographs, so this is the regime where they are best behaved.
#
# Every metric is also constructed with ``center_crop=-16``, which discards a 16 pixel band around
# the image before scoring it. Reconstructions are systematically worse near the borders, where the
# model has no neighbouring pixels to rely on.
#
# .. note::
#      ``denominator=1/255`` passed to :class:`deepinv.loss.metric.NIQE`: the published NIQE
#      weights were fitted on images in the :math:`[0, 255]` range, and this rescales our :math:`[0, 1]`
#      images accordingly. BRISQUE and NIMA handle this internally through their ``max_pixel`` argument.
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

# discard a 16 pixel border, so that boundary artifacts do not drive the metrics
crop = -16

metrics = {
    "BRISQUE": dinv.loss.metric.BRISQUE(device=device, center_crop=crop),
    "NIMA": dinv.loss.metric.NIMA(variant="technical", device=device, center_crop=crop),
    "NIQE": dinv.loss.metric.NIQE(denominator=1 / 255, device=device, center_crop=crop),
    "SharpnessIndex": dinv.loss.metric.SharpnessIndex(center_crop=crop),
    "BlurStrength": dinv.loss.metric.BlurStrength(center_crop=crop),
}
psnr = dinv.loss.metric.PSNR(center_crop=crop)

# %%
# A helper to sweep and plot
# --------------------------
# ``evaluate`` scores a set of candidate reconstructions with every metric, and ``plot_sweep``
# draws one panel per metric. In each panel the dashed vertical line marks the true parameter
# (which the model is not told), and the highlighted point marks the parameter that metric would
# select. A metric is useful for test-time tuning exactly when the two coincide.


def evaluate(reconstructions: dict) -> dict:
    r"""Score each reconstruction with PSNR and every no-reference metric."""
    scores = {"PSNR (oracle)": [float(psnr(xh, x)) for xh in reconstructions.values()]}
    for name, m in metrics.items():
        scores[name] = [float(m(xh)) for xh in reconstructions.values()]
    return scores


def plot_sweep(params: list, scores: dict, true_param: float, xlabel: str, title: str):
    r"""Plot each metric against the swept parameter, marking its selected value."""
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))

    for ax, (name, values) in zip(axs.ravel(), scores.items(), strict=True):
        # PSNR and NIMA are higher-is-better, the other metrics are lower-is-better
        lower_better = metrics[name].lower_better if name in metrics else False
        best = min(
            range(len(values)), key=lambda i: values[i] if lower_better else -values[i]
        )

        ax.plot(params, values, "o-", color="tab:blue")
        ax.axvline(
            true_param, color="k", linestyle="--", linewidth=1, label="true value"
        )
        ax.plot(
            params[best],
            values[best],
            "*",
            color="tab:red",
            markersize=18,
            label="selected",
        )

        direction = "lower is better" if lower_better else "higher is better"
        outcome = "matches truth" if params[best] == true_param else "misses truth"
        ax.set_title(f"{name} ({direction})\n{outcome}", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_xticks(params)
        ax.legend(fontsize=8)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


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

with torch.no_grad():
    denoised = {s: denoiser(y, s) for s in sigmas}

plot(
    [x, y] + list(denoised.values()),
    titles=["ground truth", rf"noisy ($\sigma={sigma_true}$)"]
    + [rf"DRUNet $\sigma={s}$" for s in sigmas],
    rescale_mode="clip",
)

# %%
# Sweeping the noise level
# ~~~~~~~~~~~~~~~~~~~~~~~~
# The oracle PSNR peaks at the true :math:`\sigma = 0.1`, as expected. BRISQUE, NIMA and the
# sharpness index agree with it. NIQE and BlurStrength instead both prefer :math:`\sigma = 0.05`,
# the under-smoothed reconstruction.
#

scores_denoising = evaluate(denoised)
plot_sweep(
    sigmas,
    scores_denoising,
    sigma_true,
    xlabel=r"assumed noise level $\sigma$",
    title=rf"Denoising: selecting the noise level without ground truth (true $\sigma={sigma_true}$)",
)

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
    deblurred = {}
    for s in blur_sigmas:
        # update physics to use the candidate kernel width, and run RAM
        blur_physics.update(filter=dinv.physics.functional.gaussian_blur(sigma=(s, s)))
        deblurred[s] = model(y, blur_physics)

plot(
    [x, y] + list(deblurred.values()),
    titles=["ground truth", rf"blurred ($\sigma={blur_true}$)"]
    + [rf"RAM $\sigma={s}$" for s in blur_sigmas],
    rescale_mode="clip",
)

# %%
# Sweeping the kernel width
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# This problem is much harder. The oracle PSNR
# peaks sharply at the true :math:`\sigma = 2` and collapses on either side, losing close to 20 dB
# by :math:`\sigma = 4`. NIMA is the only metric that recovers it.
#
# The other four all prefer an over-estimated kernel, and they fail in the same direction: NIQE and
# the sharpness index pick :math:`\sigma = 3`, BRISQUE and the blur strength pick :math:`\sigma = 4`.

scores_deblurring = evaluate(deblurred)
plot_sweep(
    blur_sigmas,
    scores_deblurring,
    blur_true,
    xlabel=r"assumed blur width $\sigma$",
    title=rf"Deblurring: selecting the kernel width without ground truth (true $\sigma={blur_true}$)",
)

# %%
# :References:
#
# .. footbibliography::
