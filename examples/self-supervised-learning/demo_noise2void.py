r"""
Self-supervised denoising with Noise2Void across imaging modalities
===================================================================

This example shows how to denoise a **single** noisy image without any ground truth,
using the Noise2Void loss :footcite:p:`krull2019noise2void`.

Noise2Void masks a random subset of the input pixels and asks the network to predict them
from their neighbourhood only. Because the network never sees the pixel it has to predict,
it cannot learn the identity, and the best it can do is to predict the (noise-free) signal,
provided the noise is pixel-wise independent. Notably, the loss makes no assumption on the
*distribution* of the noise, only on its independence, so the very same recipe works for
Gaussian, Poisson-Gaussian, log-Poisson or Rician noise.

We illustrate this by fitting one network per image on four modalities
(natural photograph, two-photon microscopy, CT and magnitude MRI), each corrupted with
the noise model that matches its acquisition, and compare against a simple Gaussian smoother.
"""

import deepinv as dinv
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

device = dinv.utils.get_device()
torch.manual_seed(0)

# %%
# Load the images
# ---------------
# We use four grayscale 256x256 crops, one per modality.

x_natural = dinv.utils.load_example("div2k_valid_hr_0877.png")
x_natural = x_natural.mean(1, keepdim=True)[..., 400:656, 700:956].to(device)

x_twophoton = dinv.utils.load_example("FMD_TwoPhoton_MICE_R_gt_12_avg50.png")
x_twophoton = x_twophoton[:, 1:2, 0:256, 128 : 128 + 256].to(device)

x_ct = dinv.utils.load_example("CT100_256x256_0.pt").to(device)

x_mri = dinv.utils.load_example("demo_mini_subset_fastmri_brain_0.pt")
x_mri = x_mri[:, :1, 160 - 128 : 160 + 128, 160 - 128 : 160 + 128].to(device)

# %%
# Define the physics
# ------------------
# Each image gets its own :class:`deepinv.physics.Denoising` operator, whose noise model
# reflects how the data is actually acquired:
#
# - natural photograph: sensor read noise dominates, i.e. :class:`Gaussian <deepinv.physics.GaussianNoise>`;
# - two-photon microscopy: photon shot noise plus read noise, i.e. :class:`Poisson-Gaussian <deepinv.physics.PoissonGaussianNoise>`;
# - CT: photon counting seen through the Beer-Lambert log, i.e. :class:`log-Poisson <deepinv.physics.LogPoissonNoise>`;
# - magnitude MRI: magnitude of complex Gaussian measurements, i.e. :class:`Rician <deepinv.physics.RicianNoise>`.
#
# The noise levels are chosen so that all four measurements sit in a comparable ~20-23 dB regime.

datasets = {
    "natural (DIV2K)": (
        x_natural,
        dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma=0.08), device=device),
    ),
    "two-photon (FMD)": (
        x_twophoton,
        dinv.physics.Denoising(
            dinv.physics.PoissonGaussianNoise(gain=0.075, sigma=0.02), device=device
        ),
    ),
    "CT": (
        x_ct,
        dinv.physics.Denoising(
            dinv.physics.LogPoissonNoise(N0=256.0, mu=1.0), device=device
        ),
    ),
    "MRI (magnitude)": (
        x_mri,
        dinv.physics.Denoising(dinv.physics.RicianNoise(sigma=0.075), device=device),
    ),
}

# Simulate the measurements once, so that every method below sees the same y.
measurements = {name: physics(x) for name, (x, physics) in datasets.items()}

psnr = dinv.metric.PSNR()
for name, (x, _) in datasets.items():
    print(f"{name:18s} y psnr = {psnr(measurements[name], x).item():.2f} dB")

# %%
# Model and loss
# --------------
# :class:`deepinv.loss.Noise2Void` wraps the network with ``adapt_model``, which takes care of
# masking the input pixels and of exposing the mask back to the loss. Since Noise2Void learns
# from the noisy image itself, there is nothing to pretrain: we simply fit a small U-Net from
# scratch on each image.

ITERS = 10000 if str(device) != "cpu" else 100


def train_noise2void(y, physics, iters=ITERS, lr=1e-4):
    """Fit a fresh Noise2Void network on a single noisy image."""
    model = dinv.models.UNet(
        batch_norm=False, scales=3, channels_per_scale=[16, 32, 64], device=device
    )
    loss = dinv.loss.Noise2Void()
    model = loss.adapt_model(model)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    model.train()
    for _ in range(iters):
        opt.zero_grad()
        x_net = model(y, physics, update_parameters=True)
        l = loss(x_net, y, physics, model)
        l.backward()
        opt.step()
        losses.append(l.item())

    model.eval()
    with torch.no_grad():
        x_hat = model(y, physics)
    return x_hat, losses


# %%
# Training
# --------
# Each image is treated independently: one freshly initialised network per measurement.

results = {}
for name, (x, physics) in datasets.items():
    torch.manual_seed(0)
    y = measurements[name]
    x_hat, losses = train_noise2void(y, physics)
    results[name] = {"x": x, "y": y, "x_hat": x_hat, "losses": losses}
    print(
        f"{name:18s} final loss = {losses[-1]:.3e} | n2v psnr = {psnr(x_hat, x).item():.2f} dB"
    )

# %%
# Baseline
# --------
# As a classical reference point we also denoise with a plain Gaussian smoother,
# using the same setting for every image.

for name, r in results.items():
    r["x_filt"] = TF.gaussian_blur(r["y"], kernel_size=5, sigma=1.0)
    print(
        f"{name:18s} gaussian filter psnr = {psnr(r['x_filt'], r['x']).item():.2f} dB"
    )


# %%
# Results
# -------
# Finally we compare, for each modality, the ground truth, the measurement, the Noise2Void
# reconstruction and the Gaussian smoother, with the PSNR reported in the titles.

cols = ["x", "y", "x_hat", "x_filt"]
labels = ["clean", "measurement", "noise2void", "gaussian filter"]

fig, axs = plt.subplots(
    len(results), 4, figsize=(12, 3.2 * len(results)), squeeze=False
)
for row, (name, r) in zip(axs, results.items(), strict=False):
    for ax, key, label in zip(row, cols, labels, strict=False):
        im = r[key][0, 0].detach().cpu().numpy()
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        title = (
            label if key == "x" else f"{label}\n{psnr(r[key], r['x']).item():.2f} dB"
        )
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    row[0].text(
        -0.08,
        0.5,
        name,
        transform=row[0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=11,
    )
fig.tight_layout()
plt.show()

# %%
# Loss curves
# -----------
# Note that the Noise2Void loss is a *noisy* target loss: it plateaus at roughly the noise
# variance rather than at zero, so its absolute value is not comparable across modalities.

fig, axs = plt.subplots(1, len(results), figsize=(4 * len(results), 3), squeeze=False)
for ax, (name, r) in zip(axs[0], results.items(), strict=False):
    ax.plot(r["losses"], lw=0.7)
    ax.set_yscale("log")
    ax.set_title(name)
    ax.set_xlabel("iteration")
    ax.set_ylabel("Noise2Void loss")
    ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
# Add some commented out part about training on full dataset + how to train n2v with trainer

# %%
# :References:
#
# .. footbibliography::
