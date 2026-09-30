r"""
Self-supervised denoising with Noise2Void across imaging modalities
===================================================================

This example shows how to denoise a **single** noisy image without any ground truth,
using the Noise2Void loss :footcite:p:`krull2019noise2void`.

Noise2Void masks a random subset of the input pixels and asks the network to predict them
from their neighbourhood only. Because the network never sees the pixel it has to predict,
it cannot learn the identity. If the noise is assumed pixel-wise independent, the model cannot estimate the noise, 
and only provides an estimate of the signal.

The loss makes no assumption on the *distribution* of the noise, only on its independence, so the very same recipe works for
Gaussian, Poisson-Gaussian, log-Poisson or Rician noise.

We demonstrate Noise2Void on natural images, two-photon microscopy, CT and magnitude MRI.
"""

import deepinv as dinv
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = dinv.utils.get_device()
torch.manual_seed(0)

# %%
# Load the images
# ---------------
# We use four grayscale 256x256 crops, one per modality.

x_natural = dinv.utils.load_example("div2k_valid_hr_0877.png", grayscale=True)
x_natural = x_natural[..., 400:656, 700:956].to(device)

x_twophoton = dinv.utils.load_example("FMD_TwoPhoton_MICE_R_gt_12_avg50.png")
x_twophoton = x_twophoton[:, 1:2, 0:256, 128 : 128 + 256].to(device)

x_ct = dinv.utils.load_example("CT100_256x256_0.pt").to(device)

x_mri = dinv.utils.load_example("demo_mini_subset_fastmri_brain_0.pt")
x_mri = x_mri[:, :1, 160 - 128 : 160 + 128, 160 - 128 : 160 + 128].to(device)

# %%
# Define the physics
# ------------------
# We simulate noisy acquisition with various noise models:
# - natural photograph: sensor read noise dominates, i.e. :class:`Gaussian <deepinv.physics.GaussianNoise>`;
# - two-photon microscopy: photon shot noise plus read noise, i.e. :class:`Poisson-Gaussian <deepinv.physics.PoissonGaussianNoise>`;
# - CT: photon counting seen through the Beer-Lambert log, i.e. :class:`log-Poisson <deepinv.physics.LogPoissonNoise>`;
# - magnitude MRI: magnitude of complex Gaussian measurements, i.e. :class:`Rician <deepinv.physics.RicianNoise>`.


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

measurements = {name: physics(x) for name, (x, physics) in datasets.items()}

psnr = dinv.metric.PSNR()
for name, (x, _) in datasets.items():
    print(f"{name:18s} y psnr = {psnr(measurements[name], x).item():.2f} dB")


# %%
# Training
# --------
# :class:`deepinv.loss.Noise2Void` wraps the network with ``adapt_model``, which takes care of
# masking the input pixels and of exposing the mask back to the loss. We fit a small U-Net from
# scratch on each image.

ITERS = 10000 if str(device) != "cpu" else 100
results = {}
for name, (x, physics) in datasets.items():
    torch.manual_seed(0)
    y = measurements[name]

    model = dinv.models.UNet(
        batch_norm=False, scales=3, channels_per_scale=[16, 32, 64], device=device
    )
    loss = dinv.loss.Noise2Void()

    trainer = dinv.Trainer(
        model=loss.adapt_model(model),
        physics=physics,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
        train_dataloader=DataLoader(dinv.datasets.TensorDataset(y=y)),
        losses=loss,
        epochs=ITERS,
        metrics=None,
        device=device,
        save_path=None,
        verbose=False,
        show_progress_bar=False,
    )
    model = trainer.train()

    model.eval()
    with torch.no_grad():
        x_hat = model(y, physics)
    losses = trainer.loss_history[loss.__class__.__name__]

    results[name] = {"x": x, "y": y, "x_hat": x_hat, "losses": losses}
    print(
        f"{name:18s} final loss = {losses[-1]:.3e} | n2v psnr = {psnr(x_hat, x).item():.2f} dB"
    )

# %%
# Baseline
# --------
# As a classical reference point we also denoise with a :class:`median filter <deepinv.models.MedianFilter>`.

median = dinv.models.MedianFilter(kernel_size=3)
for name, r in results.items():
    r["x_filt"] = median(r["y"])


# %%
# Results
# -------
# Finally we compare, for each modality, the ground truth, the measurement, the Noise2Void
# reconstruction and the median filter.

cols = ["x", "y", "x_hat", "x_filt"]
labels = ["clean", "measurement", "noise2void", "median filter"]

dinv.utils.plot(
    [torch.cat([r[key] for r in results.values()]) for key in cols],
    titles=labels,
    subtitles=[
        [name] + [f"{psnr(r[key], r['x']).item():.2f} dB" for key in cols[1:]]
        for name, r in results.items()
    ],
    max_imgs=len(results),
    rescale_mode="clip",
    figsize=(12, 3.2 * len(results)),
)

# %%
# # Why the MRI image is denoised less well than others
# ----------------------
# Noise2Void learns the expected value of the noisy measurement. For Gaussian and
# Poisson-Gaussian noise this is the clean image (and nearly so for log-Poisson here), but
# Rician noise is biased: a zero-valued pixel has expected value
# :math:`\sigma\sqrt{\pi/2} \approx 0.094`. Noise2Void therefore reproduces this offset
# instead of removing it, as the mean of the background (the top three rows) shows.

for name in ["CT", "MRI (magnitude)"]:
    means = [f"{results[name][k][..., :3, :].mean().item():.3f}" for k in cols]
    print(f"{name:18s} background mean ({', '.join(labels)}) = {', '.join(means)}")

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

# %%
# :References:
#
# .. footbibliography::
