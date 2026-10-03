r"""
Blind Poisson--Gaussian denoising
====================================================================================================

This example presents two simple blind Poisson--Gaussian denoising workflows.
First, we train a noise estimator and a self-supervised denoiser on fixed noisy
MNIST images. Then, we evaluate pretrained PGE-Net and FBI-Net models on a color
image.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

import deepinv as dinv


# %%
# Setup
# -----
#
# Fix the random seed, select the device, and define two small helpers for the
# fixed-measurement training loops.
#

dinv.utils.disable_tex()
torch.manual_seed(0)
device = dinv.utils.get_device()

batch_size = 64
pin = device.type == "cuda"


@torch.no_grad()
def fixed_loader(x, transform, shuffle, source=None):
    """Evaluate a transform once, then store fixed ``(x, y)`` pairs."""
    source = x if source is None else source
    y = torch.cat([transform(b.to(device)).cpu() for b in source.split(batch_size)])
    return DataLoader(
        TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle, pin_memory=pin
    )


def fit(model, physics, loss, train, test, epochs, lr, clip=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = dinv.Trainer(
        model=model,
        physics=physics,
        optimizer=optimizer,
        losses=loss,
        epochs=epochs,
        train_dataloader=train,
        eval_dataloader=test,
        online_measurements=False,
        metrics=None,
        compute_train_metrics=False,
        compute_eval_losses=True,
        grad_clip=clip,
        check_grad=clip is not None,
        device=device,
        save_path=None,
        show_progress_bar=False,
        non_blocking_transfers=pin,
    )
    return trainer.train(), trainer


# %%
# Generate fixed noisy MNIST images
# ---------------------------------
#
# We corrupt a small MNIST subset once using fixed Poisson--Gaussian parameters.
# The same noisy images are reused at every epoch.
#

sigma = gain = 0.1
epochs, n_train, n_test = 20, 2048, 256
root = dinv.utils.get_cache_home() / "datasets" / "MNIST"
physics = dinv.physics.Denoising(
    dinv.physics.PoissonGaussianNoise(sigma, gain, clip_positive=True),
    device=device,
)
gat = lambda y, p: dinv.models.generalized_anscombe_transform(
    y, p["gain"], p["sigma"], normalize=True
)
igat = lambda z, p: dinv.models.inverse_generalized_anscombe_transform(
    z, p["gain"], p["sigma"], normalize=True
)


def mnist(train, n):
    data = datasets.MNIST(root, train=train, download=True)
    return data.data[:n].float().unsqueeze(1) / 255


x_train, x_test = mnist(True, n_train), mnist(False, n_test)
train = fixed_loader(x_train, physics, True)
test = fixed_loader(x_test, physics, False)


# %%
# Estimate the noise parameters
# -----------------------------
#
# A small DnCNN predicts the Gaussian standard deviation and Poisson gain. We
# train it from noisy images using :class:`deepinv.loss.CramerGaussianLoss`.
#

estimator = dinv.models.PoissonGaussianEstimator(
    dinv.models.DnCNN(1, 2, depth=3, nf=8, pretrained=None, device=device),
    noise_map=False,
    eps=1e-5,
).to(device)
estimator, pge_trainer = fit(
    estimator,
    physics,
    dinv.loss.CramerGaussianLoss(
        gaussian_estimator=dinv.models.WaveletNoiseEstimator()
    ),
    train,
    test,
    epochs,
    1e-4,
    clip=0.1,
)
estimator.requires_grad_(False).eval()


# %%
# Train the denoiser in the Anscombe domain
# ------------------------------------------
#
# We transform the fixed measurements with the estimated parameters and train a
# Gaussian denoiser using :class:`deepinv.loss.R2RLoss`.
#

to_gat = lambda y: gat(y, estimator(y))
gat_train = fixed_loader(x_train, to_gat, True, train.dataset.tensors[1])
gat_test = fixed_loader(x_test, to_gat, False, test.dataset.tensors[1])
gaussian = dinv.physics.Denoising(dinv.physics.GaussianNoise(1.0), device=device)
r2r_loss = dinv.loss.R2RLoss(
    noise_model=gaussian.noise_model, alpha=0.2, eval_n_samples=10
)
denoiser = r2r_loss.adapt_model(
    dinv.models.ArtifactRemoval(
        dinv.models.DnCNN(1, 1, depth=5, nf=32, pretrained=None, device=device),
        mode="direct",
        device=device,
    )
)
denoiser, r2r_trainer = fit(
    denoiser, gaussian, r2r_loss, gat_train, gat_test, epochs, 5e-4
)


# %%
# Evaluate the MNIST models
# -------------------------
#
# Save the trained weights and plot five examples. The four image columns contain the
# noisy data, Anscombe data, denoised Anscombe data, and inverse-transform result.
#

torch.save(estimator.state_dict(), "noise_estimator.pth")
torch.save(denoiser.model.state_dict(), "gaussian_denoiser_r2r.pth")
torch.save(
    {"PGE": pge_trainer.loss_history, "R2R": r2r_trainer.loss_history},
    "training_trajectory.pt",
)

with torch.no_grad():
    x, y = x_test[:5].to(device), test.dataset.tensors[1][:5].to(device)
    p = estimator(y)
    z = gat(y, p)
    z_hat = denoiser.eval()(z, gaussian)
    x_hat = igat(z_hat, p).clamp(0, 1)
    unit = lambda image: image / image.amax().clamp_min(1e-6)
    dinv.utils.plot(
        [y.clamp(0, 1), unit(z), unit(z_hat), x_hat],
        ["Noisy", "Anscombe", "Denoised Anscombe", "Denoised"],
        max_imgs=5,
        rescale_mode="clip",
    )

psnr = dinv.metric.PSNR(reduction="mean")
print("\nMNIST")
print(f"PSNR noisy/denoised: {psnr(y, x):.2f}/{psnr(x_hat, x):.2f} dB")
print(f"sigma true/estimated: {sigma:.3f}/{p['sigma'].mean():.3f}")
print(f"gain  true/estimated: {gain:.3f}/{p['gain'].mean():.3f}")


# %%
# Evaluate pretrained models on a color image
# --------------------------------------------
#
# We now load pretrained PGE-Net and FBI-Net weights. As in the original model,
# each RGB channel is processed independently in the Anscombe domain.
#

torch.manual_seed(0)
x = dinv.utils.load_example("butterfly.png", device=device)
physics = dinv.physics.Denoising(
    dinv.physics.PoissonGaussianNoise(sigma=0.02, gain=0.05, clip_positive=True),
    device=device,
)
y = physics(x)

weights_dir = Path(__file__).resolve().parents[2]
pge_weights = weights_dir / "PGENet_color.pth"
fbi_weights = weights_dir / "FBINet_color.pth"
pge = dinv.models.PoissonGaussianEstimator(
    dinv.models.PGENet(square_output=True), eps=0.0, noise_map=True
).to(device)
fbi = dinv.models.FBINet().to(device)
pge.backbone_net.load_state_dict(
    torch.load(pge_weights, map_location=device, weights_only=True)
)
fbi.load_state_dict(torch.load(fbi_weights, map_location=device, weights_only=True))
pge.eval()
fbi.eval()
denoiser = dinv.models.AnscombeDenoiser(fbi)

with torch.no_grad():
    channels = y.squeeze(0).unsqueeze(1)  # Estimate each RGB channel independently.
    params = pge(channels)
    sigma_maps, gain_maps = params["sigma"], params["gain"]
    sigma_rgb = sigma_maps.mean(dim=(1, 2, 3))
    gain_rgb = gain_maps.mean(dim=(1, 2, 3))
    estimate = denoiser(channels, sigma_rgb, gain_rgb).clamp(0, 1)
    estimate = estimate.squeeze(1).unsqueeze(0)


# %%
# Plot the pretrained result
# --------------------------
#
# Display the clean, noisy, and denoised images together with PSNR and SSIM.
#

score = lambda image: (
    dinv.metric.PSNR()(image, x).mean().item(),
    dinv.metric.SSIM()(image, x).mean().item(),
)
noisy_score, estimate_score = map(score, (y, estimate))
dinv.utils.plot(
    [x, y, estimate],
    [
        "Clean",
        f"Noisy\nPSNR {noisy_score[0]:.2f} dB\nSSIM {noisy_score[1]:.4f}",
        f"Native estimate\nPSNR {estimate_score[0]:.2f} dB\n"
        f"SSIM {estimate_score[1]:.4f}",
    ],
    rescale_mode="clip",
    dpi=200,
    figsize=(10, 4.5),
)

fmt = lambda values: "[" + ", ".join(f"{v:.4f}" for v in values.tolist()) + "]"
print("\nBUTTERFLY")
print(
    f"PSNR noisy/denoised: {noisy_score[0]:.2f}/{estimate_score[0]:.2f} dB\n"
    f"SSIM noisy/denoised: {noisy_score[1]:.4f}/{estimate_score[1]:.4f}\n"
    f"gain  true/estimated RGB: 0.0500/{fmt(gain_rgb)}\n"
    f"sigma true/estimated RGB: 0.0200/{fmt(sigma_rgb)}"
)
