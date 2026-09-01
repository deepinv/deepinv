"""Two-step blind Poisson--Gaussian denoising on fixed noisy MNIST images."""

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.utils import save_image

import deepinv as dinv


# Settings ----------------------------------------------------------------------
torch.manual_seed(0)
device = dinv.utils.get_device()
sigma = gain = 0.1
epochs, batch_size = 20, 64
n_train, n_test = 2048, 256
root = dinv.utils.get_cache_home() / "datasets" / "MNIST"
pin = device.type == "cuda"

physics = dinv.physics.Denoising(
    dinv.physics.PoissonGaussianNoise(sigma, gain, clip_positive=True),
    device=device,
)

gat = lambda y, p: dinv.models.generalized_anscombe_transform(
    y, p["gain"], p["sigma"], normalize=True
)

igat = lambda z, p: dinv.models.inverse_generalized_anscombe_transform(
    z , p["gain"], p["sigma"], normalize=True
)


@torch.no_grad()
def fixed_loader(x, transform, shuffle, source=None):
    """Evaluate a transform once, then store fixed ``(x, y)`` pairs."""
    source = x if source is None else source
    y = torch.cat([transform(b.to(device)).cpu() for b in source.split(batch_size)])
    return DataLoader(
        TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle, pin_memory=pin
    )


def fit(model, physics, loss, train, test, n_epochs, lr, clip=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = dinv.Trainer(
        model=model,
        physics=physics,
        optimizer=optimizer,
        losses=loss,
        epochs=n_epochs,
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


# Fixed Poisson--Gaussian dataset ------------------------------------------------
def mnist(train, n):
    data = datasets.MNIST(root, train=train, download=True)
    return data.data[:n].float().unsqueeze(1) / 255


x_train, x_test = mnist(True, n_train), mnist(False, n_test)
train = fixed_loader(x_train, physics, True)
test = fixed_loader(x_test, physics, False)


# Step 1: estimate sigma and gain ------------------------------------------------
estimator = dinv.models.PoissonGaussianEstimator(
    dinv.models.DnCNN(1, 2, depth=3, nf=8, pretrained=None, device=device),
    noise_map=False,
).to(device)
pge_loss = dinv.loss.CramerGaussianLoss(
    gaussian_estimator=dinv.models.WaveletNoiseEstimator(),
)
estimator, pge_trainer = fit(
    estimator, physics, pge_loss, train, test, epochs, 1e-4, clip=0.1
)
estimator.requires_grad_(False).eval()


# Step 2: fixed GAT dataset and Gaussian R2R denoiser ----------------------------
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


# Save and evaluate --------------------------------------------------------------
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
    save_image(
        torch.cat((y.clamp(0, 1), unit(z), unit(z_hat), x_hat)),
        "out_mnist.png",
        nrow=5,
    )

psnr = dinv.metric.PSNR(reduction="mean")
print(f"PSNR noisy/denoised: {psnr(y, x):.2f}/{psnr(x_hat, x):.2f} dB")
print(f"sigma true/estimated: {sigma:.3f}/{p['sigma'].mean():.3f}")
print(f"gain  true/estimated: {gain:.3f}/{p['gain'].mean():.3f}")
