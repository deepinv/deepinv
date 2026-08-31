"""Minimal blind Poisson--Gaussian denoising on MNIST."""

import argparse

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

import deepinv as dinv
from deepinv.models.anscombe import generalized_anscombe_transform, inverse_generalized_anscombe_transform


anscombe         = lambda y, sigma, gain: generalized_anscombe_transform(y, gain=gain, sigma=sigma) / gain
inverse_anscombe = lambda z, sigma, gain: inverse_generalized_anscombe_transform(z * gain, gain=gain, sigma=sigma)


SIGMA = GAIN = 0.1


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--test-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-root")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def loaders(args, device):
    root = args.data_root or str(dinv.utils.get_cache_home() / "datasets" / "MNIST")

    def make(train, size):
        data = datasets.MNIST(root, train=train, transform=transforms.ToTensor(), download=True)
        data = Subset(data, range(min(size, len(data))))
        return DataLoader(
            data,
            batch_size=max(args.batch_size, 5),
            shuffle=train,
            pin_memory=device.type == "cuda",
        )

    return make(True, args.train_samples), make(False, args.test_samples)


def update(loss, model, optimizer, clip=None):
    """Take one finite-gradient step; otherwise discard the batch."""
    optimizer.zero_grad(set_to_none=True)
    if not torch.isfinite(loss).item():
        return False
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip or float("inf"))
    if not torch.isfinite(norm).item():
        optimizer.zero_grad(set_to_none=True)
        return False
    optimizer.step()
    return True


def train_estimator(model, physics, train, test, epochs, device):
    loss_fn = dinv.loss.CramerGaussianLoss(patch_size=4, stride=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)
    validation = [physics(x.to(device)) for x, _ in test]  # fixed noise
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses, skipped = [], 0
        for x, _ in train:
            y = physics(x.to(device))
            loss = loss_fn(model(y), y)
            if update(loss, model, optimizer, clip=0.1):
                losses.append(loss.item())
            else:
                skipped += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            estimates = [model(y) for y in validation]
            sigma = torch.cat([p["sigma"].flatten() for p in estimates]).mean().item()
            gain = torch.cat([p["gain"].flatten() for p in estimates]).mean().item()
        history.append((sigma, gain))
        print(
            f"Estimator {epoch:02d}/{epochs} | loss {sum(losses)/max(len(losses), 1):.4f} "
            f"| sigma {sigma:.4f} | gain {gain:.4f} | skipped {skipped}"
        )
    return history


def train_denoiser(estimator, physics, train, epochs, device):
    gaussian = dinv.physics.Denoising(dinv.physics.GaussianNoise(1.0), device=device)
    denoiser = dinv.models.ArtifactRemoval(
        dinv.models.DnCNN(1, 1, depth=5, nf=32, pretrained=None, device=device),
        mode="direct",
        device=device,
    )
    r2r = dinv.loss.R2RLoss(
        noise_model=gaussian.noise_model, alpha=0.2, eval_n_samples=10
    )
    model = r2r.adapt_model(denoiser)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    estimator.eval()

    for epoch in range(1, 2 * epochs + 1):
        model.train()
        losses, skipped = [], 0
        for x, _ in train:
            with torch.no_grad():
                y = physics(x.to(device))
                p = estimator(y)
                z = anscombe(y, p["sigma"], p["gain"])
            z_hat = model(z, gaussian, update_parameters=True)
            loss = r2r(x_net=z_hat, y=z, physics=gaussian, model=model)
            if update(loss, model, optimizer):
                losses.append(loss.item())
            else:
                skipped += 1
        print(
            f"R2R {epoch:02d}/{2*epochs} | loss {sum(losses)/max(len(losses), 1):.4f} "
            f"| skipped {skipped}"
        )
    model.eval()
    return model, gaussian


@torch.no_grad()
def save_examples(estimator, denoiser, physics, gaussian, test, device):
    x = next(iter(test))[0][:5].to(device)
    y = physics(x)
    p = estimator(y)
    z = anscombe(y, p["sigma"], p["gain"])
    z_hat = denoiser(z, gaussian)
    x_hat = inverse_anscombe(z_hat, p["sigma"], p["gain"]).clamp(0, 1)
    unit = lambda image: image / image.amax().clamp_min(1e-6)
    save_image(torch.cat((y.clamp(0, 1), unit(z), unit(z_hat), x_hat)), "out.png", nrow=5)

    psnr = dinv.metric.PSNR(reduction="mean")
    print(f"PSNR noisy/denoised: {psnr(y.clamp(0, 1), x):.2f}/{psnr(x_hat, x):.2f} dB")
    for i, (sigma, gain) in enumerate(zip(p["sigma"].flatten(), p["gain"].flatten()), 1):
        print(f"{i}: sigma {SIGMA:.1f}/{sigma:.4f}, gain {GAIN:.1f}/{gain:.4f} (true/estimated)")


def main(args=None):
    args = arguments() if args is None else args
    torch.manual_seed(args.seed)
    device = dinv.utils.get_device()
    train, test = loaders(args, device)
    physics = dinv.physics.Denoising(
        dinv.physics.PoissonGaussianNoise(SIGMA, GAIN, clip_positive=True), device=device
    )
    estimator = dinv.models.PoissonGaussianEstimator(
        dinv.models.DnCNN(1, 2, depth=3, nf=18, pretrained=None, device=device),
        noise_map=False,
    ).to(device)

    history = train_estimator(estimator, physics, train, test, args.epochs, device)
    denoiser, gaussian = train_denoiser(estimator, physics, train, args.epochs, device)
    torch.save(estimator.state_dict(), "noise_estimator.pth")
    torch.save(denoiser.model.state_dict(), "gaussian_denoiser_r2r.pth")
    torch.save(torch.tensor(history), "training_trajectory.pt")
    save_examples(estimator, denoiser, physics, gaussian, test, device)
    return estimator, denoiser


if __name__ == "__main__":
    main()
