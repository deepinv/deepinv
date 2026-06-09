r"""
Distributional consistency loss for Gaussian denoising
======================================================

This example compares two image-space iterative reconstructions on a simple
denoising problem with known Gaussian noise:

- pointwise measurement MSE, which drives the re-measured reconstruction toward
  the noisy measurements,
- :class:`deepinv.loss.DCLoss`, which instead matches the residual distribution
  implied by the Gaussian noise model.

Both objectives optimize the reconstructed image directly.
"""

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

import deepinv as dinv


def make_phantom(size, device):
    coords = torch.linspace(-1.0, 1.0, size, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    real = ((xx.square() + yy.square()) < 0.45).float()
    real += 0.25 * (((xx + 0.25).square() + (yy - 0.2).square()) < 0.06).float()
    return real.clamp(0.0, 1.0).unsqueeze(0).unsqueeze(0)


def magnitude(x):
    return torch.linalg.vector_norm(x, dim=1, keepdim=True)


def optimize_image(
    y,
    x_gt,
    physics,
    objective,
    *,
    sigma,
    n_points,
    steps,
    lr,
):
    x_est = torch.nn.Parameter(torch.zeros_like(x_gt))
    optimizer = torch.optim.Adam([x_est], lr=lr)
    dc_loss = dinv.loss.DCLoss(distribution="gaussian", sigma=sigma, n_points=n_points)
    image_mse = dinv.metric.MSE()

    history = {"image_mse": [], "measurement_mse": [], "dc_loss": []}

    for _ in range(steps):
        optimizer.zero_grad()
        q_est = physics.A(x_est)

        if objective == "mse":
            loss = F.mse_loss(q_est, y)
        elif objective == "dc":
            loss = dc_loss(y=y, x_net=x_est, physics=physics).mean()
        else:
            raise ValueError("objective must be 'mse' or 'dc'.")

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            q_est = physics.A(x_est)
            history["image_mse"].append(image_mse(x_gt, x_est).item())
            history["measurement_mse"].append(F.mse_loss(q_est, y).item())
            history["dc_loss"].append(dc_loss(y=y, x_net=x_est, physics=physics).item())

    x_est = x_est.detach()
    return physics.A(x_est), x_est, history


torch.manual_seed(0)
device = dinv.utils.get_device()

# The observations are Gaussian-noisy image intensities.
size = 48
sigma = 0.25
steps = 500
lr = 0.15
n_points = None

x = make_phantom(size=size, device=device)
physics = dinv.physics.Denoising(
    noise_model=dinv.physics.GaussianNoise(sigma=sigma),
    device=device,
)
y = physics(x)

q_mse, x_mse, history_mse = optimize_image(
    y,
    x,
    physics,
    "mse",
    sigma=sigma,
    n_points=n_points,
    steps=steps,
    lr=lr,
)
q_dc, x_dc, history_dc = optimize_image(
    y,
    x,
    physics,
    "dc",
    sigma=sigma,
    n_points=n_points,
    steps=steps,
    lr=lr,
)

image_mse = dinv.metric.MSE()
print(f"Final image MSE (measurement MSE): {image_mse(x, x_mse).item():.4f}")
print(f"Final image MSE (DC loss):         {image_mse(x, x_dc).item():.4f}")
print(f"Final measurement MSE (measurement MSE): " f"{F.mse_loss(q_mse, y).item():.4f}")
print(f"Final measurement MSE (DC loss):         {F.mse_loss(q_dc, y).item():.4f}")

fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)

display_images = [magnitude(x).cpu(), magnitude(x_mse).cpu(), magnitude(x_dc).cpu()]
image_handle = None
vmax = max(image.max() for image in display_images)
for axis, image, title in zip(
    axes[0],
    display_images,
    ["Ground truth", "MSE reconstruction", "DC reconstruction"],
):
    image_handle = axis.imshow(image.squeeze(), cmap="rainbow_r", vmin=0.0, vmax=vmax)
    axis.set_title(title)
    axis.axis("off")

fig.colorbar(image_handle, ax=axes[0, :], label="Intensity")

axes[1, 0].plot(history_mse["image_mse"], label="Measurement MSE")
axes[1, 0].plot(history_dc["image_mse"], label="DC loss")
axes[1, 0].set_title("Image-domain MSE")
axes[1, 0].set_xlabel("Iteration")
axes[1, 0].legend()

axes[1, 1].plot(history_mse["measurement_mse"], label="Measurement MSE")
axes[1, 1].plot(history_dc["measurement_mse"], label="DC loss")
axes[1, 1].set_title("Measurement MSE to noisy data")
axes[1, 1].set_xlabel("Iteration")
axes[1, 1].legend()

axes[1, 2].plot(history_mse["dc_loss"], label="Measurement MSE")
axes[1, 2].plot(history_dc["dc_loss"], label="DC loss")
axes[1, 2].set_title("DC objective value")
axes[1, 2].set_xlabel("Iteration")
axes[1, 2].legend()

plt.show()
