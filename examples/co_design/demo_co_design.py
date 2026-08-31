r"""
End-to-end co-design baseline with compressed sensing and MNIST
================================================================

This example jointly trains the full compressed sensing matrix and an
:class:`deepinv.models.ArtifactRemoval` network to reconstruct MNIST images
from compressed sensing measurements.

The reconstruction objective combines a supervised mean squared error (MSE)
with a Tikhonov regularizer:

.. math::

    \mathcal{L}(\hat{x}, x) = \operatorname{MSE}(\hat{x}, x)
    + \alpha \frac{1}{2}\|\hat{x}\|_2^2.

The measurements are generated online from a trainable
:class:`deepinv.physics.CompressedSensing` operator. Both the acquisition
matrix and the reconstruction network receive gradients from the same MSE
plus Tikhonov objective.
"""

from __future__ import annotations

from pathlib import Path

import torch
import matplotlib.pyplot as plt
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

import deepinv as dinv


class ImageOnlyDataset(Dataset):
    """Expose only images from a torchvision dataset.

    DeepInverse interprets a two-element dataset sample as ``(x, y)``. MNIST
    returns ``(image, label)``, so the label must be removed when measurements
    are generated online by :class:`deepinv.Trainer`.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tensor:
        image, _ = self.dataset[index]
        return image


class TrainableCompressedSensing(dinv.physics.CompressedSensing):
    """Compressed sensing operator whose complete matrix is trainable.

    The example uses the adjoint reconstruction mode, so the adjoint is
    computed from the current matrix on every forward pass. This keeps the
    adjoint synchronized with the matrix updated by the optimizer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._A = nn.Parameter(self._A.detach().clone())
        self._buffers.pop("_A_adjoint")
        self._buffers.pop("_A_dagger")

    def A_adjoint(self, y: Tensor, **kwargs) -> Tensor:
        if self.channelwise:
            raise NotImplementedError(
                "This co-design example supports channelwise=False only."
            )

        batch_size = y.shape[0]
        channels, height, width = self.img_size
        y = y.reshape(batch_size, -1).to(dtype=self._A.dtype)
        x = torch.einsum("im,nm->in", y, self._A.conj().T)
        return x.reshape(batch_size, channels, height, width)

    def A_dagger(self, y: Tensor, **kwargs) -> Tensor:
        """Apply the pseudoinverse of the current trainable matrix."""

        if self.channelwise:
            raise NotImplementedError(
                "This co-design example supports channelwise=False only."
            )

        batch_size = y.shape[0]
        channels, height, width = self.img_size
        y = y.reshape(batch_size, -1).to(dtype=self._A.dtype)
        x = torch.einsum("im,nm->in", y, torch.linalg.pinv(self._A))
        return x.reshape(batch_size, channels, height, width)


class TikhonovLoss(dinv.loss.Loss):
    r"""Tikhonov penalty added to a supervised reconstruction loss.

    :param float weight: Weight of the Tikhonov penalty.
    """

    def __init__(self, weight: float = 1e-4):
        super().__init__()
        self.weight = weight
        self.prior = dinv.optim.Tikhonov()

    def forward(self, x_net: Tensor, **kwargs) -> Tensor:
        return self.weight * self.prior(x_net)


# %%
# Setup
# -----

torch.manual_seed(0)
device = dinv.utils.get_device()

base_dir = Path(".")
dataset_dir = dinv.utils.get_cache_home() / "datasets" / "MNIST"

image_size = 28
image_shape = (1, image_size, image_size)
num_measurements = 196  # 25% of the 784 MNIST pixels.

# Keep the CPU example short enough to run as a tutorial. Increase these
# values for a meaningful reconstruction model.
num_train = 512 if device.type == "cuda" else 128
num_test = 128 if device.type == "cuda" else 32
epochs = 5 if device.type == "cuda" else 1
batch_size = 64 if device.type == "cuda" else 16

mnist_transform = transforms.ToTensor()
mnist_train = ImageOnlyDataset(
    datasets.MNIST(
        root=dataset_dir,
        train=True,
        transform=mnist_transform,
        download=True,
    )
)
mnist_test = ImageOnlyDataset(
    datasets.MNIST(
        root=dataset_dir,
        train=False,
        transform=mnist_transform,
        download=True,
    )
)

train_dataset = Subset(mnist_train, range(num_train))
test_dataset = Subset(mnist_test, range(num_test))

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)

# %%
# Physics
# -------
#
# The full compressed sensing matrix is trainable. Measurements are generated
# online from each MNIST image using the current matrix.

physics = TrainableCompressedSensing(
    m=num_measurements,
    img_size=image_shape,
    device=device,
    rng=torch.Generator(device=device).manual_seed(0),
)

# %%
# Reconstruction model
# --------------------
#
# ArtifactRemoval first applies the adjoint of the physics and then uses a
# trainable denoising backbone to remove the resulting artifacts.

backbone = dinv.models.UNet(
    in_channels=1,
    out_channels=1,
    scales=2,
    channels_per_scale=[32, 64],
    batch_norm=False,
    device=device,
)
model = dinv.models.ArtifactRemoval(
    backbone_net=backbone,
    mode="adjoint",
    device=device,
)

# %%
# Training objective
# ------------------
#
# ``SupLoss`` provides the MSE term. ``TikhonovLoss`` uses the existing
# ``deepinv.optim.Tikhonov`` prior as an additional image regularizer.

losses = [
    dinv.loss.SupLoss(metric=dinv.metric.MSE()),
    TikhonovLoss(weight=1e-4),
]

initial_matrix = physics._A.detach().clone()

optimizer = torch.optim.Adam(
    [
        {"params": model.parameters(), "lr": 1e-3},
        {"params": [physics._A], "lr": 1e-4},
    ]
)

trainer = dinv.Trainer(
    model=model,
    physics=physics,
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    online_measurements=True,
    epochs=epochs,
    losses=losses,
    metrics=dinv.metric.PSNR(),
    device=device,
    save_path=None,
    plot_images=False,
    plot_measurements=False,
    verbose=False,
    show_progress_bar=False,
)

model = trainer.train()

# %%
# Matrix before and after training
# ---------------------------------

trained_matrix = physics._A.detach().cpu()
initial_matrix = initial_matrix.cpu()
matrix_min = min(initial_matrix.min().item(), trained_matrix.min().item())
matrix_max = max(initial_matrix.max().item(), trained_matrix.max().item())

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
for axis, matrix, title in zip(
    axes,
    (initial_matrix, trained_matrix),
    ("Matrix before training", "Matrix after training"),
    strict=True,
):
    image = axis.imshow(
        matrix.numpy(),
        aspect="auto",
        cmap="coolwarm",
        vmin=matrix_min,
        vmax=matrix_max,
    )
    axis.set_title(title)
    axis.set_xlabel("Input pixel")
axes[0].set_ylabel("Measurement row")
fig.colorbar(image, ax=axes, label="Matrix coefficient", shrink=0.8)
fig.tight_layout()
plt.show()

fig, axis = plt.subplots(figsize=(10, 3))
axis.plot(initial_matrix.norm(dim=1).numpy(), label="Before training")
axis.plot(trained_matrix.norm(dim=1).numpy(), label="After training")
axis.set_xlabel("Measurement row")
axis.set_ylabel(r"Row $\ell_2$ norm")
axis.set_title("Norm of every sensing-matrix row")
axis.legend()
fig.tight_layout()
plt.show()

# %%
# Evaluation
# ----------

trainer.test(test_dataloader)

sample = next(iter(test_dataloader)).to(device)
measurement = physics(sample)
with torch.no_grad():
    reconstruction = model(measurement, physics)

print(
    f"MSE: {dinv.metric.MSE()(reconstruction, sample).mean().item():.4f} | "
    f"PSNR: {dinv.metric.PSNR()(reconstruction, sample).mean().item():.2f} dB"
)

dinv.utils.plot(
    {
        "Ground truth": sample,
        "Measurement adjoint": physics.A_adjoint(measurement),
        "Reconstruction": reconstruction,
    },
    save_dir=base_dir / "results" / "co_design",
)
