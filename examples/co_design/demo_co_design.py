r"""
End-to-end co-design baseline
=============================

This example shows how to train the parameters of the physics and the
reconstructor jointly using end-to-end training. :footcite:t:`arguello2023deep`

Consider the forward model

.. math::

    y = \operatorname{noise}(\operatorname{forw}(x, \theta)),

where :math:`A` is the physics operator, :math:`N` models the noise, and
:math:`\theta` is the trainable physics parameter. The reconstructor is
defined as

.. math::

    \hat{x} = R(y, A, \alpha).

The goal is to learn :math:`\theta` jointly with the reconstructor by solving

.. math::

    \min_{\theta,R} \frac{1}{2}
    \left\|R(\operatorname{forw}(x, \theta)) - x\right\|^2
    + \operatorname{Reg}(\theta),

where :math:`\operatorname{Reg}` is a regularizer for the physics parameter.

This example jointly trains the complete
:class:`deepinv.physics.CompressedSensing` matrix and an
:class:`deepinv.models.ArtifactRemoval` model to reconstruct MNIST images.
The supervised MSE loss is combined with a binary regularizer on the sensing
matrix.

The sensing matrix and reconstruction network are optimized together from the
same end-to-end objective.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import random
import numpy as np

import deepinv as dinv
from deepinv.utils import get_cache_home

# %%
# Imports and setup
# -----------------


# %%
# Setup paths for data loading and results.

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "measurements"
CKPT_DIR = BASE_DIR / "ckpts"
ORIGINAL_DATA_DIR = get_cache_home() / "datasets" / "MNIST"

# Set the global random seed from pytorch to ensure reproducibility of the example.
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Important: Enable gradient computation for the sensing matrix to allow co-design optimization.

device = dinv.utils.get_device()


# %%
# Load base image datasets
# ------------------------
#
# In this example, we use the MNIST dataset as the base image dataset.

train_dataset_name = "MNIST"

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(
    root=ORIGINAL_DATA_DIR, train=True, transform=transform, download=True
)
test_dataset = datasets.MNIST(
    root=ORIGINAL_DATA_DIR, train=False, transform=transform, download=True
)


# Wrapper to return only images (no labels) for the deepinv trainer.
class ImageOnlyDataset(torch.utils.data.Dataset):
    """Wrapper dataset that returns only images, discarding labels."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image


# Wrap datasets to return only images.
train_dataset = ImageOnlyDataset(train_dataset)
test_dataset = ImageOnlyDataset(test_dataset)

# Create dataloaders.
batch_size = 256
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# %%
# Define the physics
# ------------------
#
# In this example, we use :class:`deepinv.physics.CompressedSensing` as the
# forward model. The operator acquires 196 measurements for each 28 x 28
# image, corresponding to 25% of the pixels. The complete 196 x 784 sensing
# matrix is trainable and measurements are generated online during training.

image_shape = (1, 28, 28)
num_measurements = 196

physics = dinv.physics.CompressedSensing(
    m=num_measurements,
    img_size=image_shape,
    device=device,
    rng=torch.Generator(device=device).manual_seed(0),
)


initial_matrix_np = torch.empty_like(physics._A).uniform_(
    -0.1 / (num_measurements**0.5),
    0.1 / (num_measurements**0.5),
    generator=physics.rng,
)

physics.update(
    _A=initial_matrix_np
)  # Reparameterize the sensing matrix to enforce binary constraints.
physics._A.requires_grad_(
    True
)  # Enable gradient computation for co-design optimization.

# Convert to CPU and NumPy for visualization later.
initial_matrix_np = initial_matrix_np.detach().cpu().numpy()

# Get a sample from the dataloader (now it returns only images).
sample = next(iter(train_dataloader)).to(device)
measurement = physics(sample)
print("Image shape:      ", tuple(sample.shape))
print("Measurement shape:", tuple(measurement.shape))


# %%
# Reconstruction Algorithm
# ------------------------
#
# In this example, we use :class:`deepinv.models.ArtifactRemoval` as the
# reconstruction model. It first applies the adjoint of the physics operator
# and then uses a trainable U-Net to remove reconstruction artifacts.

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

with torch.no_grad():
    initial_reconstruction = model(measurement, physics)
print("Initial reconstruction shape:", tuple(initial_reconstruction.shape))


# %%
# Define the Loss and Regularization
# -----------------------------------
#
# In this example, we define a combined loss function: ``SupLoss`` provides
# the supervised mean squared error (MSE) term, while a custom binary
# regularizer encourages the sensing matrix coefficients to converge to -1 or
# +1.


from deepinv.loss import BinaryRegularization

# Define the combined loss function
losses = [
    dinv.loss.SupLoss(metric=dinv.metric.MSE()),  # Supervised MSE loss
    BinaryRegularization(m=num_measurements, weight=1e3),
]

# Save initial sensing matrix for later comparison


# Set up optimizer with different learning rates for different components
optimizer = torch.optim.Adam(
    [
        {"params": model.parameters(), "lr": 1e-3},  # Reconstruction network
        {"params": [physics._A], "lr": 5e-2},  # Sensing matrix (slower learning)
    ]
)


# %%
# Train
# -----
#
# The trainer creates a new measurement from the current MNIST image at every
# iteration by calling ``physics(x)``, since ``online_measurements=True``. The
# optimizer updates both the ArtifactRemoval network and the complete sensing
# matrix.

epochs = 30

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
    verbose=True,
    show_progress_bar=False,
)

model = trainer.train()


# %%
# Visualization of Learned Sensing Patterns
# ------------------------------------------
#
# Visualize a few selected measurement rows from the trained sensing matrix
# reshaped as 28 x 28 images, showing how the network has learned to acquire
# informative measurements.

# Visualize before/after: 4 rows x 2 columns (Initial vs Trained).
trained_matrix_np = physics._A.detach().cpu().numpy()

# Select 4 rows evenly distributed across the 196 measurements.
selected_rows = [0, 65, 130, 195]

fig, axes = plt.subplots(4, 2, figsize=(10, 12))

for row_idx, (ax_init, ax_trained) in enumerate(zip(axes[:, 0], axes[:, 1])):
    measurement_idx = selected_rows[row_idx]

    # Initial pattern (left column).
    initial_pattern = initial_matrix_np[measurement_idx].reshape(28, 28)
    initial_pattern = initial_pattern * (num_measurements**0.5)
    im_init = ax_init.imshow(initial_pattern, cmap="gray")
    ax_init.set_title(f"Initial Row {measurement_idx}", fontsize=10, fontweight="bold")
    ax_init.axis("off")
    plt.colorbar(im_init, ax=ax_init, fraction=0.046, pad=0.04)

    # Trained pattern (right column).
    trained_pattern = trained_matrix_np[measurement_idx].reshape(28, 28)
    trained_pattern = trained_pattern * (num_measurements**0.5)
    im_trained = ax_trained.imshow(trained_pattern, cmap="gray")
    ax_trained.set_title(
        f"Trained Row {measurement_idx}", fontsize=10, fontweight="bold"
    )
    ax_trained.axis("off")
    plt.colorbar(im_trained, ax=ax_trained, fraction=0.046, pad=0.04)

fig.suptitle(
    "Sensing Matrix Evolution: Before vs After Training",
    fontsize=14,
    fontweight="bold",
    y=0.995,
)
fig.tight_layout()
plt.show()


# %%
# Evaluate the reconstruction
# ----------------------------

model.eval()

# Test on a sample image.
sample_test = next(iter(test_dataloader)).to(device)
measurement = physics(sample_test)
with torch.no_grad():
    reconstruction = model(measurement, physics)

# Compute metrics.
mse = dinv.metric.MSE()(reconstruction, sample_test).mean().item()
psnr = dinv.metric.PSNR()(reconstruction, sample_test).mean().item()
print(f"Test Results - MSE: {mse:.4f} | PSNR: {psnr:.2f} dB")

# Visualize the reconstruction result.
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(sample_test[0, 0].cpu().numpy(), cmap="gray")
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(physics.A_adjoint(measurement)[0, 0].detach().cpu().numpy(), cmap="gray")
axes[1].set_title("Measurement Adjoint (Zero-filled)")
axes[1].axis("off")

axes[2].imshow(reconstruction[0, 0].detach().cpu().numpy(), cmap="gray")
axes[2].set_title(f"Reconstruction\nPSNR: {psnr:.2f} dB")
axes[2].axis("off")

fig.tight_layout()
plt.show()
