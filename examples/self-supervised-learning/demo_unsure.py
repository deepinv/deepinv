r"""
Self-supervised denoising with the UNSURE loss.
====================================================================================================

This example shows you how to train a denoiser network in a fully self-supervised way,
i.e., using noisy images with unknown noise level only via the UNSURE loss, which is introduced by :footcite:t:`tachella2024unsure`.

The UNSURE optimization problem for Gaussian denoising with unknown noise level is defined as:

.. math::

    \min_{R} \max_{\sigma^2} \frac{1}{m}\|y-\inverse{y}\|_2^2 +\frac{2\sigma^2}{m\tau}b^{\top} \left(\inverse{y+\tau b}-\inverse{y}\right)

where :math:`R` is the trainable network, :math:`y` is the noisy image with :math:`m` pixels,
:math:`b\sim \mathcal{N}(0,1)` is a Gaussian random variable,
:math:`\tau` is a small positive number, and :math:`\odot` is an elementwise multiplication.

"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import deepinv as dinv
from deepinv.utils.demo import get_data_home

# %%
# Setup paths for data loading and results.
# ---------------------------------------------------------------
#

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "measurements"
CKPT_DIR = BASE_DIR / "ckpts"
ORIGINAL_DATA_DIR = get_data_home()

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# Load base image datasets
# ----------------------------------------------------------------------------------
# In this example, we use the MNIST dataset as the base image dataset.
#

operation = "denoising"
train_dataset_name = "MNIST"

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(
    root=ORIGINAL_DATA_DIR, train=True, transform=transform, download=True
)
test_dataset = datasets.MNIST(
    root=ORIGINAL_DATA_DIR, train=False, transform=transform, download=True
)

# %%
# Generate a dataset of noisy images
# ----------------------------------------------------------------------------------
#
# We generate a dataset of noisy images corrupted by Gaussian noise.
#
# .. note::
#
#       We use a subset of the whole training set to reduce the computational load of the example.
#       We recommend to use the whole set by setting ``n_images_max=None`` to get the best results.

true_sigma = 0.1

# defined physics
physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma=true_sigma))

# Use parallel dataloader if using a GPU to speed up training,
# otherwise, as all computes are on CPU, use synchronous data loading.
num_workers = 4 if torch.cuda.is_available() else 0

n_images_max = (
    100 if torch.cuda.is_available() else 5
)  # number of images used for training

measurement_dir = DATA_DIR / train_dataset_name / operation
deepinv_datasets_path = dinv.datasets.generate_dataset(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    physics=physics,
    device=device,
    save_dir=measurement_dir,
    train_datapoints=n_images_max,
    test_datapoints=n_images_max,
    num_workers=num_workers,
    dataset_filename="demo_sure",
)

train_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=True)
test_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=False)

# %%
# Set up the denoiser network
# ---------------------------------------------------------------
#
# We use a simple U-Net architecture with 2 scales as the denoiser network.

model = dinv.models.ArtifactRemoval(
    dinv.models.UNet(in_channels=1, out_channels=1, scales=2).to(device)
)


# %%
# Set up the training parameters
# --------------------------------------------
# We set :class:`deepinv.loss.SureGaussianLoss` as the training loss with the ``unsure=True`` option.
# The optimization with respect to the noise level is done by stochastic gradient descent with momentum
# inside the loss class, so it is seamlessly integrated into the training process.
#
# .. note::
#
#       There are (UN)SURE losses for various noise distributions. See also :class:`deepinv.loss.SurePGLoss` for mixed Poisson-Gaussian noise.
#
# .. note::
#
#       We train for only 10 epochs to reduce the computational load of the example. We recommend to train for more epochs to get the best results.
#

epochs = 10  # choose training epochs
learning_rate = 5e-4
batch_size = 32 if torch.cuda.is_available() else 1

sigma_init = 0.05  # initial guess for the noise level
step_size = 1e-4  # step size for the optimization of the noise level
momentum = 0.9  # momentum for the optimization of the noise level

# choose self-supervised training loss
loss = dinv.loss.SureGaussianLoss(
    sigma=sigma_init, unsure=True, step_size=step_size, momentum=momentum
)

# choose optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)

print(f"INIT. noise level {loss.sigma2.sqrt().item():.3f}")

# %%
# Train the network
# --------------------------------------------
# We train the network using the :class:`deepinv.Trainer` class.
#

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
)

# Initialize the trainer
trainer = dinv.Trainer(
    model=model,
    physics=physics,
    epochs=epochs,
    losses=loss,
    optimizer=optimizer,
    device=device,
    train_dataloader=train_dataloader,
    plot_images=False,
    save_path=str(CKPT_DIR / operation),
    verbose=True,  # print training information
    show_progress_bar=False,  # disable progress bar for better vis in sphinx gallery.
)

# Train the network
model = trainer.train()


# %%
# Check learned noise level
# --------------------------------------------
# We can verify the learned noise level by checking the estimated noise level from the loss function.
#

est_sigma = loss.sigma2.sqrt().item()

print(f"LEARNED noise level {est_sigma:.3f}")
print(f"Estimation error noise level {abs(est_sigma-true_sigma):.3f}")


# %%
# Test the network
# --------------------------------------------
#

test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)

trainer.plot_images = True
trainer.test(test_dataloader=test_dataloader)

# %%
# :References:
#
# .. footbibliography::
