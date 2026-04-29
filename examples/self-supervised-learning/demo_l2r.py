r"""
Self-supervised denoising with the Learning to Recorrupt (L2R) loss.
====================================================================================================

This example shows how to train a denoiser network in a fully self-supervised way,
using only noisy images through the Learning to Recorrupt (L2R) loss
:footcite:p:`monroy2026learning`, without requiring explicit knowledge of the noise distribution.

The core idea of L2R is to avoid comparing predictions to clean targets (which are unavailable in
self-supervised settings). Instead, the method learns a small trainable re-corruption module that
maps input noisy image to desired recorruption distribution. For this, the recorrupted image :math:`y_1`
is constructed by applying the recorruption network as follows

.. math::

    y_1 = y + \alpha h(\omega, y),

then, the L2R loss is defined as

.. math::

    \mathcal{L}_{\mathrm{L2R}}(f,h)
    = \mathbb{E}_{y\sim p(y)}\left[  \|AR(y_1) - y\|_2^2 + \frac{2}{\alpha} h(\omega, y)^{\top} (A R(y_1) )  \right],

and optimize it through the adversarial objective

.. math::

    \min_{f}\;\max_{h}\;\mathcal{L}_{\mathrm{L2R}}(f,h),

where :math:`f` is the denoiser, :math:`h` is the learned re-corruption model,
:math:`y` is the noisy measurement. Here, the denoiser is encouraged to align predictions with
noisy obervations and reduce noise correlation with the input noisy image, while the re-corruption model is trained
to maximize this noise correlation.

To build measurements, we choose a noise model in the physics simulator. By default,
this demo uses Poisson noise, but you can switch to Gaussian noise by changing ``noise_name``.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import deepinv as dinv
from deepinv.loss.l2r import Learning2RecorruptLoss
from deepinv.utils import get_data_home

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

device = dinv.utils.get_device()
print(device)

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
# --------------------------------------------------------------------------------------------------
#
# We use Poisson noise by default to generate noisy measurements.
# You can switch to Gaussian noise by setting ``noise_name = "gaussian"``.
#
# .. note::
#
#       We use a subset of the whole training set to reduce the computational load of the example.
#       We recommend to use the whole set by setting ``n_images_max=None`` to get the best results.

predefined_noise_models = dict(
    gaussian=dinv.physics.GaussianNoise(sigma=0.1),
    poisson=dinv.physics.PoissonNoise(gain=0.5),
)

noise_name = "poisson"  # default noise model for this demo
noise_model = predefined_noise_models[noise_name]
physics = dinv.physics.Denoising(noise_model)
operation = f"{operation}_{noise_name}_l2r"

# Use parallel dataloader if using a GPU to speed up training,
# otherwise, as all computes are on CPU, use synchronous data loading.
num_workers = 0 if torch.cuda.is_available() else 0

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
    dataset_filename="demo_l2r",
)

train_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=True)
test_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=False)

# %%
# Set up the denoiser network
# ---------------------------------------------------------------
#
# We use a simple U-Net architecture with 2 scales as the denoiser network.

model = dinv.models.ArtifactRemoval(
    dinv.models.UNet(in_channels=1, out_channels=1, scales=2, residual=False).to(device)
)


# %%
# Set up the training parameters
# --------------------------------------------
# We set :class:`deepinv.loss.l2r.Learning2RecorruptLoss` as the training loss.
#
# .. note::
#
#       L2R learns an internal trainable re-corruption network and does not require
#       explicit knowledge of the measurement noise distribution during optimization.

epochs = 3  # choose training epochs
learning_rate = 1e-3
batch_size = 64 if torch.cuda.is_available() else 1

# choose self-supervised training loss
loss = Learning2RecorruptLoss(metric=torch.nn.MSELoss(), alpha=0.5, eval_n_samples=2)
model = loss.adapt_model(model).to(device)  # important step!


# choose optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.8) + 1)


# %%
# Train the network
# --------------------------------------------
# To simulate a realistic self-supervised learning scenario, we do not use any supervised metrics for training,
# such as PSNR or SSIM, which require clean ground truth images.
#
# .. tip::
#
#       We can use the same self-supervised loss for evaluation, as it does not require clean images,
#       to monitor the training process (e.g. for early stopping). This is done automatically when `metrics=None` and `early_stop>0` in the trainer.


verbose = True  # print training information

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)

test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)

# Initialize the trainer
trainer = dinv.Trainer(
    model=model,
    physics=physics,
    epochs=epochs,
    scheduler=scheduler,
    losses=loss,
    optimizer=optimizer,
    device=device,
    metrics=dinv.metric.PSNR(),
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    early_stop=100,  # early stop using the self-supervised loss on the test set
    compute_eval_losses=True,  # use self-supervised loss for evaluation
    early_stop_on_losses=False,  # stop using self-supervised eval loss
    plot_images=True,
    plot_interval=100,
    save_path=str(CKPT_DIR / operation),
    verbose=verbose,
    show_progress_bar=False,  # disable progress bar for better vis in sphinx gallery.
)

# Train the network
model = trainer.train()


# %%
# Test the network
# --------------------------------------------
# We now assume that we have access to a small test set of clean images to evaluate the performance of the trained network,
# and we compute the PSNR between denoised images and clean ground truth images.
#
trainer.test(test_dataloader, metrics=dinv.metric.PSNR())

# %%
# :References:
#
# .. footbibliography::
