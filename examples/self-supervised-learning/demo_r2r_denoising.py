r"""
Self-supervised denoising with the Generalized R2R loss.
====================================================================================================

This example shows you how to train a denoiser network in a fully self-supervised way,
using noisy images only via the Generalized Recorrupted2Recorrupted (GR2R) loss :footcite:t:`monroy2025generalized`,
which exploits knowledge about the noise distribution. You can change the noise distribution by selecting
from predefined noise models such as Gaussian, Poisson, and Gamma noise.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import deepinv as dinv
from deepinv.utils.demo import get_data_home
from deepinv.models.utils import get_weights_url

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
# Generate a dataset of noisy images corrupted by Poisson noise.
# The predefined noise models in the physics module include Gaussian, Poisson, and Gamma noise.
# Here, we use Poisson noise as an example, but you can also use Gaussian or Gamma noise.
#
# .. note::
#
#       We use a subset of the whole training set to reduce the computational load of the example.
#       We recommend to use the whole set by setting ``n_images_max=None`` to get the best results.

# defined physics
predefined_noise_models = dict(
    gaussian=dinv.physics.GaussianNoise(sigma=0.1),
    poisson=dinv.physics.PoissonNoise(gain=0.5),
    gamma=dinv.physics.GammaNoise(l=10.0),
)

noise_name = "poisson"
noise_model = predefined_noise_models[noise_name]
physics = dinv.physics.Denoising(noise_model)
operation = f"{operation}_{noise_name}"

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
    dataset_filename="demo_r2r",
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
# We set :class:`deepinv.loss.R2RLoss` as the training loss.
#
# .. note::
#
#       There are GR2R losses for various noise distributions, which can be specified by the noise model.
#

epochs = 1  # choose training epochs
learning_rate = 1e-4
batch_size = 64 if torch.cuda.is_available() else 1

# choose self-supervised training loss
loss = dinv.loss.R2RLoss(noise_model=None)
model = loss.adapt_model(model)  # important step!

# choose optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.8) + 1)

# # start with a pretrained model to reduce training time

if noise_name == "poisson":
    file_name = "ckp_10_demo_r2r_poisson.pth"
    url = get_weights_url(model_name="demo", file_name=file_name)
    ckpt = torch.hub.load_state_dict_from_url(
        url, map_location=lambda storage, loc: storage, file_name=file_name
    )
    model.load_state_dict(ckpt["state_dict"])

# %%
# Train the network
# --------------------------------------------
#
#

verbose = True  # print training information
wandb_vis = False  # plot curves and images in Weight&Bias

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
    train_dataloader=train_dataloader,
    eval_dataloader=None,
    plot_images=True,
    save_path=str(CKPT_DIR / operation),
    verbose=verbose,
    show_progress_bar=False,  # disable progress bar for better vis in sphinx gallery.
    wandb_vis=wandb_vis,
)

# Train the network
model = trainer.train()


# %%
# Test the network
# --------------------------------------------
#
#

trainer.test(test_dataloader)

# %%
# :References:
#
# .. footbibliography::
