r"""
Self-supervised denoising with the SURE loss.
====================================================================================================

This example shows you how to train a denoiser network in a fully self-supervised way,
i.e., using noisy images only via the SURE loss, which exploits knowledge about the noise distribution.

The SURE loss for Poisson denoising acts as an unbiased estimator of the supervised loss and is computed as:


.. math::

    \frac{1}{m}\|y-\inverse{y}\|_2^2-\frac{\gamma}{m} 1^{\top}y
    +\frac{2\gamma}{m\tau}(b\odot y)^{\top} \left(\inverse{y+\tau b}-\inverse{y}\right)

where :math:`R` is the trainable network, :math:`y` is the noisy image with :math:`m` pixels,
:math:`b` is a Bernoulli random variable taking values of -1 and 1 each with a probability of 0.5,
:math:`\tau` is a small positive number, and :math:`\odot` is an elementwise multiplication.


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
# We generate a dataset of noisy images corrupted by Poisson noise.
#
# .. note::
#
#       We use a subset of the whole training set to reduce the computational load of the example.
#       We recommend to use the whole set by setting ``n_images_max=None`` to get the best results.

# defined physics
physics = dinv.physics.Denoising(dinv.physics.PoissonNoise(0.1))

# Use parallel dataloader if using a GPU to fasten training,
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
# We set :class:`deepinv.loss.SurePoissonLoss` as the training loss.
#
# .. note::
#
#       There are SURE losses for various noise distributions. See also :class:`deepinv.loss.SureGaussianLoss`
#       for Gaussian noise and :class:`deepinv.loss.SurePGLoss` for mixed Poisson-Gaussian noise.
#
# .. note::
#
#       We use a pretrained model to reduce training time. You can get the same results by training from scratch
#       for 10 epochs.

epochs = 1  # choose training epochs
learning_rate = 5e-4
batch_size = 32 if torch.cuda.is_available() else 1

# choose self-supervised training loss
loss = dinv.loss.SurePoissonLoss(gain=0.1)

# choose optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.8) + 1)

# start with a pretrained model to reduce training time
file_name = "ckp_10_demo_sure.pth"
url = get_weights_url(model_name="demo", file_name=file_name)
ckpt = torch.hub.load_state_dict_from_url(
    url, map_location=lambda storage, loc: storage, file_name=file_name
)
# load a checkpoint to reduce training time
model.load_state_dict(ckpt["state_dict"])
optimizer.load_state_dict(ckpt["optimizer"])

# %%
# Train the network
# --------------------------------------------
#
#

verbose = True  # print training information
wandb_vis = False  # plot curves and images in Weight&Bias

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
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
    eval_dataloader=test_dataloader,
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
