r"""
Self-supervised learning from binary measurements.
====================================================================================================

This example shows you how to train a reconstruction network for an MRI inverse problem on a fully self-supervised way,
i.e., using measurement data only.

The dataset consists of pairs :math:`(y_i,A_{g_i})` where :math:`y_i` are the measurements and :math:`A_{g_i}` is a
binary sampling operator out of :math:`G` (i.e., :math:`g\in \{1,\dots,G\}`.

This self-supervised learning approach is presented in `"Unsupervised Learning From Incomplete Measurements for Inverse Problems"
 <https://openreview.net/pdf?id=aV9WSvM6N3>`_, and minimizes the loss function:

.. math::

    \mathcal{L}(\theta) = \sum_{i=1}^{N} \left\|A_{g_i} f_{\theta}(y_i,A_{g_i}) - y_i \right\|_2^2 +
    \left\|\hat{x}_i - f_{\theta}(A_s\hat{x}_i,A_s) \right\|_2^2

where :math:`f_{\theta}` is a reconstruction network with parameters :math:`\theta`, :math:`y_i` are the measurements,
:math:`A_s` is a binary sampling operator, and :math:`\hat{x}_i = f_{\theta}(y_i,A_{g_i})`.

"""

import deepinv as dinv
from torch.utils.data import DataLoader
import torch
from pathlib import Path
from torchvision import transforms
from deepinv.utils.demo import load_degradation
from deepinv.training_utils import train, test
from torchvision import datasets

# %%
# Setup paths for data loading and results.
# ---------------------------------------------------------------
#

BASE_DIR = Path(".")
ORIGINAL_DATA_DIR = BASE_DIR / "datasets"
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
DEG_DIR = BASE_DIR / "degradations"
CKPT_DIR = BASE_DIR / "ckpts"

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# Load base image datasets and degradation operators.
# ----------------------------------------------------------------------------------
# In this example, we use the MNIST dataset for training and testing.
#

transform = transforms.Compose([transforms.ToTensor()])

train_base_dataset = datasets.MNIST(root='../datasets/', train=True, transform=transform, download=True)
test_base_dataset = datasets.MNIST(root='../datasets/', train=False, transform=transform, download=True)

# %%
# Generate a dataset of knee images and load it.
# ----------------------------------------------------------------------------------
# We generate 10 different inpainting operators, each one with a different random mask.
# If the :func:`dinv.datasets.generate_dataset` receives a list of physics operators, it
# generates a dataset for each operator and returns a list of paths to the generated datasets.
#
# .. note::
#
#    We only use 10 training images to reduce the computational time of this example. You can use the whole
#   dataset by setting ``n_images_max = None``.

number_of_operators = 10

# defined physics
physics = [dinv.physics.Inpainting(mask=.5, tensor_size=(1, 28, 28),
                                          device=device) for _ in range(number_of_operators)]

# Use parallel dataloader if using a GPU to fasten training,
# otherwise, as all computes are on CPU, use synchronous data loading.
num_workers = 4 if torch.cuda.is_available() else 0
n_images_max = (
    100 if torch.cuda.is_available() else 5
)  # number of images used for training
# (the dataset has up to 973 images, however here we use only 100)

operation = "CS"
my_dataset_name = "demo_multioperator_imaging"
measurement_dir = DATA_DIR / "MNIST" / operation
deepinv_datasets_path = dinv.datasets.generate_dataset(
    train_dataset=train_base_dataset,
    test_dataset=test_base_dataset,
    physics=physics,
    device=device,
    save_dir=measurement_dir,
    train_datapoints=n_images_max,
    num_workers=num_workers,
    dataset_filename=str(my_dataset_name),
)

train_dataset = [dinv.datasets.HDF5Dataset(path=path, train=True) for path in deepinv_datasets_path]
test_dataset = [dinv.datasets.HDF5Dataset(path=path, train=False) for path in deepinv_datasets_path]

# %%
# Set up the reconstruction network
# ---------------------------------------------------------------
#
# As a reconstruction network, we use a simple artifact removal network based on a U-Net.
# The network is defined as a :math:`f(y,A)=\phi(A^{\top}y)` where :math:`\phi` is the U-Net.

# Define the unfolded trainable model.
model = dinv.models.ArtifactRemoval(backbone_net=dinv.models.UNet(in_channels=1, out_channels=1, scales=3))
model = model.to(device)

# %%
# Set up the training parameters
# --------------------------------------------
# We choose a self-supervised training scheme with two losses: the measurement consistency loss (MC)
# and the multi-operator imaging loss (MOI).
# Necessary and sufficient conditions on the number of operators and measurements are described
# `here <https://www.jmlr.org/papers/v24/22-0315.html>`_.
#
# .. note::
#
#       We use a pretrained model to reduce training time. You can get the same results by training from scratch
#       for 100 epochs.

epochs = 1  # choose training epochs
learning_rate = 5e-4
batch_size = 64 if torch.cuda.is_available() else 1

# choose self-supervised training losses
# generates 4 random rotations per image in the batch
losses = [dinv.loss.MCLoss(), dinv.loss.MOILoss()]

# choose optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.8) + 1)

# start with a pretrained model to reduce training time
#url = online_weights_path() + "demo_ei_ckp_150.pth"
#ckpt = torch.hub.load_state_dict_from_url(
#    url, map_location=lambda storage, loc: storage, file_name="demo_ei_ckp_150.pth"
#)
# load a checkpoint to reduce training time
#model.load_state_dict(ckpt["state_dict"])
#optimizer.load_state_dict(ckpt["optimizer"])

# %%
# Train the network
# --------------------------------------------
#
#


verbose = True  # print training information
wandb_vis = False  # plot curves and images in Weight&Bias

train_dataloader = [DataLoader(
    dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
) for dataset in train_dataset]
test_dataloader = [DataLoader(
    dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
) for dataset in test_dataset]

train(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    epochs=epochs,
    scheduler=scheduler,
    losses=losses,
    physics=physics,
    optimizer=optimizer,
    device=device,
    save_path=str(CKPT_DIR / operation),
    verbose=verbose,
    wandb_vis=wandb_vis,
    log_interval=1,
    eval_interval=1,
    ckp_interval=10,
)

# %%
# Test the network
# --------------------------------------------
#
#

plot_images = True
save_images = True
method = "multioperator_imaging"

test(
    model=model,
    test_dataloader=test_dataloader,
    physics=physics,
    device=device,
    plot_images=plot_images,
    save_images=save_images,
    save_folder=RESULTS_DIR / method / operation,
    verbose=verbose,
    wandb_vis=wandb_vis,
)
