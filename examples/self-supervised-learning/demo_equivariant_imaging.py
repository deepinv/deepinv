r"""
Self-supervised learning with Equivariant Imaging for MRI.
====================================================================================================

This example shows you how to train a reconstruction network for an MRI inverse problem on a fully self-supervised way, i.e., using measurement data only.

The equivariant imaging loss is presented in `"Equivariant Imaging: Learning Beyond the Range Space"
<http://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Equivariant_Imaging_Learning_Beyond_the_Range_Space_ICCV_2021_paper.pdf>`_.

"""

import deepinv as dinv
from torch.utils.data import DataLoader
import torch
from pathlib import Path
from torchvision import transforms
from deepinv.optim.prior import PnP
from deepinv.utils.demo import load_dataset, load_degradation
from deepinv.models.utils import get_weights_url

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
# In this example, we use a subset of the single-coil `FastMRI dataset <https://fastmri.org/>`_
# as the base image dataset. It consists of 973 knee images of size 320x320.
#
# .. note::
#
#       We reduce to the size to 128x128 for faster training in the demo.
#

operation = "MRI"
train_dataset_name = "fastmri_knee_singlecoil"
img_size = 128

transform = transforms.Compose([transforms.Resize(img_size)])

train_dataset = load_dataset(
    train_dataset_name, ORIGINAL_DATA_DIR, transform, train=True
)
test_dataset = load_dataset(
    train_dataset_name, ORIGINAL_DATA_DIR, transform, train=False
)

# %%
# Generate a dataset of knee images and load it.
# ----------------------------------------------------------------------------------
#
#

mask = load_degradation("mri_mask_128x128.npy", ORIGINAL_DATA_DIR)

# defined physics
physics = dinv.physics.MRI(mask=mask, device=device)

# Use parallel dataloader if using a GPU to fasten training,
# otherwise, as all computes are on CPU, use synchronous data loading.
num_workers = 4 if torch.cuda.is_available() else 0
n_images_max = (
    900 if torch.cuda.is_available() else 5
)  # number of images used for training
# (the dataset has up to 973 images, however here we use only 900)

my_dataset_name = "demo_equivariant_imaging"
measurement_dir = DATA_DIR / train_dataset_name / operation
deepinv_datasets_path = dinv.datasets.generate_dataset(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    physics=physics,
    device=device,
    save_dir=measurement_dir,
    train_datapoints=n_images_max,
    num_workers=num_workers,
    dataset_filename=str(my_dataset_name),
)

train_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=True)
test_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=False)

# %%
# Set up the reconstruction network
# ---------------------------------------------------------------
#
# As a reconstruction network, we use an unrolled network (half-quadratic splitting)
# with a trainable denoising prior based on the DnCNN architecture.

# Select the data fidelity term
data_fidelity = dinv.optim.L2()
n_channels = 2  # real + imaginary parts

# If the prior dict value is initialized with a table of length max_iter, then a distinct model is trained for each
# iteration. For fixed trained model prior across iterations, initialize with a single model.
prior = PnP(
    denoiser=dinv.models.DnCNN(
        in_channels=n_channels,
        out_channels=n_channels,
        pretrained=None,
        train=True,
        depth=7,
    ).to(device)
)

# Unrolled optimization algorithm parameters
max_iter = 3  # number of unfolded layers
lamb = [1.0] * max_iter  # initialization of the regularization parameter
stepsize = [1.0] * max_iter  # initialization of the step sizes.
sigma_denoiser = [0.01] * max_iter  # initialization of the denoiser parameters
params_algo = {  # wrap all the restoration parameters in a 'params_algo' dictionary
    "stepsize": stepsize,
    "g_param": sigma_denoiser,
    "lambda": lamb,
}

trainable_params = [
    "lambda",
    "stepsize",
    "g_param",
]  # define which parameters from 'params_algo' are trainable

# Define the unfolded trainable model.
model = dinv.unfolded.unfolded_builder(
    "HQS",
    params_algo=params_algo,
    trainable_params=trainable_params,
    data_fidelity=data_fidelity,
    max_iter=max_iter,
    prior=prior,
)


# %%
# Set up the training parameters
# --------------------------------------------
# We choose a self-supervised training scheme with two losses: the measurement consistency loss (MC)
# and the equivariant imaging loss (EI).
# The EI loss requires a group of transformations to be defined. The forward model `should not be equivariant to
# these transformations <https://www.jmlr.org/papers/v24/22-0315.html>`_.
# Here we use the group of 4 rotations of 90 degrees, as the accelerated MRI acquisition is
# not equivariant to rotations (while it is equivariant to translations).
#
# .. note::
#
#       We use a pretrained model to reduce training time. You can get the same results by training from scratch
#       for 150 epochs.

epochs = 1  # choose training epochs
learning_rate = 5e-4
batch_size = 16 if torch.cuda.is_available() else 1

# choose self-supervised training losses
# generates 4 random rotations per image in the batch
losses = [dinv.loss.MCLoss(), dinv.loss.EILoss(dinv.transform.Rotate(4))]

# choose optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.8) + 1)

# start with a pretrained model to reduce training time
file_name = "new_demo_ei_ckp_150_v3.pth"
url = get_weights_url(model_name="demo", file_name=file_name)
ckpt = torch.hub.load_state_dict_from_url(
    url,
    map_location=lambda storage, loc: storage,
    file_name=file_name,
)
# load a checkpoint to reduce training time
model.load_state_dict(ckpt["state_dict"])
optimizer.load_state_dict(ckpt["optimizer"])

# %%
# Train the network
# --------------------------------------------
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
    model,
    physics=physics,
    epochs=epochs,
    scheduler=scheduler,
    losses=losses,
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    plot_images=True,
    device=device,
    save_path=str(CKPT_DIR / operation),
    verbose=verbose,
    wandb_vis=wandb_vis,
    show_progress_bar=False,  # disable progress bar for better vis in sphinx gallery.
    ckp_interval=10,
)

model = trainer.train()

# %%
# Test the network
# --------------------------------------------
#
#

trainer.test(test_dataloader)
