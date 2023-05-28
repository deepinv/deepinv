r"""
Unfolded algorithms for super-resolution
====================================================================================================

This example shows you how to use unfolded architectures to solve a super-resolution problem.

"""

import deepinv as dinv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from deepinv.models.denoiser import Denoiser
from deepinv.optim.data_fidelity import L2
from deepinv.unfolded import Unfolded
from deepinv.training_utils import train, test
from torchvision import transforms
from deepinv.utils.demo import load_dataset

# %%
# Setup paths for data loading and results.
# ----------------------------------------------------------------------------------------
#

BASE_DIR = Path("../plug-and-play")
ORIGINAL_DATA_DIR = BASE_DIR / "datasets"
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
CKPT_DIR = BASE_DIR / "ckpts"

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

# %%
# Load base image datasets and degradation operators.
# ----------------------------------------------------------------------------------------
# In this example, we use the CBSD68 dataset
# for training and the Set3C dataset for testing.

img_size = 128 if torch.cuda.is_available() else 32
n_channels = 3  # 3 for color images, 1 for gray-scale images
operation = "super-resolution"
train_dataset_name = "CBSD68"
test_dataset_name = "set3c"
# Generate training and evaluation datasets in HDF5 folders and load them.
test_transform = transforms.Compose(
    [transforms.CenterCrop(img_size), transforms.ToTensor()]
)
train_transform = transforms.Compose(
    [transforms.RandomCrop(img_size), transforms.ToTensor()]
)
train_dataset = load_dataset(
    train_dataset_name, ORIGINAL_DATA_DIR, transform=train_transform
)
test_dataset = load_dataset(
    test_dataset_name, ORIGINAL_DATA_DIR, transform=test_transform
)


# %%
# Generate a dataset of low resolution images and load it.
# ----------------------------------------------------------------------------------------
# We use the Downsampling class from the physics module to generate a dataset of low resolution images.


# Use parallel dataloader if using a GPU to fasten training, otherwise, as all computes are on CPU, use synchronous
# dataloading.
num_workers = 4 if torch.cuda.is_available() else 0

# Degradation parameters
factor = 2
noise_level_img = 0.03

# Generate the gaussian blur downsampling operator.
physics = dinv.physics.Downsampling(
    img_size=(n_channels, img_size, img_size),
    factor=factor,
    mode="gauss",
    device=dinv.device,
    noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
)
my_dataset_name = "demo_unfolded_sr"
n_images_max = (
    1000 if torch.cuda.is_available() else 10
)  # maximal number of images used for training
measurement_dir = DATA_DIR / train_dataset_name / operation
generated_datasets_path = dinv.datasets.generate_dataset(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    physics=physics,
    device=dinv.device,
    save_dir=measurement_dir,
    train_datapoints=n_images_max,
    num_workers=num_workers,
    dataset_filename=str(my_dataset_name),
)

train_dataset = dinv.datasets.HDF5Dataset(path=generated_datasets_path, train=True)
test_dataset = dinv.datasets.HDF5Dataset(path=generated_datasets_path, train=False)

# %%
# Define the unfolded PnP algorithm.
# ----------------------------------------------------------------------------------------
# We use the Unfolded class to define the unfolded PnP algorithm.
# For both 'stepsize' and 'g_param', if initialized with a table of length max_iter, then a distinct stepsize/g_param
# value is trained for each iteration. For fixed trained 'stepsize' and 'g_param' values across iterations,
# initialize them with a single float.

# Select the data fidelity term
data_fidelity = L2()

# Set up the trainable denoising prior
denoiser_spec = {
    "name": "dncnn",
    "args": {
        "in_channels": n_channels,
        "out_channels": n_channels,
        "depth": 7,
        "pretrained": None,
        "train": True,
        "device": dinv.device,
    },
}

# If the prior dict value is initialized with a table of lenght max_iter, then a distinct model is trained for each
# iteration. For fixed trained model prior across iterations, initialize with a single model.
prior = {
    "prox_g": Denoiser(denoiser_spec)
}  # here the prior model is common for all iterations

# Unrolled optimization algorithm parameters
max_iter = 5  # number of unfolded layers
lamb = [1.0] * max_iter  # initialization of the regularization parameter
stepsize = [1.0] * max_iter  # initialization of the stepsizes.
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
model = Unfolded(
    "DRS",
    params_algo=params_algo,
    trainable_params=trainable_params,
    data_fidelity=data_fidelity,
    max_iter=max_iter,
    prior=prior,
)

# %%
# Define the training parameters.
# ----------------------------------------------------------------------------------------
# We use the Adam optimizer and the StepLR scheduler.


# training parameters
epochs = 10 if torch.cuda.is_available() else 2
learning_rate = 5e-4
train_batch_size = 32 if torch.cuda.is_available() else 1
test_batch_size = 32 if torch.cuda.is_available() else 1

# choose optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.8))

# choose supervised training loss
losses = [dinv.loss.SupLoss(metric=dinv.metric.mse())]

# Logging parameters
verbose = True
wandb_vis = False  # plot curves and images in Weight&Bias

train_dataloader = DataLoader(
    train_dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True
)
test_dataloader = DataLoader(
    test_dataset, batch_size=test_batch_size, num_workers=num_workers, shuffle=False
)

# %%
# Train the network
# ----------------------------------------------------------------------------------------
# We train the network using the library's train function.

train(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    epochs=epochs,
    scheduler=scheduler,
    losses=losses,
    physics=physics,
    optimizer=optimizer,
    device=dinv.device,
    save_path=str(CKPT_DIR / operation),
    verbose=verbose,
    wandb_vis=wandb_vis,
)

# %%
# Test the network
# --------------------------------------------
#
#

plot_images = True
save_images = True
method = "unfolded_drs"

test(
    model=model,
    test_dataloader=test_dataloader,
    physics=physics,
    device=dinv.device,
    plot_images=plot_images,
    save_images=save_images,
    save_folder=RESULTS_DIR / method / operation,
    verbose=verbose,
    wandb_vis=wandb_vis,
)
