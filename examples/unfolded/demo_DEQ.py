r"""
Deep Equilibrium (DEQ) algorithms for image deblurring
====================================================================================================

This a toy example to show you how to use DEQ to solve a deblurring problem. 
Note that this is a small dataset for training. For optimal results, use a larger dataset.
For visualizing the training, you can use Weight&Bias (wandb) by setting ``wandb_vis=True``.

For now DEQ is only possible with PGD, HQS and GD optimization algorithms. 

"""

import deepinv as dinv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from deepinv.models import DnCNN
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.unfolded import DEQ_builder
from deepinv.training import train, test
from torchvision import transforms
from deepinv.utils.demo import load_dataset

# %%
# Setup paths for data loading and results.
# ----------------------------------------------------------------------------------------
#

BASE_DIR = Path(".")
ORIGINAL_DATA_DIR = BASE_DIR / "datasets"
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
CKPT_DIR = BASE_DIR / "ckpts"

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# Load base image datasets and degradation operators.
# ----------------------------------------------------------------------------------------
# In this example, we use the CBSD500 dataset and the Set3C dataset for testing.

img_size = 32
n_channels = 3  # 3 for color images, 1 for gray-scale images
operation = "deblurring"
# For simplicity, we use a small dataset for training.
# To be replaced for optimal results. For example, you can use the larger "drunet" dataset.
train_dataset_name = "CBSD500"
test_dataset_name = "set3c"
# Generate training and evaluation datasets in HDF5 folders and load them.
test_transform = transforms.Compose(
    [transforms.CenterCrop(img_size), transforms.ToTensor()]
)
train_transform = transforms.Compose(
    [transforms.RandomCrop(img_size), transforms.ToTensor()]
)
train_base_dataset = load_dataset(
    train_dataset_name, ORIGINAL_DATA_DIR, transform=train_transform
)
test_base_dataset = load_dataset(
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
noise_level_img = 0.03

# Generate the gaussian blur downsampling operator.
physics = dinv.physics.BlurFFT(
    img_size=(n_channels, img_size, img_size),
    filter=dinv.physics.blur.gaussian_blur(),
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
)
my_dataset_name = "demo_DEQ"
n_images_max = (
    1000 if torch.cuda.is_available() else 10
)  # maximal number of images used for training
measurement_dir = DATA_DIR / train_dataset_name / operation
generated_datasets_path = dinv.datasets.generate_dataset(
    train_dataset=train_base_dataset,
    test_dataset=test_base_dataset,
    physics=physics,
    device=device,
    save_dir=measurement_dir,
    train_datapoints=n_images_max,
    num_workers=num_workers,
    dataset_filename=str(my_dataset_name),
)

train_dataset = dinv.datasets.HDF5Dataset(path=generated_datasets_path, train=True)
test_dataset = dinv.datasets.HDF5Dataset(path=generated_datasets_path, train=False)

# %%
# Define the  DEQ algorithm.
# ----------------------------------------------------------------------------------------
# We use the helper function :meth:`deepinv.unfolded.DEQ_builder` to defined the DEQ architecture.
# The chosen algorithm is here HQS (Half Quadratic Splitting).
# Note for DEQ, the prior and regularization parameters should be common for all iterations
# to keep a constant fixed-point operator.


# Select the data fidelity term
data_fidelity = L2()

# Set up the trainable denoising prior
denoiser = DnCNN(
    in_channels=3, out_channels=3, depth=7, device=device, pretrained=None, train=True
)

# Here the prior model is common for all iterations
prior = PnP(denoiser=denoiser)

# Unrolled optimization algorithm parameters
max_iter = 20 if torch.cuda.is_available() else 10
stepsize = 1.0  # Initial value for the stepsize. A single stepsize is common for each iterations.
sigma_denoiser = 0.03  # Initial value for the denoiser parameter. A single value is common for each iterations.
anderson_acceleration_forward = True  # use Anderson acceleration for the forward pass.
anderson_acceleration_backward = (
    True  # use Anderson acceleration for the backward pass.
)
anderson_history_size = (
    5 if torch.cuda.is_available() else 3
)  # history size for Anderson acceleration.

params_algo = {  # wrap all the restoration parameters in a 'params_algo' dictionary
    "stepsize": stepsize,
    "g_param": sigma_denoiser,
}
trainable_params = [
    "stepsize",
    "g_param",
]  # define which parameters from 'params_algo' are trainable

# Define the unfolded trainable model.
model = DEQ_builder(
    iteration="HQS",  # For now DEQ is only possible with PGD, HQS and GD optimization algorithms.
    params_algo=params_algo.copy(),
    trainable_params=trainable_params,
    data_fidelity=data_fidelity,
    max_iter=max_iter,
    prior=prior,
    anderson_acceleration=anderson_acceleration_forward,
    anderson_acceleration_backward=anderson_acceleration_backward,
    history_size_backward=anderson_history_size,
    history_size=anderson_history_size,
)

# %%
# Define the training parameters.
# -------------------------------
# We use the Adam optimizer and the StepLR scheduler.


# training parameters
epochs = 10
learning_rate = 5e-4
train_batch_size = 32 if torch.cuda.is_available() else 1
test_batch_size = 3

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
# -----------------
# We train the network using the library's train function.

trainer = dinv.Trainer(
    model=model,
    physics=physics,
    epochs=epochs,
    scheduler=scheduler,
    device=device,
    losses=losses,
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    save_path=str(CKPT_DIR / operation),
    verbose=verbose,
    show_progress_bar=False,  # disable progress bar for better vis in sphinx gallery.
    wandb_vis=wandb_vis,  # training visualization can be done in Weight&Bias
)

model = trainer.train()

# %%
# Test the network
# --------------------------------------------
#
#

trainer.test(test_dataloader)
