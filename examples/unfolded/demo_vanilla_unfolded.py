r"""
Vanilla Unfolded algorithm for super-resolution
====================================================================================================

This is a simple example to show how to use vanilla unfolded Plug-and-Play.
The DnCNN denoiser and the algorithm parameters (stepsize, regularization parameters) are trained jointly.
For simplicity, we show how to train the algorithm on a  small dataset. For optimal results, use a larger dataset.
For visualizing the training, you can use Weight&Bias (wandb) by setting ``wandb_vis=True``.
"""

import deepinv as dinv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.unfolded import unfolded_builder
from torchvision import transforms
from deepinv.utils.demo import load_dataset

# %%
# Setup paths for data loading and results.
# ----------------------------------------------------------------------------------------
#

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
CKPT_DIR = BASE_DIR / "ckpts"

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# Load base image datasets and degradation operators.
# ----------------------------------------------------------------------------------------
# In this example, we use the CBSD500 dataset for training and the Set3C dataset for testing.

img_size = 64 if torch.cuda.is_available() else 32
n_channels = 3  # 3 for color images, 1 for gray-scale images
operation = "super-resolution"

# %%
# Generate a dataset of low resolution images and load it.
# ----------------------------------------------------------------------------------------
# We use the Downsampling class from the physics module to generate a dataset of low resolution images.

# For simplicity, we use a small dataset for training.
# To be replaced for optimal results. For example, you can use the larger "drunet" dataset.
train_dataset_name = "CBSD500"
test_dataset_name = "set3c"
# Specify the  train and test transforms to be applied to the input images.
test_transform = transforms.Compose(
    [transforms.CenterCrop(img_size), transforms.ToTensor()]
)
train_transform = transforms.Compose(
    [transforms.RandomCrop(img_size), transforms.ToTensor()]
)
# Define the base train and test datasets of clean images.
train_base_dataset = load_dataset(train_dataset_name, transform=train_transform)
test_base_dataset = load_dataset(test_dataset_name, transform=test_transform)

# Use parallel dataloader if using a GPU to speed up training, otherwise, as all computes are on CPU, use synchronous
# dataloading.
num_workers = 4 if torch.cuda.is_available() else 0

# Degradation parameters
factor = 2
noise_level_img = 0.03

# Generate the gaussian blur downsampling operator.
physics = dinv.physics.Downsampling(
    filter="gaussian",
    img_size=(n_channels, img_size, img_size),
    factor=factor,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
)
my_dataset_name = "demo_unfolded_sr"
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
# Define the unfolded PnP algorithm.
# ----------------------------------------------------------------------------------------
# We use the helper function :func:`deepinv.unfolded.unfolded_builder` to defined the Unfolded architecture.
# The chosen algorithm is here DRS (Douglas-Rachford Splitting).
# Note that if the prior (resp. a parameter) is initialized with a list of lenght max_iter,
# then a distinct model (resp. parameter) is trained for each iteration.
# For fixed trained model prior (resp. parameter) across iterations, initialize with a single element.

# Unrolled optimization algorithm parameters
max_iter = 5  # number of unfolded layers

# Select the data fidelity term
data_fidelity = L2()

# Set up the trainable denoising prior
# Here the prior model is common for all iterations
prior = PnP(denoiser=dinv.models.DnCNN(depth=7, pretrained=None).to(device))

# The parameters are initialized with a list of length max_iter, so that a distinct parameter is trained for each iteration.
stepsize = [1] * max_iter  # stepsize of the algorithm
sigma_denoiser = [0.01] * max_iter  # noise level parameter of the denoiser
beta = 1  # relaxation parameter of the Douglas-Rachford splitting
params_algo = {  # wrap all the restoration parameters in a 'params_algo' dictionary
    "stepsize": stepsize,
    "g_param": sigma_denoiser,
    "beta": beta,
}
trainable_params = [
    "g_param",
    "stepsize",
    "beta",
]  # define which parameters from 'params_algo' are trainable

# Logging parameters
verbose = True
wandb_vis = False  # plot curves and images in Weight&Bias

# Define the unfolded trainable model.
model = unfolded_builder(
    iteration="DRS",
    params_algo=params_algo.copy(),
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
test_batch_size = 3

# choose optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.8))

# choose supervised training loss
losses = [dinv.loss.SupLoss(metric=dinv.metric.MSE())]

train_dataloader = DataLoader(
    train_dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True
)
test_dataloader = DataLoader(
    test_dataset, batch_size=test_batch_size, num_workers=num_workers, shuffle=False
)

# %%
# Train the network
# ----------------------------------------------------------------------------------------
# We train the network using the :class:`deepinv.Trainer` class.

trainer = dinv.Trainer(
    model,
    physics=physics,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    epochs=epochs,
    scheduler=scheduler,
    losses=losses,
    optimizer=optimizer,
    device=device,
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

test_sample, _ = next(iter(test_dataloader))
model.eval()
test_sample = test_sample.to(device)

# Get the measurements and the ground truth
y = physics(test_sample)
with torch.no_grad():
    rec = model(y, physics=physics)

backprojected = physics.A_adjoint(y)

dinv.utils.plot(
    [backprojected, rec, test_sample],
    titles=["Linear", "Reconstruction", "Ground truth"],
    suptitle="Reconstruction results",
)


# %%
# Plotting the weights of the network.
# ------------------------------------
#
# We now plot the weights of the network that were learned and check that they are different from their initialization
# values. Note that ``g_param`` corresponds to :math:`\lambda` in the proximal gradient algorithm.
#

dinv.utils.plotting.plot_parameters(
    model, init_params=params_algo, save_dir=RESULTS_DIR / "unfolded_pgd" / operation
)
