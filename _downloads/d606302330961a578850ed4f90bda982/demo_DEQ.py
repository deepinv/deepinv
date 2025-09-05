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
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.unfolded import DEQ_builder
from torchvision import transforms
from deepinv.utils.demo import load_dataset, load_degradation


# %%
# Setup paths for data loading and results.
# ----------------------------------------------------------------------------------------
#

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
CKPT_DIR = BASE_DIR / "ckpts"
DEG_DIR = BASE_DIR / "degradations"

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
train_base_dataset = load_dataset(train_dataset_name, transform=train_transform)
test_base_dataset = load_dataset(test_dataset_name, transform=test_transform)


# %%
# Generate a dataset of low resolution images and load it.
# ----------------------------------------------------------------------------------------
# We use the Downsampling class from the physics module to generate a dataset of low resolution images.


# Use parallel dataloader if using a GPU to speed up training, otherwise, as all computes are on CPU, use synchronous
# dataloading.
num_workers = 4 if torch.cuda.is_available() else 0

# Degradation parameters
noise_level_img = 0.03

# Generate a motion blur operator.
kernel_index = 1  # which kernel to chose among the 8 motion kernels from 'Levin09.mat'
kernel_torch = load_degradation("Levin09.npy", DEG_DIR / "kernels", index=kernel_index)
kernel_torch = kernel_torch.unsqueeze(0).unsqueeze(
    0
)  # add batch and channel dimensions

# Generate the gaussian blur downsampling operator.
physics = dinv.physics.BlurFFT(
    img_size=(n_channels, img_size, img_size),
    filter=kernel_torch,
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
# We use the helper function :func:`deepinv.unfolded.DEQ_builder` to defined the DEQ architecture.
# The chosen algorithm is here HQS (Half Quadratic Splitting).
# Note for DEQ, the prior and regularization parameters should be common for all iterations
# to keep a constant fixed-point operator.


# Select the data fidelity term
data_fidelity = L2()

# Set up the trainable denoising prior. Here the prior model is common for all iterations. We use here a pretrained denoiser.
prior = PnP(denoiser=dinv.models.DnCNN(depth=20, pretrained="download").to(device))

# Unrolled optimization algorithm parameters
max_iter = 20 if torch.cuda.is_available() else 10
stepsize = [1.0]  # stepsize of the algorithm
sigma_denoiser = [0.03]  # noise level parameter of the denoiser
jacobian_free = False  # does not perform Jacobian inversion.

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
    iteration="PGD",  # For now DEQ is only possible with PGD, HQS and GD optimization algorithms.
    params_algo=params_algo.copy(),
    trainable_params=trainable_params,
    data_fidelity=data_fidelity,
    max_iter=max_iter,
    prior=prior,
    anderson_acceleration=True,
    anderson_acceleration_backward=True,
    history_size_backward=3,
    history_size=3,
    max_iter_backward=20,
    jacobian_free=jacobian_free,
)

# %%
# Define the training parameters.
# -------------------------------
# We use the Adam optimizer and the StepLR scheduler.


# training parameters
epochs = 10 if torch.cuda.is_available() else 2
learning_rate = 1e-4
train_batch_size = 32 if torch.cuda.is_available() else 1
test_batch_size = 3


# choose optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.8))

# choose supervised training loss
losses = [dinv.loss.SupLoss(metric=dinv.metric.MSE())]

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
    show_progress_bar=True,  # disable progress bar for better vis in sphinx gallery.
    wandb_vis=wandb_vis,  # training visualization can be done in Weight&Bias
)

trainer.train()
model = trainer.load_best_model()  # load model with best validation PSNR

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
