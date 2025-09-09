r"""
Training unfolded neural networks without out-of-memory issues
====================================================================================================

This example shows how to train an unfolded neural network with a memory complexity that is independent of the number of iterations in least squares solver (O(1) memory complexity).

"""

# %%
import tracemalloc
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

device = (
    dinv.utils.get_freer_gpu() if torch.cuda.is_available() else torch.device("cpu")
)

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

physics = dinv.physics.Blur(
    filter=dinv.physics.blur.gaussian_blur(sigma=(1.5, 1.5)),
    padding="valid",
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=0.1),
    max_iter=50,
    tol=1e-8,
    implicit_backward_solver=False,
)
my_dataset_name = "demo_unfolded_memory"
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

# Define the unfolded trainable model.
model = unfolded_builder(
    iteration="HQS",
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
learning_rate = 5e-4
train_batch_size = 32 if torch.cuda.is_available() else 1

# choose optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
train_dataloader = DataLoader(
    train_dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True
)

# %%
# Train the network
# ----------------------------------------------------------------------------------------
# Here we write explicitly the training loop to show how implicit differentiation can be used to avoid out-of-memory issues and sometimes accelerate training. But you can also use the :class:`deepinv.Trainer` class as shown in other examples.


# Some helper functions for measuring memory usage
use_cuda = device.type == "cuda" and torch.cuda.is_available()


def sync():
    if use_cuda:
        torch.cuda.synchronize()


def reset_memory():
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        sync()
    else:
        tracemalloc.stop()
        tracemalloc.start()


def peak_memory():
    if use_cuda:
        peak_bytes = int(torch.cuda.max_memory_allocated(device=device))
    else:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_bytes = int(peak)
    return peak_bytes


# We first train the model will full backpropagation to compare the memory usage.
model.to(device)
model.train()

# Setting this parameter to False to use full backpropagation
physics.implicit_backward_solver = False

sync()
reset_memory()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for batch in train_dataloader:
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad()
    x_hat = model(physics=physics, y=y)
    loss = torch.nn.functional.mse_loss(x_hat, x)
    loss.backward()
    optimizer.step()
sync()
end.record()
auto_peak_memory_bytes = peak_memory()
auto_time_per_iter = start.elapsed_time(end) / len(train_dataloader)

# Setting this parameter to True to use implicit differentiation
physics.implicit_backward_solver = True

sync()
reset_memory()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for batch in train_dataloader:
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad()
    x_hat = model(physics=physics, y=y)
    loss = torch.nn.functional.mse_loss(x_hat, x)
    loss.backward()
    optimizer.step()
sync()
end.record()
implicit_peak_memory_bytes = peak_memory()
implicit_time_per_iter = start.elapsed_time(end) / len(train_dataloader)

# %% Compare the memory usage
# ----------------------------------------------------------------------------------------
print(
    f"Full backpropagation: time per iteration: {auto_time_per_iter:.2f} ms, peak memory: {auto_peak_memory_bytes / 1e6:.2f} MB"
)
print(
    f"Implicit differentiation: time per iteration: {implicit_time_per_iter:.2f} ms, peak memory: {implicit_peak_memory_bytes / 1e6:.2f} MB"
)
