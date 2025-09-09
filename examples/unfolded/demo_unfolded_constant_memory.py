r"""
Training unfolded neural networks without out-of-memory issues
====================================================================================================

This example shows how to train an unfolded neural network with a memory complexity that is independent of the number of iterations in least squares solver (O(1) memory complexity).

"""

# %%
import deepinv as dinv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.unfolded import unfolded_builder
from torchvision import transforms
from deepinv.utils.demo import load_dataset
import time

# %%
# Setup paths for data loading and results.
# ----------------------------------------------------------------------------------------
#

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "measurements"

device = (
    dinv.utils.get_freer_gpu() if torch.cuda.is_available() else torch.device("cpu")
)
device = torch.device("cpu")
# %%
# Load base image datasets and degradation operators.
# ----------------------------------------------------------------------------------------
# In this example, we use the CBSD500 dataset for training and the Set3C dataset for testing.

img_size = 64 if torch.cuda.is_available() else 32
n_channels = 3  # 3 for color images, 1 for gray-scale images

# %%
# Generate a dataset of low resolution images and load it.
# ----------------------------------------------------------------------------------------
# We use the Blur class from the physics module to generate a dataset of blurry images.
# For simplicity, we use a small dataset for training.
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
measurement_dir = DATA_DIR / train_dataset_name / "deblurring"
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

# %%
# Define the unfolded parameters.
# ----------------------------------------------------------------------------------------

# Unrolled optimization algorithm parameters
max_iter = 5  # number of unfolded layers

# Select the data fidelity term
data_fidelity = L2()
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


# %%
# Define the training parameters.
# ----------------------------------------------------------------------------------------

learning_rate = 5e-4
train_batch_size = 32 if torch.cuda.is_available() else 2
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


def peak_memory():
    if use_cuda:
        peak_bytes = int(torch.cuda.max_memory_allocated(device=device))
    else:  # Stats on CPU is not very accurate
        peak_bytes = 0
    return peak_bytes


# %%
# We first train the model will full backpropagation to compare the memory usage.
# Define the unfolded trainable model.
torch.manual_seed(42)  # Make sure that we have the same initialization for both runs
prior = PnP(denoiser=dinv.models.DnCNN(depth=7, pretrained=None).to(device))
model = unfolded_builder(
    iteration="HQS",
    params_algo=params_algo.copy(),
    trainable_params=trainable_params,
    data_fidelity=data_fidelity,
    max_iter=max_iter,
    prior=prior,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
model.train()

# Setting this parameter to False to use full backpropagation
physics.implicit_backward_solver = False

reset_memory()
sync()
start = time.perf_counter()
avg_loss = 0.0
for batch in train_dataloader:
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad()
    x_hat = model(physics=physics, y=y)
    loss = torch.nn.functional.mse_loss(x_hat, x)
    avg_loss += loss.item()
    loss.backward()
    optimizer.step()
sync()
end = time.perf_counter()
auto_peak_memory_mb = peak_memory() / (10**6)
auto_time_per_iter = (end - start) / len(train_dataloader)
auto_avg_loss = avg_loss / len(train_dataloader)
# %%
# Setting this parameter to True to use implicit differentiation
physics.implicit_backward_solver = True

# Define the unfolded trainable model.
torch.manual_seed(42)  # Make sure that we have the same initialization for both runs
prior = PnP(denoiser=dinv.models.DnCNN(depth=7, pretrained=None).to(device))
model = unfolded_builder(
    iteration="HQS",
    params_algo=params_algo.copy(),
    trainable_params=trainable_params,
    data_fidelity=data_fidelity,
    max_iter=max_iter,
    prior=prior,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
model.train()

reset_memory()
sync()
start = time.perf_counter()
avg_loss = 0.0
for batch in train_dataloader:
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad()
    x_hat = model(physics=physics, y=y)
    loss = torch.nn.functional.mse_loss(x_hat, x)
    avg_loss += loss.item()
    loss.backward()
    optimizer.step()
sync()
end = time.perf_counter()
implicit_peak_memory_mb = peak_memory() / (10**6)
implicit_time_per_iter = (end - start) / len(train_dataloader)
implicit_avg_loss = avg_loss / len(train_dataloader)
# %%
# Compare the memory usage
# ----------------------------------------------------------------------------------------
print(
    f"Full backpropagation: time per iteration: {auto_time_per_iter:.2f} ms, avg loss: {auto_avg_loss:.6f}"
)
print(
    f"Implicit differentiation: time per iteration: {implicit_time_per_iter:.2f} ms, avg loss: {implicit_avg_loss:.6f}"
)

if use_cuda:
    print(f"Full backpropagation: peak memory usage: {auto_peak_memory_mb:.1f} MB")
    print(
        f"Implicit differentiation: peak memory usage: {implicit_peak_memory_mb:.1f} MB"
    )
    print(
        f"Memory reduction factor: {auto_peak_memory_mb/implicit_peak_memory_mb:.1f}x"
    )
