r"""
Reducing the memory and computational complexity of unfolded network training
====================================================================================================

Some unfolded architectures rely on a `least-squares solver <deepinv.optim.utils.least_squares>` to compute the proximal step w.r.t. the data-fidelity term (e.g., :class:`ADMM <deepinv.optim.optim_iterators.ADMMIteration>` or :class:`HQS <deepinv.optim.optim_iterators.HQSIteration>`):  

.. math::  

     \operatorname{prox}_{\gamma f}(z)  = \underset{x}{\arg\min} \; \frac{\gamma}{2}\|A_\theta x-y\|^2 + \frac{1}{2}\|x-z\|^2  

During backpropagation, a naive implementation requires storing the gradients of every intermediate step of the least squares solver (which is an iterative method), resulting in significant memory and computational costs which are proportional to number of iterations done by the solver.  

The library provides a memory-efficient back-propagation strategy that reduces the memory footprint during training, by computing the gradients of the proximal step in closed-form, without storing any intermediate steps. This closed-form computation requires evaluating the least-squares solver one additional time during the gradient computation.  

Let :math:`h(z, y, \theta, \gamma) = \operatorname{prox}_{\gamma f}(z)` be the proximal operator. During the backward pass, we need to compute the vector-Jacobian products (VJPs), in the input variables :math:`(z, y, \theta, \gamma)` is required for backpropagation:

.. math::

    \left( \frac{\partial h}{\partial z} \right)^T v, \quad \left( \frac{\partial h}{\partial y} \right)^T v, \quad \left( \frac{\partial h}{\partial \theta} \right)^T v, \quad \left( \frac{\partial h}{\partial \gamma} \right)^T v

and :math:`v` is the upstream gradient. The VJPs can be computed in closed-form by solving a second least-squares problem, as shown in the following. 
When the forward least-squares solver converges to the exact minimizer, we have the following closed-form expressions for :math:`h(z, y, \theta, \gamma)`:

.. math::

    h(z, y, \theta, \gamma) = \left( A_\theta^T A_\theta + \frac{1}{\gamma} I \right)^{-1} \left( A_\theta^T y + \frac{1}{\gamma} z \right)

Let :math:`M` denote the inverse :math:`\left( A_\theta^T A_\theta + \frac{1}{\gamma} I \right)^{-1}`. The VJPs can be computed as follows:

.. math::

    \left( \frac{\partial h}{\partial z} \right)^T v               &= \frac{1}{\gamma} M v \\
    \left( \frac{\partial h}{\partial y} \right)^T v               &= A_\theta M v \\
    \left( \frac{\partial h}{\partial \gamma} \right)^T v          &= \langle M v, h - z \rangle / \gamma^2 \\
    \left( \frac{\partial h}{\partial \theta} \right)^T v          &= \frac{\partial p}{\partial \theta} 
    
where :math:`p = \langle M v, A_\theta^T (y - A_\theta h) \rangle` and :math:`\frac{\partial p}{\partial \theta}` can be computed using the standard backpropagation mechanism (autograd).

.. note::

    Linear forward operators that have a :class:`closed-form singular value decomposition <deepinv.physics.DecomposablePhysics>` benefit from a :func:`closed-form formula <deepinv.physics.DecomposablePhysics.prox_l2>` for computing the proximal step, and thus we shouldn't expect speed-ups in these specific cases.  


This example shows how to train an unfolded neural network with a memory complexity that is independent of the number of iterations in least squares solver (O(1) memory complexity) used for computing the data-fidelity proximal step.

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
import numpy as np
import matplotlib.pyplot as plt

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

# Specify the transforms to be applied to the input images.
train_transform = transforms.Compose(
    [transforms.RandomCrop(img_size), transforms.ToTensor()]
)
# Define the base train and test datasets of clean images.
train_dataset = load_dataset(train_dataset_name, transform=train_transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=32 if torch.cuda.is_available() else 2,
    num_workers=4 if torch.cuda.is_available() else 0,
    shuffle=True,
)
physics = dinv.physics.Blur(
    filter=dinv.physics.blur.gaussian_blur(sigma=(1.5, 1.5)),
    padding="valid",
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=0.1),
    max_iter=50,
    tol=1e-8,
    implicit_backward_solver=False,
)


# %%
# Define the unfolded parameters.
# ----------------------------------------------------------------------------------------

# Unrolled optimization algorithm parameters
max_iter = 5  # number of unfolded layers

# Select the data fidelity term
data_fidelity = L2()
stepsize = [1] * max_iter  # stepsize of the algorithm
sigma_denoiser = [0.01] * max_iter  # noise level parameter of the denoiser
params_algo = {  # wrap all the restoration parameters in a 'params_algo' dictionary
    "stepsize": stepsize,
    "g_param": sigma_denoiser,
}
trainable_params = [
    "g_param",
    "stepsize",
]  # define which parameters from 'params_algo' are trainable


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

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-8)
model.train()

# Setting this parameter to False to use full backpropagation
physics.implicit_backward_solver = False

reset_memory()
sync()
start = time.perf_counter()
auto_losses = []
for x in train_loader:
    x = x.to(device)
    y = physics(x)
    optimizer.zero_grad()
    x_hat = model(physics=physics, y=y)
    loss = torch.nn.functional.mse_loss(x_hat, x)
    auto_losses.append(loss.item())
    loss.backward()
    optimizer.step()
sync()
end = time.perf_counter()
auto_peak_memory_mb = peak_memory() / (10**6)
auto_time_per_iter = (end - start) / len(train_loader)
auto_avg_loss = np.array(auto_losses)
auto_avg_loss = np.cumsum(auto_avg_loss) / (np.arange(len(auto_avg_loss)) + 1)


# %%
# We now train the model using the closed-form gradients of the proximal step.
# We can do this by setting `implicit_backward_solver` to `True`.
#

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

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-8)
model.train()

reset_memory()
sync()
start = time.perf_counter()
implicit_losses = []
for x in train_loader:
    x = x.to(device)
    y = physics(x)
    optimizer.zero_grad()
    x_hat = model(physics=physics, y=y)
    loss = torch.nn.functional.mse_loss(x_hat, x)
    implicit_losses.append(loss.item())
    loss.backward()
    optimizer.step()
sync()
end = time.perf_counter()
implicit_peak_memory_mb = peak_memory() / (10**6)
implicit_time_per_iter = (end - start) / len(train_loader)
implicit_avg_loss = np.array(implicit_losses)
implicit_avg_loss = np.cumsum(implicit_avg_loss) / (
    np.arange(len(implicit_avg_loss)) + 1
)

# %%
# Compare the memory usage
# ----------------------------------------------------------------------------------------
print(f"Full backpropagation: time per iteration: {auto_time_per_iter:.2f} ms. ")
print(f"Implicit differentiation: time per iteration: {implicit_time_per_iter:.2f} ms.")

# Compare the memory usage
if use_cuda:
    print(f"Full backpropagation: peak memory usage: {auto_peak_memory_mb:.1f} MB")
    print(
        f"Implicit differentiation: peak memory usage: {implicit_peak_memory_mb:.1f} MB"
    )
    print(
        f"Memory reduction factor: {auto_peak_memory_mb/implicit_peak_memory_mb:.1f}x"
    )


# Compare the training loss
plt.figure(figsize=(8, 4))
plt.plot(auto_avg_loss, label="Full backpropagation")
plt.plot(implicit_avg_loss, label="Implicit differentiation")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Training loss (MSE)")
plt.legend()
plt.title(
    f"Training loss. Avg loss difference: {np.mean(np.abs(auto_avg_loss - implicit_avg_loss)):.2e}"
)
plt.grid()
plt.show()
# %%
