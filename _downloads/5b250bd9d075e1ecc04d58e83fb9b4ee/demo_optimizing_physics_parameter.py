r"""
Solving blind inverse problems / estimating physics parameters
==============================================================

This demo shows you how to use
:class:`deepinv.physics.Physics` together with automatic differentiation to optimize your operator.

Consider the forward model

.. math::
    y = \noise{\forw{x, \theta}}

where :math:`N` is the noise model, :math:`\forw{\cdot, \theta}` is the forward operator, and the goal is to learn the parameter :math:`\theta` (e.g., the filter in :class:`deepinv.physics.Blur`).

In a typical blind inverse problem, given a measurement :math:`y`, we would like to recover both the underlying image :math:`x` and the operator parameter :math:`\theta`,
resulting in a highly ill-posed inverse problem.

In this example, we only focus on a much more simpler problem: given the measurement :math:`y` and the ground truth :math:`x`, find the parameter :math:`\theta`.
This can be reformulated as the following optimization problem:

.. math::
    \min_{\theta} \frac{1}{2} \|\forw{x, \theta} - y \|^2

This problem can be addressed by first-order optimization if we can compute the gradient of the above function with respect to :math:`\theta`.
The dependence between the operator :math:`A` and the parameter :math:`\theta` can be complicated.
DeepInverse provides a wide range of physics operators, implemented as differentiable classes.
We can leverage the automatic differentiation engine provided in Pytorch to compute the gradient of the above loss function w.r.t. the physics parameters :math:`\theta`.

The purpose of this demo is to show how to use the physics classes in DeepInverse to estimate the physics parameters, together with the automatic differentiation.
We show 3 different ways to do this: manually implementing the projected gradient descent algorithm, using a Pytorch optimizer and optimizing the physics as a usual neural network.
"""

# %%
# Import required packages
#
import deepinv as dinv
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# %%
# Define the physics
# ------------------
#
# In this first example, we use the convolution operator, defined in the :class:`deepinv.physics.Blur` class.
# We also generate a random convolution kernel of motion blur

generator = dinv.physics.generator.MotionBlurGenerator(
    psf_size=(25, 25), rng=torch.Generator(device), device=device
)
true_kernel = generator.step(1, seed=123)["filter"]
physics = dinv.physics.Blur(noise_model=dinv.physics.GaussianNoise(0.02), device=device)

# %% Load and example image and compute the measurement
x = dinv.utils.load_url_image(
    dinv.utils.demo.get_image_url("celeba_example.jpg"),
    img_size=256,
    resize_mode="resize",
).to(device)

y = physics(x, filter=true_kernel)

dinv.utils.plot([x, y, true_kernel], titles=["Sharp", "Blurry", "True kernel"])

# %%
# Define an optimization algorithm
# --------------------------------
#
# The convolution kernel lives in the simplex, ie the kernel must have positive entries summing to 1.
# We can use a simple optimization algorithm - Projected Gradient Descent - to enforce this constraint.
# The following function allows one to compute the orthogonal projection onto the simplex, by a sorting algorithm
# (Reference: `Large-scale Multiclass Support Vector Machine Training via Euclidean Projection onto the Simplex
# -- Mathieu Blondel, Akinori Fujino, and Naonori Ueda
# <https://ieeexplore.ieee.org/document/6976941>`_)
#


@torch.no_grad()
def projection_simplex_sort(v: torch.Tensor) -> torch.Tensor:
    r"""
    Projects a tensor onto the simplex using a sorting algorithm.
    """
    shape = v.shape
    B = shape[0]
    v = v.view(B, -1)
    n_features = v.size(1)
    u = torch.sort(v, descending=True, dim=-1).values
    cssv = torch.cumsum(u, dim=-1) - 1.0
    ind = torch.arange(n_features, device=v.device)[None, :].expand(B, -1) + 1.0
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    w = torch.maximum(v - theta, torch.zeros_like(v))
    return w.reshape(shape)


# We also define a data fidelity term
data_fidelity = dinv.optim.L2()
# %%
#
# Run the algorithm
#
# Initialize a constant kernel
kernel_init = torch.zeros_like(true_kernel)
kernel_init[..., 5:-5, 5:-5] = 1.0
kernel_init = projection_simplex_sort(kernel_init)
n_iter = 1000
stepsize = 0.7

kernel_hat = kernel_init
losses = []
for i in tqdm(range(n_iter)):
    # compute the gradient
    with torch.enable_grad():
        kernel_hat.requires_grad_(True)
        physics.update(filter=kernel_hat)
        loss = data_fidelity(y=y, x=x, physics=physics) / y.numel()
        loss.backward()
    grad = kernel_hat.grad

    # gradient step and projection step
    with torch.no_grad():
        kernel_hat = kernel_hat - stepsize * grad
        kernel_hat = projection_simplex_sort(kernel_hat)

    losses.append(loss.item())

dinv.utils.plot(
    [true_kernel, kernel_init, kernel_hat],
    titles=["True kernel", "Init. kernel", "Estimated kernel"],
    suptitle="Result with Projected Gradient Descent",
)

# %%
#
# We can plot the loss to make sure that it decreases
#
plt.figure()
plt.plot(range(n_iter), losses)
plt.title("Loss evolution")
plt.yscale("log")
plt.xlabel("Iteration")
plt.tight_layout()
plt.show()

# %%
# Combine with arbitrary optimizer
# --------------------------------
#
# Pytorch provides a wide range of optimizers for training neural networks.
# We can also pick one of those to optimizer our parameter

kernel_init = torch.zeros_like(true_kernel)
kernel_init[..., 5:-5, 5:-5] = 1.0
kernel_init = projection_simplex_sort(kernel_init)

kernel_hat = kernel_init.clone()
optimizer = torch.optim.Adam([kernel_hat], lr=0.1)

# We will alternate a gradient step and a projection step
losses = []
n_iter = 200
for i in tqdm(range(n_iter)):
    optimizer.zero_grad()
    # compute the gradient, this will directly change the gradient of `kernel_hat`
    with torch.enable_grad():
        kernel_hat.requires_grad_(True)
        physics.update(filter=kernel_hat)
        loss = data_fidelity(y=y, x=x, physics=physics) / y.numel()
        loss.backward()

    # a gradient step
    optimizer.step()
    # projection step, when doing additional steps, it's important to change only
    # the tensor data to avoid breaking the gradient computation
    kernel_hat.data = projection_simplex_sort(kernel_hat.data)
    # loss
    losses.append(loss.item())

dinv.utils.plot(
    [true_kernel, kernel_init, kernel_hat],
    titles=["True kernel", "Init. kernel", "Estimated kernel"],
    suptitle="Result with ADAM",
)

# %%
#
# We can plot the loss to make sure that it decreases
#
plt.figure()
plt.semilogy(range(n_iter), losses)
plt.title("Loss evolution")
plt.xlabel("Iteration")
plt.tight_layout()
plt.show()


# %%
#
# Optimizing the physics as a usual neural network
# ------------------------------------------------
#
# Below we show another way to optimize the parameter of the physics, as we usually do for neural networks

kernel_init = torch.zeros_like(true_kernel)
kernel_init[..., 5:-5, 5:-5] = 1.0
kernel_init = projection_simplex_sort(kernel_init)

# The gradient is off by default, we need to enable the gradient of the parameter
physics = dinv.physics.Blur(
    filter=kernel_init.clone().requires_grad_(True), device=device
)

# Set up the optimizer by giving the parameter to an optimizer
# Try to change your favorite optimizer
optimizer = torch.optim.AdamW([physics.filter], lr=0.1)


# Try to change another loss function
# loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.L1Loss()

# We will alternate a gradient step and a projection step
losses = []
n_iter = 100
for i in tqdm(range(n_iter)):
    # update the gradient
    optimizer.zero_grad()
    y_hat = physics.A(x)
    loss = loss_fn(y_hat, y)
    loss.backward()

    # a gradient step
    optimizer.step()

    # projection step.
    # Note: when doing additional steps, it's important to change only
    # the tensor data to avoid breaking the gradient computation
    physics.filter.data = projection_simplex_sort(physics.filter.data)

    # loss
    losses.append(loss.item())

kernel_hat = physics.filter.data
dinv.utils.plot(
    [true_kernel, kernel_init, kernel_hat],
    titles=["True kernel", "Init. kernel", "Estimated kernel"],
    suptitle="Result with AdamW and L1 Loss",
)

# %%
#
# We can plot the loss to make sure that it decreases
#
plt.figure()
plt.semilogy(range(n_iter), losses)
plt.title("Loss evolution")
plt.xlabel("Iteration")
plt.tight_layout()
plt.show()
