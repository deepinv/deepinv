r"""
Solving inverse problems with sinusoidal representation networks (SIRENs)
============================================================================

This notebook presents several examples of inverse problems solved using the framework of Implicit Neural Representation (INR). It is based on the paper "SIREN" of :footcite:t:`sitzmann2020implicit`.

The method reconstructs an image by minimizing the loss function

.. math::
    \min_{\theta}  \|y-Af_{\theta}(z)\|^2_2 + \lambda \mathcal R(f_\theta),

where

- :math:`f_{\theta} : \mathbb{R}^d \to \mathbb{R}` is a SIREN network with parameter :math:`\theta`;
- :math:`z` is an input grid of coordinates of shape :math:`n\times d`, where :math:`d` is the number of dimensions of the input (e.g. :math:`d=2` for an image) and :math:`n` is the total number of pixels. This means that :math:`f_\theta` is applied pixelwise;
- :math:`\lambda > 0` is a regularization parameter;
- :math:`\mathcal{R}(f_\theta)` is a regularizer applied on the SIREN network.

In this notebook, we will use the TV regularizer defined as :math:`\mathcal{R}(f_\theta) = \|\nabla f_{\theta}(z)\|_{1}.`

Note that :math:`\nabla` is here the continuous nabla operator implemented with autograd.

A SIREN is a neural network composed of 2 parts: the positional encoding followed by an MLP with sine activation functions. Formally, it reads as

.. math::
    f_\theta(z) = \text{MLP} \circ \text{PosEnc}(z) :=  \phi^{(L)} (\cdots \phi^{(0)}(\sin(\omega_0 W z + b)) \cdots), \quad \phi^{(i)}(z) = \sin(\omega_0' W^{(i)} z + b^{(i)}).

For more information about INRs, we refer to :footcite:t:`sitzmann2020implicit, yuce2022structured`.

.. tip::
    The INR framework looks like Deep Image Prior. The difference is that :math:`z` is no longer a random latent vector but instead any grid of coordinates of :math:`\mathbb R^d`. After training the network :math:`f_\theta`, one can evaluate it at any spatial input coordinate in :math:`\mathbb R^d`, which allows us to naturally interpolate the training grid.
"""

# %%
import deepinv as dinv
from deepinv.utils.plotting import plot
import torch
from deepinv.models.siren import get_mgrid
from pathlib import Path
from torchvision import transforms
from deepinv.utils.demo import load_dataset, load_degradation

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

# %%
# Load dataset
# ------------
# For simplicity, we perform all the experiments on the same dataset 'set3c'.

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
DEG_DIR = BASE_DIR / "degradations"

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# Set up the variable to fetch dataset and operators.
# ---------------------------------------------------
dataset_name = "set3c"
img_size = 256 if torch.cuda.is_available() else 64
val_transform = transforms.Compose(
    [transforms.CenterCrop(img_size), transforms.ToTensor()]
)

# Generate a motion blur operator.
kernel_index = 1  # which kernel to chose among the 8 motion kernels from 'Levin09.mat'
kernel_torch = load_degradation("Levin09.npy", DEG_DIR / "kernels", index=kernel_index)
kernel_torch = kernel_torch.unsqueeze(0).unsqueeze(
    0
)  # add batch and channel dimensions
dataset = load_dataset(dataset_name, transform=val_transform)

# %%
# 1. Image processing tasks
# -------------------------
# In the first part of the notebook, we propose three examples of image processing tasks: debluring, inpainting, and super-resolution.
#
# 1.1 Image deblurring
# --------------------
noise_level_img = 0.01  # Gaussian Noise standard deviation for the degradation
n_channels = 1  # 3 for color images, 1 for gray-scale images
physics = dinv.physics.BlurFFT(
    img_size=(n_channels, img_size, img_size),
    filter=kernel_torch,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
)

# Select the first image from the dataset
x = dataset[0][:1].unsqueeze(0).to(device)

# Apply the degradation to the image
y = physics(x)

# %%
iterations = 1000
lr = 1e-2
# Define the SIREN architecture and the hyperparameters
siren_net = dinv.models.SIREN(
    input_dim=2,
    encoding_dim=64,
    siren_dims=[64] * 5,
    out_channels=1,
    omega0={"encoding": 30.0, "siren": 2.0},
    device=device,
).to(device)
# Define the Deep Image Prior model with the SIREN network
f = dinv.models.ImplicitNeuralRepresentation(
    siren_net,
    learning_rate=lr,
    iterations=iterations,
    verbose=True,
    img_size=None,
    regul_param=1e-4,
).to(device)

# %%
dip = f(y, physics, z=get_mgrid(x.shape[2:]), shape=x.shape)

# %%
# Compute PSNR
print(f"Init PSNR: {dinv.metric.PSNR()(x, y).item():.2f} dB")
print(f"SIREN PSNR: {dinv.metric.PSNR()(x, dip).item():.2f} dB")

# plot results
plot([y, x, dip], titles=["measurements", "ground truth", "reconstruction"])

# %%
# 1.2 Image inpainting
# --------------------

# %%
sigma = 0.05  # noise level
physics = dinv.physics.Inpainting(mask=0.5, img_size=x.shape[1:], device=device)
physics.noise_model = dinv.physics.GaussianNoise(sigma=sigma)
y = physics(x)

# %%
iterations = 500
lr = 5e-3
siren_net = dinv.models.SIREN(
    input_dim=2,
    encoding_dim=64,
    siren_dims=[64] * 5,
    out_channels=1,
    omega0={"encoding": 30.0, "siren": 1.5},
    device=device,
).to(device)

f = dinv.models.ImplicitNeuralRepresentation(
    siren_net,
    learning_rate=lr,
    iterations=iterations,
    verbose=True,
    img_size=None,
    regul_param=5e-4,
).to(device)

# %%
dip = f(y, physics, z=get_mgrid(x.shape[2:]), shape=x.shape)

# %%
# Compute PSNR
print(f"Init PSNR: {dinv.metric.PSNR()(x, y).item():.2f} dB")
print(f"SIREN PSNR: {dinv.metric.PSNR()(x, dip).item():.2f} dB")

# plot results
plot([y, x, dip.clip(0, 1)], titles=["measurement", "ground truth", "reconstruction"])

# %%
# 1.3 Super-resolution
# --------------------
noise_level_img = 0.03  # Gaussian Noise standard deviation for the degradation
n_channels = 1  # 3 for color images, 1 for gray-scale images

factor = 2
# Define the downsampling operator to generate the low-resolution image
physics = dinv.physics.Downsampling(
    img_size=x.shape[1:],
    factor=factor,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
)

# Apply the degradation to the image
y = physics(x)

# %%
iterations = 500
lr = 1e-2
siren_net = dinv.models.SIREN(
    input_dim=2,
    encoding_dim=32,
    siren_dims=[32] * 5,
    out_channels=1,
    omega0={"encoding": 30.0, "siren": 1.5},
    device=device,
).to(device)

f = dinv.models.ImplicitNeuralRepresentation(
    siren_net,
    learning_rate=lr,
    iterations=iterations,
    verbose=True,
    img_size=None,
    regul_param=1e-3,
).to(device)


# %%
# Define an identity physics with zero noise to perform the reconstruction at the low resolution
physics_f = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma=0.0))
dip = f(y, physics_f, z=get_mgrid(y.shape[2:]), shape=y.shape)

# %%
# Super-resolution by evaluating the trained network at a finer grid
dip_super_resolved = f.siren_net(get_mgrid(x.shape[2:])).view(x.shape)  # compute PSNR
print(f"DIP PSNR: {dinv.metric.PSNR()(y, dip).item():.2f} dB")
print(
    f"super-resolved DIP PSNR: {dinv.metric.PSNR()(x, dip_super_resolved.clip(0,1)).item():.2f} dB"
)

# plot results
plot(
    [y, x, dip, dip_super_resolved.clip(0, 1)],
    titles=["measurement", "ground truth", "SIREN", "SIREN super-res"],
)

# %%
# 2 Gradient supervised reconstruction
# ------------------------------------
# In the second part, we exploit the fact that INRs are intrisicly smooth continuous models; their gradient can be computed exactly with autograd. We show an example of an image recontruction problem where the forward operator :math:`A` is a gradient operator.

# %%
#
# --------------------------------------
# We aim at solving the problem:
#
# .. math::
#   \text{Find } \Phi : \mathbb R^d \to \mathbb R \text{ such that } y = \nabla \Phi(z).
#
# We model the unkown function :math:`\Phi` by a SIREN :math:`f_\theta` and solve
#
# .. math::
#   \min_{\theta} \frac{1}{2} \|y - \nabla f_\theta(z) \|_2^2,
#
# using the ADAM optimizer to train :math:`\theta`. Finally, we evaluate :math:`f_\theta(z)` and compare it with the ground truth :math:`\Phi(z)`.

# %%
z = get_mgrid(x.shape[2:])
z.requires_grad_(True)
y = dinv.models.TVDenoiser.nabla(x).view(z.shape, 2)


class Gradient(dinv.physics.Physics):
    r"""
    Compute the continuous gradient of a deep neural network model with autograd.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return dinv.models.siren.nabla(x, z)


physics_f = Gradient()

# %%
iterations = 1000
lr = 1e-2
siren_net = dinv.models.SIREN(
    input_dim=2,
    encoding_dim=64,
    siren_dims=[64] * 5,
    out_channels=1,
    omega0={"encoding": 30.0, "siren": 2.0},
    device=device,
).to(device)

f = dinv.models.ImplicitNeuralRepresentation(
    siren_net,
    learning_rate=lr,
    iterations=iterations,
    verbose=True,
    img_size=None,
    regul_param=None,
).to(device)

# %%
dip = f(y, physics_f, z=z, shape=x.shape)

# %%
print(
    f"DIP PSNR: {dinv.metric.PSNR()(x, (dip-dip.min()) / (dip.max()-dip.min())).item():.2f} dB"
)

# plot results
plot(
    [
        y.view(*x.shape, 2)[..., 0] + y.view(*x.shape, 2)[..., 1],
        x,
        (dip - dip.min()) / (dip.max() - dip.min()),
    ],
    titles=["divergence", "ground truth", "reconstruction"],
)

# %%
# :References:
#
# .. footbibliography::
