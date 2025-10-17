"""
Using Spyrit with DeepInverse
====================================================================================================

This example shows how to use Spyrit linear models and measurements with DeepInverse
Here we use the HadamSplit2d linear model from Spyrit.
"""

# %%
# Loads images
# -----------------------------------------------------------------------------

###############################################################################
# We load a batch of images from the :attr:`/images/` folder with values in (0,1).
import os
import torchvision
import torch.nn

from spyrit.misc.disp import imagesc
from spyrit.misc.statistics import transform_gray_norm

import deepinv as dinv

spyritPath = os.getcwd()
imgs_path = os.path.join(spyritPath, "images/")

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# Grayscale images of size (32, 32), no normalization to keep values in (0,1)
transform = transform_gray_norm(img_size=32, normalize=False)

# Create dataset and loader (expects class folder 'images/test/')
dataset = torchvision.datasets.ImageFolder(root=imgs_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=7)

x, _ = next(iter(dataloader))
print(f"Ground-truth images: {x.shape}")

###############################################################################
# We select the second image in the batch and plot it.

i_plot = 1
imagesc(x[i_plot, 0, :, :], r"$32\times 32$ image $X$")

# %%
# Basic example
# -----------------------------------------------------------------------------

######################################################################
# We instantiate an HadamSplit2d object and simulate the 2D hadamard transform of the input images. Reshape output is necesary for deepinv. We also add Poisson noise.
from spyrit.core.meas import HadamSplit2d
import spyrit.core.noise as noise
from spyrit.core.prep import UnsplitRescale

meas_spyrit = HadamSplit2d(32, 512, device=device, reshape_output=True)
alpha = 50  # image intensity
meas_spyrit.noise_model = noise.Poisson(alpha)
y = meas_spyrit(x)

# preprocess
prep = UnsplitRescale(alpha)
m_spyrit = prep(y)

print(y.shape)


######################################################################
# The norm has to be computed to be passed to deepinv. We need to use the max singular value of the linear operator.
norm = torch.linalg.norm(meas_spyrit.H, ord=2)
print(norm)


# %%
# Forward operator
# ----------------------------------------------------------------------

###############################################################################
# You can direcly give the forward operator to deepinv. You can also add noise using deepinv model or spyrit model.
meas_deepinv = dinv.physics.LinearPhysics(
    lambda y: meas_spyrit.measure_H(y) / norm,
    A_adjoint=lambda y: meas_spyrit.unvectorize(meas_spyrit.adjoint_H(y) / norm),
)
# meas_deepinv.noise_model = dinv.physics.GaussianNoise(sigma=0.01)
m_deepinv = meas_deepinv(x)
print("diff:", torch.linalg.norm(m_spyrit / norm - m_deepinv))


# %%
# Reconstruction with deepinverse
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

######################################################################
# First, use the adjoint and dagger (pseudo-inverse) operators to reconstruct the image.
x_adj = meas_deepinv.A_adjoint(m_spyrit / norm)
imagesc(x_adj[1, 0, :, :].cpu(), "Adjoint")

x_pinv = meas_deepinv.A_dagger(m_spyrit / norm)
imagesc(x_pinv[1, 0, :, :].cpu(), "Pinv")


######################################################################
# You can also use optimization-based methods from deepinv. Here, we use Total Variation (TV) regularization with a projected gradient descent (PGD) algorithm. You can note the use of the custom_init parameter to initialize the algorithm with the dagger operator.
model_tv = dinv.optim.optim_builder(
    iteration="PGD",
    prior=dinv.optim.TVPrior(),
    data_fidelity=dinv.optim.L2(),
    params_algo={"stepsize": 1, "lambda": 5e-2},
    max_iter=10,
    custom_init=lambda y, Physics: {"est": (Physics.A_dagger(y),)},
)

x_tv, metrics_TV = model_tv(m_spyrit / norm, meas_deepinv, compute_metrics=True, x_gt=x)
dinv.utils.plot_curves(metrics_TV)
imagesc(x_tv[1, 0, :, :].cpu(), "TV recon")

######################################################################
# Deep Plug and Play (DPIR) algorithm can also be used with a pretrained denoiser. Here, we use the DRUNet denoiser.
denoiser = dinv.models.DRUNet(in_channels=1, out_channels=1, device=device)
model_dpir = dinv.optim.DPIR(sigma=1e-1, device=device, denoiser=denoiser)
model_dpir.custom_init = lambda y, Physics: {"est": (Physics.A_dagger(y),)}
with torch.no_grad():
    x_dpir = model_dpir(m_spyrit / norm, meas_deepinv)
imagesc(x_dpir[1, 0, :, :].cpu(), "DIPR recon")

######################################################################
# Reconstruct Anything Model (RAM) can also be used.
model_ram = dinv.models.RAM(pretrained=True, device=device)
model_ram.sigma_threshold = 1e-1
with torch.no_grad():
    x_ram = model_ram(m_spyrit / norm, meas_deepinv)
imagesc(x_ram[1, 0, :, :].cpu(), "RAM recon")
