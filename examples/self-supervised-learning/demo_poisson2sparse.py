r"""
Poissong denoising using Poisson2Sparse
===================================================

This code shows how to restore an image corrupted by Poisson noise using Poisson2Sparse

This method is based on the paper "Poisson2Sparse" :footcite:t:`ta2022poisson2sparse` and restores an image by learning a sparse non-linear dictionary parametrized by a neural network using a combination of Neighbor2Neighbor :footcite:t:`huang2021neighbor2neighbor`, of the negative log Poisson likelihood, of the :math:`\ell^1` pixel distance and of a sparsity-inducing :math:`\ell^1` regularization function on the weights.

"""

import deepinv as dinv
from deepinv.utils.plotting import plot
import torch
import torchvision.transforms as transforms
import deepinv as dinv
import torch
import torchvision.transforms as transforms
import deepinv as dinv
import numpy as np
import random


# %%
# Load a Poisson corrupted image
# ----------------------------
#
# This example uses an image from the microscopy dataset FMD :footcite:t:`zhang2018poisson`.

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

physics = dinv.physics.Denoising(
    dinv.physics.PoissonNoise(gain=0.01, normalize=True)
)

x = dinv.utils.demo.load_example("FMD_TwoPhoton_MICE_R_gt_12_avg50.png", img_size=(256, 256)).to(device)
x = x[:, 0:1, :64, :64]
y = physics(x)

# Seed the RNGs for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)
np.random.seed(0)

# %%
# Define the Poisson2Sparse model

backbone = dinv.models.ConvLista(
    channels=1,
    kernel_size=3,
    norm=False,
    num_filters=512,
    num_iter=10,
    stride=1,
    threshold=0.01,
)

model = dinv.models.Poisson2Sparse(
    backbone=backbone,
    lr=1e-4,
    num_iter=200,
    weight_n2n=2.0,
    weight_l1_regularization=1e-5,
    verbose=True,
).to(device)

# %%
# Run the model
# ----------------------------------

x_hat = model(y)

# Compute PSNR
print(f"Measurement PSNR: {dinv.metric.PSNR()(x, y).item():.2f} dB")
print(f"Poisson2Sparse PSNR: {dinv.metric.PSNR()(x, x_hat).item():.2f} dB")

# Plot results
plot([y, x_hat, x], titles=["Measurement", "Poisson2Sparse", "Ground truth"])
