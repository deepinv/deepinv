r"""
Poisson denoising using Poisson2Sparse
=======================================

This code shows how to restore a single image corrupted by Poisson noise using Poisson2Sparse, without requiring external training or knowledge of the noise level.

This method is based on the paper "Poisson2Sparse" :footcite:t:`ta2022poisson2sparse` and restores an image by learning a sparse non-linear dictionary parametrized by a neural network using a combination of Neighbor2Neighbor :footcite:t:`huang2021neighbor2neighbor`, of the negative log Poisson likelihood, of the :math:`\ell^1` pixel distance and of a sparsity-inducing :math:`\ell^1` regularization function on the weights.

"""

import deepinv as dinv
import torch


# %%
# Load a Poisson corrupted image
# ------------------------------
#
# This example uses an image from the microscopy dataset FMD :footcite:t:`zhang2018poisson`.

# Seed the RNGs for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

physics = dinv.physics.Denoising(dinv.physics.PoissonNoise(gain=0.01, normalize=True))

x = dinv.utils.demo.load_example(
    "FMD_TwoPhoton_MICE_R_gt_12_avg50.png", img_size=(256, 256)
).to(device)
x = x[:, 0:1, :64, :64]
x = x.clamp(0, 1)
y = physics(x)

# %%
# Define the Poisson2Sparse model

backbone = dinv.models.ConvLista(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    num_filters=512,
    num_iter=10,
    stride=2,
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
# -------------
#
# Note that we do not pass in the physics model as Poisson2Sparse assumes a
# Poisson noise model internally and does not depend on the noise level.

x_hat = model(y)

# Compute and display PSNR values
learning_free_psnr = dinv.metric.PSNR()(y, x).item()
model_psnr = dinv.metric.PSNR()(x_hat, x).item()
print(f"Measurement PSNR: {learning_free_psnr:.1f} dB")
print(f"Poisson2Sparse PSNR: {model_psnr:.1f} dB")

# Plot results
dinv.utils.plot(
    [y, x_hat, x],
    titles=["Measurement", "Poisson2Sparse", "Ground truth"],
    subtitles=[f"{learning_free_psnr:.1f} dB", f"{model_psnr:.1f} dB", ""],
)

# %%
# :References:
#
# .. footbibliography::
