r"""
Reconstructing an image using the deep image prior.
===================================================

This code shows how to reconstruct a noisy and incomplete image using the deep image prior.

This method is based on the paper "Deep Image Prior" :footcite:t:`ulyanov2018deep` and reconstructs
an image by minimizing the loss function

.. math::

    \min_{\theta}  \|y-Af_{\theta}(z)\|^2

where :math:`z` is a random input and :math:`f_{\theta}` is a convolutional decoder network with parameters
:math:`\theta`. The minimization should be stopped early to avoid overfitting. The method uses the Adam
optimizer.


"""

# %%
import deepinv as dinv
from deepinv.utils.plotting import plot
import torch
from deepinv.utils import load_example

# %%
# Load image from the internet
# ----------------------------
#
# This example uses an image of Messi.

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

x = load_example("messi.jpg", img_size=32).to(device)

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

# %%
# Define forward operator and noise model
# ---------------------------------------
#
# We use image inpainting as the forward operator and Gaussian noise as the noise model.

sigma = 0.1  # noise level
physics = dinv.physics.Inpainting(mask=0.5, img_size=x.shape[1:], device=device)
physics.noise_model = dinv.physics.GaussianNoise(sigma=sigma)

# %%
# Generate the measurement
# ------------------------
# We apply the forward model to generate the noisy measurement.

y = physics(x)

# %%
# Define the deep image prior
# ----------------------------
#
# This method only works with certain convolutional decoder networks. We recommend using the
# network :class:`deepinv.models.ConvDecoder`.
#
# .. note::
#
#     The number of iterations and learning rate have been set manually to obtain good results. However, these
#     values may not be optimal for all problems. We recommend experimenting with different values.
#
# .. note::
#
#     Here we run a small number of iterations to reduce the runtime of the example. However, the results could
#     be improved by running more iterations.

iterations = 100
lr = 1e-2  # learning rate for the optimizer.
channels = 64  # number of channels per layer in the decoder.
in_size = [2, 2]  # size of the input to the decoder.
backbone = dinv.models.ConvDecoder(
    img_size=x.shape[1:], in_size=in_size, channels=channels
).to(device)

f = dinv.models.DeepImagePrior(
    backbone,
    learning_rate=lr,
    iterations=iterations,
    verbose=True,
    input_size=[channels] + in_size,
).to(device)

# %%
# Run DIP algorithm and plot results
# ----------------------------------
# We run the DIP algorithm and plot the results.
#
# The good performance of DIP is somewhat surprising, since the network has many parameters and could potentially
# overfit the noisy measurement data. However, the architecture acts as an implicit regularizer, providing good
# reconstructions if the optimization is stopped early.
# While this phenomenon is not yet well understood, there has been some efforts to explain it. For example, see :footcite:t:`tachella2021neural`.
dip = f(y, physics)

# compute linear inverse
x_lin = physics.A_adjoint(y)

# compute PSNR
print(f"Linear reconstruction PSNR: {dinv.metric.PSNR()(x, x_lin).item():.2f} dB")
print(f"DIP PSNR: {dinv.metric.PSNR()(x, dip).item():.2f} dB")

# plot results
plot([x_lin, x, dip], titles=["measurement", "ground truth", "DIP"])

# %%
# :References:
#
# .. footbibliography::
