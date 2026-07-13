r"""
Uncertainty quantification with VBLE-z.
====================================================================================================

This code shows you how to use VBLE-z sampling algorithm to quantify uncertainty of a reconstruction
from incomplete and noisy measurements.

VBLE-z first approximate the posterior distribution with by a parametric distribution :math:`q(z)` by maximizing the
          Evidence Lower Bound (ELBO):

.. math::

    \mathbb{E}_{q(z)}[\log p(y|z)] - \lambda D_{KL}(q(z) || p_\theta(z))

where

- :math:`p(y|z)` is the likelihood of the measurements given the latent variable :math:`z`,
  defined as :math:`p(y|z) = p(y|x=D_\theta(z))` with :math:`D_\theta` the decoder of a pretrained generative model,

- :math:`p_\theta(z)` is the prior distribution on the latent variable :math:`z`.

The VBLE-z method is described in the paper
`"Variational Bayes Image Restoration with Compressive Autoencoders" <https://arxiv.org/abs/2311.17744>`_.
"""

# %%
import deepinv as dinv
from deepinv.utils.plotting import plot, plot_curves
import torch
from deepinv.utils import load_example

# %%
# Load image from the internet
# --------------------------------------------
#
# This example uses an image of Messi.

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

x = load_example(
    "butterfly.png",
    img_size=256,
    grayscale=False,
    resize_mode="resize",
).to(device)


# %%
# Define forward operator and noise model
# --------------------------------------------------------------
#
# This example uses deblur as the forward operator and Gaussian noise as the noise model.

sigma = 0.05  # noise level
physics = dinv.physics.Blur(
    dinv.physics.blur.gaussian_blur(1), device=device, padding="circular"
)
physics.noise_model = dinv.physics.GaussianNoise(sigma=sigma)

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

# %%
# Generate the measurement

y = physics(x)

# Plot the original image and the measurements
plot([x, y], show=True, rescale_mode="clip", titles=["Original Image", "Measurement"])

# %%
# Define the likelihood
# --------------------------------------------------------------
#
# Since the noise model is Gaussian, the negative log-likelihood is the L2 loss.
#
# .. math::
#   -\log p(y|x) \propto \frac{1}{2\sigma^2} \|y-Ax\|^2

# load Gaussian Likelihood
likelihood = dinv.optim.data_fidelity.L2(sigma=sigma)

# %%
# Define the generative prior
# -------------------------------------------
#
# The prior is a "one-pass" generative model (such as a VAE, GAN, or discrete flow model). Currently
# supported models include :class:`deepinv.models.VAE`, :class:`deepinv.models.DCGANGenerator` and :class:`deepinv.models.MbtCAE`
# (variational compressive autoencoder (CAE) introduced in
# `"Joint Autoregressive and Hierarchical Priors for Learned Image Compression" <https://arxiv.org/abs/1808.02736>`_
# and that can be used as a regularizer in VBLE).

prior = dinv.models.MbtCAE(pretrained="download", decode_mean_only=False)

# %% Optimize the variational parameters
# --------------------------------------------------------------
#
# We create a :class:`deepinv.models.VBLEzOptimizer` model to optimize the variational parameters.
# `rate_scale` controls the rate-distortion trade-off for CAEs (between 0 and 1, the lower the stronger the regularization).
#
# .. note::
# VBLE-xz algorithm can also be used by replacing :class:`deepinv.models.VBLEzOptimizer` with :class:`deepinv.models.VBLExzOptimizer`.
# It enables to jointly optimize the variational parameters in the latent space and the image space.
# This produces more accurate uncertainty estimates (especially for inpainting problems) at the cost of a higher computational cost.
vble_model = dinv.models.VBLEzOptimizer(prior, max_iters=150, rate_scale=0.25).to(
    device
)

# Compute the optimal variational parameters, and returns one posterior sample
print("VBLE optimization...")
x_hat = vble_model(y, physics, likelihood, verbose=True)

# Show the reconstruction
plot(
    [y, x_hat, x],
    show=True,
    vmin=0,
    vmax=1,
    rescale_mode="clip",
    titles=["Measurement", "Reconstruction", "Ground Truth"],
)

loss_dict = vble_model.get_loss()

# Plot loss curves
plot_curves(loss_dict, show=True)

# %%
# Create the VBLE-z sampler
# --------------------------------------------------------------
#
# We create a :class:`deepinv.sampling.VBLESampling` sampler to sample from the posterior distribution.
# `max_iter` controls the total number of samples drawn while `batch_size` controls the number of samples drawn in each batch.
# `save_chain` indicates whether to save all samples drawn during the sampling process.

vble_sampler = dinv.sampling.VBLESampling(
    vble_model, save_chain=True, max_iter=100, batch_size=16
)

# %%
# Run sampling algorithm and plot results
# --------------------------------------------------------------
# The sampling algorithm returns the posterior mean and variance.

print("Sampling 100 samples from the posterior distribution...")
mean, var = vble_sampler.sample()
samples = vble_sampler.get_chain()

imgs = [y, x_hat, mean]
plot(
    imgs,
    show=False,
    rescale_mode="clip",
    titles=["Measurement", "One Sample", "Posterior Mean"],
)
plot([var.sqrt()], show=True, titles=["Posterior Standard Deviation"])
