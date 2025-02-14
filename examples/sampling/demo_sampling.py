r"""
Uncertainty quantification with PnP-ULA.
====================================================================================================

This code shows you how to use sampling algorithms to quantify uncertainty of a reconstruction
from incomplete and noisy measurements.

ULA obtains samples by running the following iteration:

.. math::

    x_{k+1} = x_k +  \alpha \eta \nabla \log p_{\sigma}(x_k) + \eta \nabla \log p(y|x_k)  + \sqrt{2 \eta} z_k

where :math:`z_k \sim \mathcal{N}(0, I)` is a Gaussian random variable, :math:`\eta` is the step size and
:math:`\alpha` is a parameter controlling the regularization.

The PnP-ULA method is described in the paper `"Bayesian imaging using Plug & Play priors: when Langevin meets Tweedie
" <https://arxiv.org/abs/2103.04715>`_.

"""

import deepinv as dinv
from deepinv.utils.plotting import plot
import torch
from deepinv.utils.demo import load_url_image

# %%
# Load image from the internet
# --------------------------------------------
#
# This example uses an image of Lionel Messi from Wikipedia.

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

url = (
    "https://upload.wikimedia.org/wikipedia/commons/b/b4/"
    "Lionel-Messi-Argentina-2022-FIFA-World-Cup_%28cropped%29.jpg"
)
x = load_url_image(url=url, img_size=32).to(device)

# %%
# Define forward operator and noise model
# --------------------------------------------------------------
#
# This example uses inpainting as the forward operator and Gaussian noise as the noise model.

sigma = 0.1  # noise level
physics = dinv.physics.Inpainting(mask=0.5, tensor_size=x.shape[1:], device=device)
physics.noise_model = dinv.physics.GaussianNoise(sigma=sigma)

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

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
# Define the prior
# -------------------------------------------
#
# The score a distribution can be approximated using Tweedie's formula via the
# :class:`deepinv.optim.ScorePrior` class.
#
# .. math::
#
#            \nabla \log p_{\sigma}(x) \approx \frac{1}{\sigma^2} \left(D(x,\sigma)-x\right)
#
# This example uses a pretrained DnCNN model.
# From a Bayesian point of view, the score plays the role of the gradient of the
# negative log prior
# The hyperparameter ``sigma_denoiser`` (:math:`sigma`) controls the strength of the prior.
#
# In this example, we use a pretrained DnCNN model using the :class:`deepinv.loss.FNEJacobianSpectralNorm` loss,
# which makes sure that the denoiser is firmly non-expansive (see
# `"Building firmly nonexpansive convolutional neural networks" <https://hal.science/hal-03139360>`_), and helps to
# stabilize the sampling algorithm.

sigma_denoiser = 2 / 255
prior = dinv.optim.ScorePrior(
    denoiser=dinv.models.DnCNN(pretrained="download_lipschitz")
).to(device)

# %%
# Create the MCMC sampler
# --------------------------------------------------------------
#
# Here we use the Unadjusted Langevin Algorithm (ULA) to sample from the posterior defined in
# :class:`deepinv.sampling.ULA`.
# The hyperparameter ``step_size`` controls the step size of the MCMC sampler,
# ``regularization`` controls the strength of the prior and
# ``iterations`` controls the number of iterations of the sampler.

regularization = 0.9
step_size = 0.01 * (sigma**2)
iterations = int(5e3) if torch.cuda.is_available() else 10
f = dinv.sampling.ULA(
    prior=prior,
    data_fidelity=likelihood,
    max_iter=iterations,
    alpha=regularization,
    step_size=step_size,
    verbose=True,
    sigma=sigma_denoiser,
)

# %%
# Generate the measurement
# --------------------------------------------------------------
# We apply the forward model to generate the noisy measurement.

y = physics(x)


# %%
# Run sampling algorithm and plot results
# --------------------------------------------------------------
# The sampling algorithm returns the posterior mean and variance.
# We compare the posterior mean with a simple linear reconstruction.

mean, var = f(y, physics)

# compute linear inverse
x_lin = physics.A_adjoint(y)

# compute PSNR
print(f"Linear reconstruction PSNR: {dinv.metric.PSNR()(x, x_lin).item():.2f} dB")
print(f"Posterior mean PSNR: {dinv.metric.PSNR()(x, mean).item():.2f} dB")

# plot results
error = (mean - x).abs().sum(dim=1).unsqueeze(1)  # per pixel average abs. error
std = var.sum(dim=1).unsqueeze(1).sqrt()  # per pixel average standard dev.
imgs = [x_lin, x, mean, std / std.flatten().max(), error / error.flatten().max()]
plot(
    imgs,
    titles=["measurement", "ground truth", "post. mean", "post. std", "abs. error"],
)
