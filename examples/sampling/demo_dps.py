r"""
Implementing DPS
================

In this tutorial, we will go over the steps in the Diffusion Posterior Sampling (DPS) algorithm introduced in
:footcite:t:`chung2022diffusion`. The full algorithm is implemented in :class:`deepinv.sampling.DPS`.
"""

# %%
# Installing dependencies
# -----------------------
# Let us ``import`` the relevant packages, and load a sample
# image of size 64 x 64. This will be used as our ground truth image.
#
# .. note::
#           We work with an image of size 64 x 64 to reduce the computational time of this example.
#           The DiffUNet we use in the algorithm works best with images of size 256 x 256.
#

import torch

import deepinv as dinv
from deepinv.utils.plotting import plot
from deepinv.utils import load_example

device = dinv.utils.get_device()

x_true = load_example("butterfly.png", img_size=64, device=device)
x = x_true.clone()

# %%
# In this tutorial we consider random inpainting as the inverse problem, where the forward operator is implemented
# in :class:`deepinv.physics.Inpainting`. In the example that we use, 90% of the pixels will be masked out randomly,
# and we will additionally have Additive White Gaussian Noise (AWGN) of standard deviation  12.75/255.

sigma = 12.75 / 255.0  # noise level

physics = dinv.physics.Inpainting(
    img_size=(3, x.shape[-2], x.shape[-1]),
    mask=0.1,
    pixelwise=True,
    noise_model=dinv.physics.GaussianNoise(sigma),
    device=device,
)

y = physics(x_true)

plot(
    {
        "Measurement": y,
        "Ground Truth": x_true,
    }
)


# %%
# Diffusion model loading
# -----------------------
#
# We will take a pre-trained diffusion model that was also used for the DiffPIR algorithm, namely the one trained on
# the FFHQ 256x256 dataset. Note that this means that the diffusion model was trained with human face images,
# which is very different from the image that we consider in our example. Nevertheless, we will see later on that
# ``DPS`` generalizes sufficiently well even in such case.


denoiser = dinv.models.DRUNet(device=device)

# %%
# Define diffusion schedule
# -------------------------
#
# We will use the standard linear diffusion noise schedule. Once :math:`\beta_t` is defined to follow a linear schedule
# that interpolates between :math:`\beta_{\rm min}` and :math:`\beta_{\rm max}`,
# we have the following additional definitions:
# :math:`\alpha_t := 1 - \beta_t`, :math:`\bar\alpha_t := \prod_{j=1}^t \alpha_j`.
# The following equations will also be useful
# later on (we always assume that :math:`\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})` hereafter.)
#
# .. math::
#
#           \mathbf{x}_t = \sqrt{1 - \beta_t}\mathbf{x}_{t-1} + \sqrt{\beta_t}\mathbf{\epsilon}
#
#           \mathbf{x}_t = \sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1 - \bar\alpha_t}\mathbf{\epsilon}
#
# where we use the reparametrization trick.
#
#  In continuous time, this schedule corresponds to the Variance Preserving (VP) SDE, see :class:`deepinv.sampling.VariancePreservingSDE`.

# %%
# The DPS algorithm
# -----------------
#
# Now that the inverse problem is defined, we can apply the DPS algorithm to solve it. The DPS algorithm is
# a diffusion algorithm that alternates between a denoising step, a gradient step and a reverse diffusion sampling step.
# The algorithm writes as follows, for :math:`t` decreasing from :math:`T` to :math:`1`:
#
# .. math::
#
#         \widehat{\mathbf{x}}_{0} (\mathbf{x}_t) &= \denoiser{\mathbf{x}_t}{\sqrt{1-\overline{\alpha}_t}/\sqrt{\overline{\alpha}_t}}
#         \\
#         \mathbf{g}_t &= \nabla_{\mathbf{x}_t} \log p( \widehat{\mathbf{x}}_{0}(\mathbf{x}_t) | \mathbf{y} ) \\
#         \mathbf{\varepsilon}_t &= \mathcal{N}(0, \mathbf{I}) \\
#         \mathbf{x}_{t-1} &= a_t \,\, \mathbf{x}_t
#         + b_t \, \, \widehat{\mathbf{x}}_0
#         + \tilde{\sigma}_t \, \, \mathbf{\varepsilon}_t + \mathbf{g}_t,
#
#
# where :math:`\denoiser{\cdot}{\sigma}` is a denoising network for noise level :math:`\sigma`,
# :math:`\eta` is a hyperparameter in [0, 1], and the constants :math:`\tilde{\sigma}_t, a_t, b_t` are defined as
#
# .. math::
#
#           \tilde{\sigma}_t &= \eta \sqrt{ (1 - \frac{\overline{\alpha}_t}{\overline{\alpha}_{t-1}})
#           \frac{1 - \overline{\alpha}_{t-1}}{1 - \overline{\alpha}_t}} \\
#           a_t &= \sqrt{1 - \overline{\alpha}_{t-1} - \tilde{\sigma}_t^2}/\sqrt{1-\overline{\alpha}_t} \\
#           b_t &= \sqrt{\overline{\alpha}_{t-1}} - \sqrt{1 - \overline{\alpha}_{t-1} - \tilde{\sigma}_t^2}
#           \frac{\sqrt{\overline{\alpha}_{t}}}{\sqrt{1 - \overline{\alpha}_{t}}}
#
#


# %%
# Denoising step
# --------------
#
# The first step of DPS consists of applying a denoiser function to the current image :math:`\mathbf{x}_t`,
# with standard deviation :math:`\sigma_t = \sqrt{1 - \overline{\alpha}_t}/\sqrt{\overline{\alpha}_t}`.
#
# This is equivalent to sampling :math:`\mathbf{x}_t \sim q(\mathbf{x}_t|\mathbf{x}_0)`, and then computing the
# posterior mean.
#


t = 200
# choose some arbitrary noise level
sigma_t = 0.2

x0 = x_true
xt = x0 + sigma_t * torch.randn_like(x0)

# apply denoiser
with torch.no_grad():
    x0_t = denoiser(xt, sigma_t)

# Visualize
plot(
    {
        "Ground Truth": x0,
        "Noisy": xt,
        "Posterior Mean": x0_t,
    }
)

# %%
# DPS approximation
# -----------------
#
# In order to perform gradient-based **posterior sampling** with diffusion models, we have to be able to compute
# :math:`\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t|\mathbf{y})`. Applying Bayes rule, we have
#
# .. math::
#
#           \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t|\mathbf{y}) = \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)
#           + \nabla_{\mathbf{x}_t} \log p(\mathbf{y}|\mathbf{x}_t)
#
# For the former term, we can simply plug-in our estimated score function as in Tweedie's formula. As the latter term
# is intractable, DPS proposes the following approximation (for details, see Theorem 1 of :footcite:t:`chung2022diffusion`)
#
# .. math::
#
#           \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t|\mathbf{y}) \approx \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)
#           + \nabla_{\mathbf{x}_t} \log p(\mathbf{y}|\widehat{\mathbf{x}}_{0}(\mathbf{x_t}))
#
# Remarkably, we can now compute the latter term when we have Gaussian noise, as
#
# .. math::
#
#       \log p(\mathbf{y}|\widehat{\mathbf{x}}_0(\mathbf{x_t})) =
#       -\frac{\|\mathbf{y} - A\widehat{\mathbf{x}}_0((\mathbf{x_t})\|_2^2}{2\sigma_y^2}.
#
# Moreover, taking the gradient w.r.t. :math:`\mathbf{x}_t` can be performed through automatic differentiation.
# We provide an implementation of this approximation in :class:`deepinv.sampling.DPSDataFidelity`, which is a subclass of :class:`deepinv.optim.data_fidelity.NoisyDataFidelity`.
#
# .. note::
#           The DPS algorithm assumes that the images are in the range [-1, 1], whereas standard denoisers
#           usually output images in the range [0, 1]. This is why we rescale the images before applying the steps.

from deepinv.sampling import DPSDataFidelity

x0 = x_true * 2.0 - 1.0  # [0, 1] -> [-1, 1]

data_fidelity = DPSDataFidelity(denoiser=denoiser, clip=(-1.0, 1.0))

# choose some arbitrary noise level
sigma_t = 0.2
xt = x0 + sigma_t * torch.randn_like(x0)

# DPS
grad, x0_t = data_fidelity.grad(
    xt / 2 + 0.5, y=y, physics=physics, sigma=sigma_t / 2, get_model_outputs=True
)  # Set get_model_outputs to True to also retrieve the denoised output


# Visualize
plot(
    {
        "Ground Truth": x0,
        "Noisy": xt,
        "Posterior Mean": x0_t,
        "Gradient": grad,
    }
)

# %%
# DPS Algorithm
# -------------
#
# As we visited all the key components of DPS, we are now ready to define the algorithm. For every denoising
# timestep, the algorithm iterates the following
#
# 1. Get :math:`\hat{\mathbf{x}}` using the denoiser network.
# 2. Compute :math:`\nabla_{\mathbf{x}_t} \log p(\mathbf{y}|\hat{\mathbf{x}}_t)` through backpropagation.
# 3. Perform reverse diffusion sampling with DDPM(IM), corresponding to an update with :math:`\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)`.
# 4. Take a gradient step with :math:`\nabla_{\mathbf{x}_t} \log p(\mathbf{y}|\hat{\mathbf{x}}_t)`.
#
# There are two caveats here. First, in the original work, DPS used DDPM ancestral sampling. As the DDIM sampler :footcite:t:`song2020denoising`
# is a generalization of DDPM in a sense that it retrieves DDPM when
# :math:`\eta = 1.0`, here we consider DDIM sampling.
# One can freely choose the :math:`\eta` parameter here, but since we will consider 1000
# neural function evaluations (NFEs),
# it is advisable to keep it :math:`\eta = 1.0`. Second, when taking the log-likelihood gradient step,
# the gradient is weighted so that the actual implementation is a static step size times the :math:`\ell_2`
# norm of the residual:
#
# .. math::
#
#           \nabla_{\mathbf{x}_t} \log p(\mathbf{y}|\hat{\mathbf{x}}_{t}(\mathbf{x}_t)) \simeq
#           \rho \nabla_{\mathbf{x}_t} \|\mathbf{y} - \mathbf{A}\hat{\mathbf{x}}_{t}\|_2
#
# With DeepInverse, we can use the :class:`deepinv.sampling.DPS` class to perform the above steps with minimal code, with some important parameters:
#   - `weight`: corresponds to the :math:`\rho` parameter in the above equation, which controls the strength of the gradient step.
#   - `eta`: corresponds to the :math:`\eta` parameter in the DDIM, which controls the strength of the noise in the reverse diffusion sampling step.
#   - `num_steps`: corresponds to the number of denoising steps, which is usually set to 1000 for best performance, but can be reduced to 200 for faster sampling.


# %%
# .. note::
#
#   We only use 200 steps to reduce the computational time of this example. As suggested by the authors of DPS, the
#   algorithm works best with ``num_steps = 1000``.
#

# Instantiate the model
model = dinv.sampling.DPS(
    denoiser=denoiser,
    num_steps=200,
    weight=2.0,
    eta=1.0,
    verbose=True,
    device=device,
    dtype=torch.float32,
)

# Run the sampling
sample, trajectory = model(
    y,
    physics,
    seed=123,  # for reproducibility!
    get_trajectory=True,
)
# plot the results
plot(
    {
        "Measurement": y,
        "Model Output": sample,
        "Ground Truth": x_true,
    }
)

dinv.utils.plot_videos(
    trajectory[::10], time_dim=0, suptitle="DPS Trajectory", display=True
)

# %%
# :References:
#
# .. footbibliography::
