r"""
Implementing DiffPIR
====================

In this tutorial, we revisit the implementation of the DiffPIR diffusion algorithm for image reconstruction from
`Zhou et al. <https://arxiv.org/abs/2305.08995>`_. The full algorithm is implemented in
:class:`deepinv.sampling.DiffPIR`.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import deepinv as dinv
from deepinv.utils.plotting import plot
from deepinv.optim.data_fidelity import L2
from deepinv.utils.demo import load_url_image, get_image_url

# Use matplotlib config from deepinv to get nice plots
from deepinv.utils.plotting import config_matplotlib

config_matplotlib()

# %%
# Generate an inverse problem
# ---------------------------
#
# We first generate a deblurring problem with the butterfly image. We use a square blur kernel of size 5x5 and
# Gaussian noise with standard deviation 12.75/255.0.
#
# .. note::
#           We work with an image of size 64x64 to reduce the computational time of this example.
#           The algorithm works best with images of size 256x256.
#


device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

url = get_image_url("butterfly.png")

x_true = load_url_image(url=url, img_size=64, device=device)

x = x_true.clone()

sigma_noise = 12.75 / 255.0  # noise level

physics = dinv.physics.BlurFFT(
    img_size=(3, x.shape[-2], x.shape[-1]),
    filter=torch.ones((1, 1, 5, 5), device=device) / 25,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=sigma_noise),
)

y = physics(x)

imgs = [y, x_true]
plot(
    imgs,
    titles=["measurement", "ground-truth"],
)

# %%
# The DiffPIR algorithm
# ---------------------
#
# Now that the inverse problem is defined, we can apply the DiffPIR algorithm to solve it. The DiffPIR algorithm is
# a diffusion algorithm that alternates between a denoising step, a proximal step and a reverse diffusion sampling step.
# The algorithm writes as follows, for :math:`t` decreasing from :math:`T` to :math:`1`:
#
# .. math::
#         \begin{equation*}
#         \begin{aligned}
#         \mathbf{x}_{0}^{t} &= \denoiser{\mathbf{x}_t}{\sqrt{1-\overline{\alpha}_t}/\sqrt{\overline{\alpha}_t}} \\
#         \widehat{\mathbf{x}}_{0}^{t} &= \operatorname{prox}_{2 f(y, \cdot) /{\rho_t}}(\mathbf{x}_{0}^{t}) \\
#         \widehat{\mathbf{\varepsilon}} &= \left(\mathbf{x}_t - \sqrt{\overline{\alpha}_t} \,\,
#         \widehat{\mathbf{x}}_{0}^t\right)/\sqrt{1-\overline{\alpha}_t} \\
#         \mathbf{\varepsilon}_t &= \mathcal{N}(0, \mathbf{I}) \\
#         \mathbf{x}_{t-1} &= \sqrt{\overline{\alpha}_t} \,\, \widehat{\mathbf{x}}_{0}^t + \sqrt{1-\overline{\alpha}_t}
#         \left(\sqrt{1-\zeta} \,\, \widehat{\mathbf{\varepsilon}} + \sqrt{\zeta} \,\, \mathbf{\varepsilon}_t\right),
#         \end{aligned}
#         \end{equation*}
#
# where :math:`\denoiser{\cdot}{\sigma}` is a denoising network with noise level :math:`\sigma`,
# :math:`\mathcal{N}(0, \mathbf{I})` is a Gaussian noise
# with zero mean and unit variance, :math:`\zeta` is a parameter that controls the amount of noise added at each
# iteration and :math:`f` refers to the data fidelity/measurement consistency term,
# which for Gaussian Noise (implemented as :class:`deepinv.optim.L2`) is given by:
#
# .. math::
#               f(\mathbf{y}, \mathbf{x}) = \frac{1}{2}\|\mathbf{y} - \mathcal{A}(\mathbf{x})\|^2
#
# Note that other data fidelity terms can be used, such as :class:`deepinv.optim.PoissonLikelihood`.
# The parameters :math:`(\overline{\alpha}_t)_{0\leq t\leq T}` and :math:`(\rho_t)_{0\leq t\leq T}` are
# sequences of positive numbers, which we will detail later on.
#
# Let us now implement each step of this algorithm.


# %%
# Denoising step
# --------------
#
# In this section, we show how to use the denoising diffusion model from DiffPIR.
# The denoising step is implemented by a denoising network conditioned on the noise power. The authors
# of DiffPIR use a U-Net architecture from `Ho et al. <https://arxiv.org/abs/2108.02938>`_,
# which can be loaded as follows:

model = dinv.models.DiffUNet(large_model=False).to(device)

# %%
#
# Before being able to use the pretrained model, we need to define the sequence
# :math:`(\overline{\alpha}_t)_{0\leq t\leq T}`.
# The following function returns these sequence:
#

T = 1000  # Number of timesteps used during training


def get_alphas(beta_start=0.1 / 1000, beta_end=20 / 1000, num_train_timesteps=T):
    betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    betas = torch.from_numpy(betas).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas.cpu(), axis=0)  # This is \overline{\alpha}_t
    return torch.tensor(alphas_cumprod)


alphas = get_alphas()


# %%
# Now that we have the sequence of interest, there remains to link noise power to the timestep. The following function
# returns the timestep corresponding to a given noise power, which is given by
#
# .. math::
#           \sigma_t = \sqrt{1-\overline{\alpha}_t}/\overline{\alpha}_t.


sigmas = torch.sqrt(1.0 - alphas) / alphas.sqrt()


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


t = 100  # choose arbitrary timestep

# We can now apply the model to a noisy image. We first generate a noisy image
x_noisy = x_true + torch.randn_like(x_true) * sigmas[t]

den = model(x_noisy, sigmas[t])

imgs = [x_noisy, den, den - x_true]
plot(
    imgs,
    titles=["noisy input", "denoised image", "error"],
)

# %%
# Data fidelity step
# ------------------
#
# The data fidelity step is easily implemented in this library. We simply need to define a data fidelity function and use
# its prox attribute. For instance:

data_fidelity = L2()

# In order to take a meaningful data fidelity step, it is best if we apply it to denoised measurements.
# First, denoise the measurements:
y_denoised = model(y, sigmas[t])

# Next, apply the proximity operator of the data fidelity term (this is the data fidelity step). In the algorithm,
# the regularization parameter is carefully chosen. Here, for simplicity, we set it to :math:`1/\sigma`.
x_prox = data_fidelity.prox(y_denoised, y, physics, gamma=1 / sigmas[t])

imgs = [y, y_denoised, x_prox]
plot(
    imgs,
    titles=["measurement", "denoised measurement", "data fidelity step"],
)

# %%
# Sampling step
# -------------
#
# The last step to be implemented is the DiffPIR sampling step and this can be computed in two steps.
# Firstly, we need to compute the effective noise in the estimated reconstruction,
# i.e. the residual between the previous
# reconstruction and the data fidelity step. This is done as follows:
#
# .. note::
#           The diffPIR algorithm assumes that the images are in the range [-1, 1], whereas standard denoisers
#           usually output images in the range [0, 1]. This is why we rescale the images before applying the steps.

x_prox_scaled = 2 * x_prox - 1  # Rescale the output of the proximal step in [-1, 1]
y_scaled = 2 * y - 1  # Rescale the measurement in [-1, 1]

t_i = find_nearest(
    sigmas.cpu().numpy(), sigma_noise * 2
)  # time step associated with the noise level sigma
eps = (y_scaled - alphas[t_i].sqrt() * x_prox_scaled) / torch.sqrt(
    1.0 - alphas[t_i]
)  # effective noise

# (notice the rescaling)
#
# Secondly, we need to perform the sampling step, which is a linear combination between the estimated noise and
# the realizations of a Gaussian white noise. This is done as follows:
zeta = 0.3
x_sampled_scaled = alphas[t_i - 1].sqrt() * x_prox_scaled + torch.sqrt(
    1.0 - alphas[t_i - 1]
) * (np.sqrt(1 - zeta) * eps + np.sqrt(zeta) * torch.randn_like(x))

x_sampled = (x_sampled_scaled + 1) / 2  # Rescale the output in [0, 1]

imgs = [y, y_denoised, x_prox, x_sampled]
plot(
    imgs,
    titles=[
        "measurement",
        "denoised measurement",
        "data fidelity step",
        "sampling step",
    ],
)

# %%
# Putting it all together: the DiffPIR algorithm
# ------------------------------------------------
#
# We can now put all the steps together and implement the DiffPIR algorithm. The only remaining step is to set the
# noise schedule (i.e. the sequence of noise powers and regularization parameters) appropriately. This is done with the
# following function:
#
# .. note::
#
#   We only use 30 steps to reduce the computational time of this example. As suggested by the authors of DiffPIR, the
#   algorithm works best with ``diffusion_steps = 100``.
#

diffusion_steps = 30  # Maximum number of iterations of the DiffPIR algorithm

lambda_ = 7.0  # Regularization parameter

rhos = lambda_ * (sigma_noise**2) / (sigmas**2)

# get timestep sequence
seq = np.sqrt(np.linspace(0, T**2, diffusion_steps))
seq = [int(s) for s in list(seq)]
seq[-1] = seq[-1] - 1


# Plot the noise and regularization schedules
plt.figure(figsize=(6, 3))
plt.rcParams.update({"font.size": 9})
plt.subplot(121)
plt.plot(
    2 / rhos.cpu().numpy()[::-1]
)  # Note that the regularization parameter is 2/rho and not rho
plt.xlabel(r"$t$")
plt.ylabel(r"$\rho$")
plt.subplot(122)
plt.plot(sigmas.cpu().numpy()[::-1])
plt.xlabel(r"$t$")
plt.ylabel(r"$\sigma$")
plt.suptitle("Regularisation parameter and noise schedules")
plt.tight_layout()
plt.show()

# %%
# Eventually, the DiffPIR algorithm is implemented as follows:
#

# Initialization
x = 2 * y - 1

with torch.no_grad():
    for i in tqdm(range(len(seq))):
        # Current and next noise levels
        curr_sigma = sigmas[T - 1 - seq[i]].cpu().numpy()

        # 1. Denoising step
        x0 = model(x, curr_sigma)

        if not seq[i] == seq[-1]:
            # 2. Data fidelity step
            t_i = find_nearest(sigmas.cpu(), curr_sigma)

            x0 = data_fidelity.prox(x0, y, physics, gamma=1 / (2 * rhos[t_i]))

            # Normalize data for sampling
            x0 = 2 * x0 - 1
            x = 2 * x - 1

            # 3. Sampling step
            next_sigma = sigmas[T - 1 - seq[i + 1]].cpu().numpy()
            t_im1 = find_nearest(
                sigmas, next_sigma
            )  # time step associated with the next noise level

            eps = (x - alphas[t_i].sqrt() * x0) / torch.sqrt(
                1.0 - alphas[t_i]
            )  # effective noise

            x = alphas[t_im1].sqrt() * x0 + torch.sqrt(1.0 - alphas[t_im1]) * (
                np.sqrt(1 - zeta) * eps + np.sqrt(zeta) * torch.randn_like(x)
            )

            # Rescale the output in [0, 1]
            x = (x + 1) / 2


# Plotting the results
imgs = [y, x, x_true]
plot(
    imgs,
    titles=["measurement", "model output", "ground-truth"],
)


# %%
# Using the DiffPIR algorithm in your inverse problem
# ------------------------------------------------------
# You can readily use this algorithm via the :meth:`deepinv.sampling.DiffPIR` class.
#
# ::
#
#       y = physics(x)
#       model = dinv.sampling.DiffPIR(dinv.models.DiffUNet(), data_fidelity=dinv.optim.L2())
#       xhat = model(y, physics)
#
