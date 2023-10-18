r"""
Implementing DiffPIR
====================

In this tutorial, we revisit the implementation of the DiffPIR diffusion algorithm for image reconstruction from
`Zhou et al. <https://arxiv.org/abs/2305.08995>`_. The full algorithm is implemented in
:class:`deepinv.sampling.diffusion.DiffPIR`.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import deepinv as dinv
from deepinv.utils.plotting import plot
from deepinv.optim.data_fidelity import L2

# %%
# Generate an inverse problem
# ---------------------------
#
# We first generate a deblurring problem with the Butterly image. We use a square blur kernel of size 5x5 and
# Gaussian noise with standard deviation 12.75/255.0.

from deepinv.utils.demo import load_url_image

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

url = (
    "https://mycore.core-cloud.net/index.php/s/9EzDqcJxQUJKYul/"
    "download?path=%2Fdatasets&files=butterfly.png"
)
x_true = load_url_image(url=url, img_size=32, device=device)
x = x_true.clone()

sigma = 12.75 / 255.0  # noise level

physics = dinv.physics.BlurFFT(
    img_size=(3, x.shape[-2], x.shape[-1]),
    filter=torch.ones((1, 1, 5, 5), device=device) / 25,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=sigma),
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
#         x_{0}^{t} &= \left(x_t + (1 - \overline{\alpha}_t)\mathbf{s}_\theta(x_t,t))\right)/\sqrt{\overline{\alpha}_t} \\
#         \widehat{x}_{0}^{t} &= \operatorname{prox}_{2 f(y, \cdot) /{\rho_t}}(x_{0}^{t}) \\
#         \widehat{\varepsilon} &= \left(x_t - \sqrt{\overline{\alpha}_t} \,\, \widehat{x}_{0}^t\right)/\sqrt{1-\overline{\alpha}_t} \\
#         \varepsilon_t &= \mathcal{N}(0, \mathbf{I}) \\
#         x_{t-1} &= \sqrt{\overline{\alpha}_t} \,\, \widehat{x}_{0}^t + \sqrt{1-\overline{\alpha}_t} \left(\sqrt{1-\zeta} \,\, \widehat{\varepsilon} + \sqrt{\zeta} \,\, \varepsilon_t\right),
#         \end{aligned}
#         \end{equation*}
#
# where :math:`\mathbf{s}_\theta` is a denoising network, :math:`\mathcal{N}(0, \mathbf{I})` is a Gaussian noise
# with zero mean and unit variance, and :math:`\zeta` is a parameter that controls the amount of noise added at each
# iteration. The parameters :math:`(\overline{\alpha}_t)_{0\leq t\leq T}` and :math:`(\rho_t)_{0\leq t\leq T}` are
# sequences of positive numbers, which we will detail later on.
#
# Let us now implement each step of this algorithm.


# %%
# Denoising step
# --------------
#
# In this section, we show how to use the denoising diffusion model from DiffPIR.
# The denoising step is implemented by a denoising network conditioned on the noise power. The authors
# of DiffPIR use a U-Net architecture, which can be loaded as follows:

model = dinv.models.DiffUNet(large_model=True).to(device)

# %%
# Now, recall that the forward diffusion can be rewritten as, for all :math:`t`,
#
# .. math::
#         \begin{equation*}
#         x_t = \sqrt{\overline{\alpha}_t} \widehat{x}_{0} + \sqrt{1-\overline{\alpha}_t} \varepsilon_t.
#         \end{equation*}
#
# In this context, the denoising diffusion model :math:`s_theta(x, t)` is trained to approximate :math:`\varepsilon_t`
# from :math:`x_t` and :math:`t`. The denoised image thus writes:
#
# .. math::
#         \begin{equation*}
#         \widehat{x}_{0} \approx \frac{1}{\sqrt{\overline{\alpha}_t}}x_t-(\frac{1}{\sqrt{\overline{\alpha}_t}}-1) \mathbf{s}_\theta(x_t, t)
#         \end{equation*}
#
# Before being able to use the pretrained model, we need to define the sequence
# :math:`(\overline{\alpha}_t)_{0\leq t\leq T}`. In practice, we will not use :math:`\overline{\alpha}_t` directly,
# but rather other sequences directly deriving from it (e.g. :math:`(1/\sqrt{\overline{\alpha}_t})_{0\leq t\leq T}`).
# The following function returns these sequences:
#


num_train_timesteps = 1000  # Number of timesteps used during training


def get_alpha_beta(
    beta_start=0.1 / 1000, beta_end=20 / 1000, num_train_timesteps=num_train_timesteps
):
    betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    betas = torch.from_numpy(betas).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas.cpu(), axis=0)  # This is \overline{\alpha}_t

    # Useful sequences deriving from alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    reduced_alpha_cumprod = torch.div(
        sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod
    )  # equivalent noise sigma on image
    sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

    return (
        sqrt_1m_alphas_cumprod,
        reduced_alpha_cumprod,
        sqrt_alphas_cumprod,
        sqrt_recip_alphas_cumprod,
        sqrt_recipm1_alphas_cumprod,
        betas,
    )


(
    sqrt_1m_alphas_cumprod,
    reduced_alpha_cumprod,
    sqrt_alphas_cumprod,
    sqrt_recip_alphas_cumprod,
    sqrt_recipm1_alphas_cumprod,
    betas,
) = get_alpha_beta()

# plot the alpha sequence
plt.rcParams.update({"font.size": 9})
plt.figure(figsize=(6, 3))
plt.plot(reduced_alpha_cumprod)
plt.ylabel(r"$\frac{\sqrt{1-\overline{\alpha}_t}}{\sqrt{\overline{\alpha}_t}}$")
plt.xlabel(r"$t$")
plt.tight_layout()
plt.show()

# %%
# Now that we have the sequence of interest, there remains to link noise power to the timestep. The following function
# returns the timestep corresponding to a given noise power.


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# We can now apply the model to a noisy image. We first generate a noisy image
x_noisy = x_true + torch.randn_like(x_true) * sigma

den = model(x_noisy, sigma)

imgs = [x_noisy, den, den - x_true]
plot(
    imgs,
    titles=["noisy input", "denoised image", "error"],
)

# %%
# Data fidelity step
# ------------------
#
# The data fidelity step is easily implemented in deepinv. We simply need to define a data fidelity function and use
# its prox attribute. For instance:

data_fidelity = L2()

# In order to take a meaningful data fidelity step, it is best if we apply it to denoised measurements.
# First, denoise the measurements:
y_denoised = model(y, sigma)

# Next, apply the proximity operator of the data fidelity term (this is the data fidelity step). In the algorithm,
# the regularization parameter is carefully chosen. Here, for simplicity, we set it to 1/sigma.
x_prox = data_fidelity.prox(y_denoised, y, physics, gamma=1 / sigma)

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
# Firstly, we need to compute the effective noise in the estimated reconstruction, i.e. the residual between the previous
# reconstruction and the data fidelity step. This is done as follows:

x_prox_scaled = 2 * x_prox - 1  # Rescale the output of the proximal step in [-1, 1]
y_scaled = 2 * y - 1  # Rescale the measurement in [-1, 1]

t_i = find_nearest(reduced_alpha_cumprod.cpu().numpy(), sigma * 2) # time step associated with the noise level sigma
eps = (y_scaled - sqrt_alphas_cumprod[t_i] * x_prox_scaled) / sqrt_1m_alphas_cumprod[
    t_i
]

# (notice the rescaling)
#
# Secondly, we need to perform the sampling step, which is a linear combination between the estimated noise and
# the realizations of a Gaussian white noise. This is done as follows:
zeta = 0.3
x_sampled_scaled = (
    sqrt_alphas_cumprod[t_i - 1] * x_prox_scaled
    + np.sqrt(1 - zeta) * (torch.sqrt(sqrt_1m_alphas_cumprod[t_i - 1] ** 2) * eps)
    + np.sqrt(zeta) * sqrt_1m_alphas_cumprod[t_i - 1] * torch.randn_like(x)
)

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
# ---------------------------------------------
#
# We can now put all the steps together and implement the DiffPIR algorithm. The only remaining step is to set the
# noise schedule (i.e. the sequence of noise powers and regularization parameters) appropriately. This is done with the
# following function:

diffusion_steps = 30  # Maximum number of iterations of the DiffPIR algorithm

# %%
# .. note::
#
#   We only use 30 steps to reduce the computational time of this example. As suggested by the authors of DiffPIR, the
#   algorithm works best with ``max_iter = 100``.
#


def get_noise_schedule(max_iter=diffusion_steps, num_train_timesteps=num_train_timesteps):
    lambda_ = 7.0

    sigmas = []
    sigma_ks = []
    rhos = []
    for i in range(num_train_timesteps):
        sigmas.append(reduced_alpha_cumprod[num_train_timesteps - 1 - i])
        sigma_ks.append((sqrt_1m_alphas_cumprod[i] / sqrt_alphas_cumprod[i]))
        rhos.append(lambda_ * (sigma**2) / (sigma_ks[i] ** 2))
    rhos, sigmas = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device)

    seq = np.sqrt(np.linspace(0, num_train_timesteps**2, max_iter))
    seq = [int(s) for s in list(seq)]
    seq[-1] = seq[-1] - 1

    return rhos, sigmas, seq


rhos, sigmas, seq = get_noise_schedule()

# Plot the noise and regularization schedules
plt.figure(figsize=(6, 3))
plt.subplot(121)
plt.plot(2 / rhos.cpu().numpy()[::-1])  # Note that the regularization parameter is 2/rho and not rho
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
x = 2 * physics.A_adjoint(y) - 1

with torch.no_grad():
    for i in tqdm(range(len(seq))):

        # Current noise level and associated time step
        curr_sigma = sigmas[seq[i]].cpu().numpy()
        t_i = find_nearest(reduced_alpha_cumprod, curr_sigma)

        # Denoising step
        x0 = model(x, curr_sigma)

        if not seq[i] == seq[-1]:
            # Data fidelity step
            x0 = data_fidelity.prox(x0, y, physics, gamma=1 / (2 * rhos[t_i]))

            # Normalize data for sampling
            x0 = 2*x0-1
            x = 2*x-1

            # Sampling step
            t_im1 = find_nearest(
                reduced_alpha_cumprod, sigmas[seq[i + 1]].cpu().numpy()
            )  # time step associated with the next noise level
            eps = (x - sqrt_alphas_cumprod[t_i] * x0) / sqrt_1m_alphas_cumprod[
                t_i
            ]  # effective noise
            x = (
                sqrt_alphas_cumprod[t_im1] * x0
                + torch.sqrt(sqrt_1m_alphas_cumprod[t_im1] ** 2) * np.sqrt(1 - zeta) * eps
                + sqrt_1m_alphas_cumprod[t_im1] * np.sqrt(zeta) * torch.randn_like(x)
            )  # sampling

            # Rescale the output in [0, 1]
            x = (x + 1) / 2


# Plotting the results
imgs = [y, x, x_true]
plot(
    imgs,
    titles=["measurement", "model output", "ground-truth"],
)
