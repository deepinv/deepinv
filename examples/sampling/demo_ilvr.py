r"""
Implementing DPS
====================

In this tutorial, we will go over the steps in the Diffusion Posterior Sampling (DPS) algorithm introduced in
`Chung et al. <https://arxiv.org/abs/2209.14687>`_ The full algorithm is implemented in
:meth:`deepinv.sampling.DPS`.
"""

# %% Installing dependencies
# -----------------------------
# Let us ``import`` the relevant packages, and load a sample
# image of size 64x64. This will be used as our ground truth image.
# .. note::
#           We work with an image of size 64x64 to reduce the computational time of this example.
#           The algorithm works best with images of size 256x256.
#

import numpy as np
import torch

import deepinv as dinv
from deepinv.utils.plotting import plot
from deepinv.optim.data_fidelity import L2
from deepinv.utils.demo import load_url_image, get_image_url
from tqdm import tqdm  # to visualize progress

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
# device = 'mps'

url = get_image_url("00000.png")
x_true = load_url_image(url=url, img_size=256).to(device)
x_true = x_true[:, :3, ...]  # remove alpha channel if present
x = x_true.clone()

# %%
# In this tutorial we consider random inpainting as the inverse problem, where the forward operator is implemented
# in :meth:`deepinv.physics.Inpainting`. In the example that we use, 90% of the pixels will be masked out randomly,
# and we will additionally have Additive White Gaussian Noise (AWGN) of standard deviation  12.75/255.

sigma = 12.75 / 255.0  # noise level

physics = dinv.physics.Downsampling(
    img_size=(3, x.shape[-2], x.shape[-1]),
    filter="bicubic",
    factor=16,
    device=device,
    noise_level=sigma,
)

y = physics(x_true)

imgs = [y, x_true]
plot(
    imgs,
    titles=["measurement", "groundtruth"],
)


# %%
# Diffusion model loading
# ----------------------------
#
# We will take a pre-trained diffusion model that was also used for the DiffPIR algorithm, namely the one trained on
# the FFHQ 256x256 dataset. Note that this means that the diffusion model was trained with human face images,
# which is very different from the image that we consider in our example. Nevertheless, we will see later on that
# ``DPS`` generalizes sufficiently well even in such case.


model = dinv.models.DiffUNet(large_model=False).to(device)

# %%
# Define diffusion schedule
# ----------------------------
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
#           \mathbf{x}_t = \sqrt{\beta_t}\mathbf{x}_{t-1} + \sqrt{1 - \beta_t}\mathbf{\epsilon}
#
#           \mathbf{x}_t = \sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1 - \bar\alpha_t}\mathbf{\epsilon}
#
# where we use the reparametrization trick.

num_train_timesteps = 1000  # Number of timesteps used during training


def get_betas(
    beta_start=0.1 / 1000, beta_end=20 / 1000, num_train_timesteps=num_train_timesteps
):
    betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    betas = torch.from_numpy(betas).to(device)

    return betas


# Utility function to let us easily retrieve \bar\alpha_t
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


betas = get_betas()


# %%
# The DPS algorithm
# ---------------------
#
# Now that the inverse problem is defined, we can apply the DPS algorithm to solve it. The DPS algorithm is
# a diffusion algorithm that alternates between a denoising step, a gradient step and a reverse diffusion sampling step.
# The algorithm writes as follows, for :math:`t` decreasing from :math:`T` to :math:`1`:
#
# .. math::
#         \begin{equation*}
#         \begin{aligned}
#         \widehat{\mathbf{x}}_{t} &= \denoiser{\mathbf{x}_t}{\sqrt{1-\overline{\alpha}_t}/\sqrt{\overline{\alpha}_t}}
#         \\
#         \mathbf{g}_t &= \nabla_{\mathbf{x}_t} \log p( \widehat{\mathbf{x}}_{t}(\mathbf{x}_t) | \mathbf{y} ) \\
#         \mathbf{\varepsilon}_t &= \mathcal{N}(0, \mathbf{I}) \\
#         \mathbf{x}_{t-1} &= a_t \,\, \mathbf{x}_t
#         + b_t \, \, \widehat{\mathbf{x}}_t
#         + \tilde{\sigma}_t \, \, \mathbf{\varepsilon}_t + \mathbf{g}_t,
#         \end{aligned}
#         \end{equation*}
#
# where :math:`\denoiser{\cdot}{\sigma}` is a denoising network for noise level :math:`\sigma`,
# :math:`\eta` is a hyperparameter, and the constants :math:`\tilde{\sigma}_t, a_t, b_t` are defined as
#
# .. math::
#         \begin{equation*}
#         \begin{aligned}
#           \tilde{\sigma}_t &= \eta \sqrt{ (1 - \frac{\overline{\alpha}_t}{\overline{\alpha}_{t-1}})
#           \frac{1 - \overline{\alpha}_{t-1}}{1 - \overline{\alpha}_t}} \\
#           a_t &= \sqrt{1 - \overline{\alpha}_{t-1} - \tilde{\sigma}_t^2}/\sqrt{1-\overline{\alpha}_t} \\
#           b_t &= \sqrt{\overline{\alpha}_{t-1}} - \sqrt{1 - \overline{\alpha}_{t-1} - \tilde{\sigma}_t^2}
#           \frac{\sqrt{\overline{\alpha}_{t}}}{\sqrt{1 - \overline{\alpha}_{t}}}
#         \end{aligned}
#         \end{equation*}
#


# %%
# Denoising step
# ----------------------------
#
# The first step of DPS consists of applying a denoiser function to the current image :math:`\mathbf{x}_t`,
# with standard deviation :math:`\sigma_t = \sqrt{1 - \overline{\alpha}_t}/\sqrt{\overline{\alpha}_t}`.
#
# This is equivalent to sampling :math:`\mathbf{x}_t \sim q(\mathbf{x}_t|\mathbf{x}_0)`, and then computing the
# posterior mean.
#


t = torch.ones(1, device=device) * 500  # choose some arbitrary timestep
at = compute_alpha(betas, t.long())
sigmat = (1 - at).sqrt() / at.sqrt()

x0 = x_true
print("sigma_t = ", sigmat)
xt = x0 + sigmat * torch.randn_like(x0)

# apply denoiser
x0_t = model(xt, sigmat)

# Visualize
imgs = [x0, xt, x0_t]
plot(
    imgs,
    titles=["ground-truth", "noisy", "posterior mean"],
)

x0 = x_true * 2.0 - 1.0  # [0, 1] -> [-1, 1]


# %%
# ILVR Approximation using noisy_datafidelity

noisy_datafidelity = dinv.sampling.noisy_datafidelity.ILVRDataFidelity(
    physics=physics, denoiser=model
)
