r"""
Implementing DDNM
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

url = get_image_url("butterfly.png")

x_true = load_url_image(url=url, img_size=64).to(device)
x = x_true.clone()

# %%
# In this tutorial we consider random inpainting as the inverse problem, where the forward operator is implemented
# in :meth:`deepinv.physics.Inpainting`. In the example that we use, 90% of the pixels will be masked out randomly,
# and we will additionally have Additive White Gaussian Noise (AWGN) of standard deviation  12.75/255.

sigma = 12.75 / 255.0  # noise level

physics = dinv.physics.Inpainting(
    tensor_size=(3, x.shape[-2], x.shape[-1]),
    mask=0.8,
    pixelwise=True,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma),
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
# Denoising step
# ----------------------------
#
# The first step of DPS consists of applying a denoiser function to the current image :math:`\mathbf{x}_t`,
# with standard deviation :math:`\sigma_t = \sqrt{1 - \overline{\alpha}_t}/\sqrt{\overline{\alpha}_t}`.
#
# This is equivalent to sampling :math:`\mathbf{x}_t \sim q(\mathbf{x}_t|\mathbf{x}_0)`, and then computing the
# posterior mean.
#


t = torch.ones(1, device=device) * 50  # choose some arbitrary timestep
at = compute_alpha(betas, t.long())
sigmat = (1 - at).sqrt() / at.sqrt()

x0 = x_true
xt = x0 + sigmat * torch.randn_like(x0)

# apply denoiser
x0_t = model(xt, sigmat)

# Visualize
imgs = [x0, xt, x0_t]
plot(
    imgs,
    titles=["ground-truth", "noisy", "posterior mean"],
)

# %%
# DDNM Approximation using noisy_datafidelity

noisy_datafidelity = dinv.sampling.noisy_datafidelity.DDNMDataFidelity(
    physics=physics, denoiser=model
)

eta = 0.0

i = 100 # choose some arbitrary timestep
t = (torch.ones(1) * i).to(device)
at = compute_alpha(betas, t.long())
xt = at.sqrt() * x0 + (1 - at).sqrt() * torch.randn_like(x0)

sigma = (1 - at).sqrt() / at.sqrt()

x0_t = model(xt / 2 + 0.5, sigma / 2) * 2 - 1

#grad_ll = noisy_datafidelity.grad_simplified(xt, y, sigma, 1.0)



#Lambda_t is teh projection of Sigma_t is the spectral space : Sigma_t = V Lambda_t V_T see eq (36) of DDNM paper
Lambda_t = torch.ones_like(x)
inverse_singulars = 1. / physics.mask #Sigma_dagger == pseudo inverse see eq (37) of DDNM paper
inverse_singulars[physics.mask == 0] = 0 #pseudo inverse see eq (37) of DDNM paper

#define sigma_noise (sigma_y in DDNM paper)
if hasattr(physics.noise_model, "sigma"):
    sigma_noise = physics.noise_model.sigma
else:
    sigma_noise = 0.01


case = sigma < at.sqrt() * sigma_noise * inverse_singulars #see svd_operators.py Lambda from DDNM code l. 377 + eq (64) of paper 
Lambda_t[case] = physics.mask[case] * sigma.item() * (1 - eta ** 2) ** 0.5 / at.sqrt().item() / sigma_noise #see svd_operators.py Lambda from DDNM code l. 378 + eq (64) of paper 

grad_ll_full = noisy_datafidelity.grad(xt, y, sigma, Lambda_t)


imgs = [x0, xt, x0_t, grad_ll_full]
plot(
    imgs,
    titles=["groundtruth", "noisy", "posterior mean", "gradient"],
)

# %%
# DPS Algorithm using noisy_datafidelity

num_steps = 200

skip = num_train_timesteps // num_steps

batch_size = 1
eta = 1.0

seq = range(0, num_train_timesteps, skip)
seq_next = [-1] + list(seq[:-1])
time_pairs = list(zip(reversed(seq), reversed(seq_next)))

# measurement
x0 = x_true * 2.0 - 1.0
y = physics(x0.to(device))

# initial sample from x_T
x = torch.randn_like(x0)

xs = [x]
x0_preds = []

for i, j in tqdm(time_pairs):
    t = (torch.ones(batch_size) * i).to(device)
    next_t = (torch.ones(batch_size) * j).to(device)

    at = compute_alpha(betas, t.long())
    at_next = compute_alpha(betas, next_t.long())

    xt = xs[-1].to(device)
    norm_grad = noisy_datafidelity.grad(xt, y, sigma)
    x0_t = model(xt / 2 + 0.5, (1 - at).sqrt() / at.sqrt() / 2) * 2 - 1
    sigma_tilde = ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt() * eta
    c2 = ((1 - at_next) - sigma_tilde**2).sqrt()

    # 3. noise step
    epsilon = torch.randn_like(xt)

    # 4. DDPM(IM) step
    xt_next = (
        (at_next.sqrt() - c2 * at.sqrt() / (1 - at).sqrt()) * x0_t
        + sigma_tilde * epsilon
        + c2 * xt / (1 - at).sqrt()
        - norm_grad
    )

    x0_preds.append(x0_t.to("cpu"))
    xs.append(xt_next.to("cpu"))

recon = xs[-1]

# plot the results
x = recon / 2 + 0.5
imgs = [y, x, x_true]
plot(imgs, titles=["measurement", "model output", "groundtruth"])


# %%
# DPS Algorithm
# --------------
#
# As we visited all the key components of DPS, we are now ready to define the algorithm. For every denoising
# timestep, the algorithm iterates the following
#
# 1. Get :math:`\hat{\mathbf{x}}` using the denoiser network.
# 2. Compute :math:`\nabla_{\mathbf{x}_t} \log p(\mathbf{y}|\hat{\mathbf{x}}_t)` through backpropagation.
# 3. Perform reverse diffusion sampling with DDPM(IM), corresponding to an update with :math:`\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)`.
# 4. Take a gradient step with :math:`\nabla_{\mathbf{x}_t} \log p(\mathbf{y}|\hat{\mathbf{x}}_t)`.
#
# There are two caveats here. First, in the original work, DPS used DDPM ancestral sampling. As the `DDIM sampler
# <https://arxiv.org/abs/2010.02502)>`_ is a generalization of DDPM in a sense that it retrieves DDPM when
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
# With these in mind, let us solve the inverse problem with DPS!


# %%
# .. note::
#
#   We only use 200 steps to reduce the computational time of this example. As suggested by the authors of DPS, the
#   algorithm works best with ``num_steps = 1000``.
#

num_steps = 200

skip = num_train_timesteps // num_steps

batch_size = 1
eta = 1.0

seq = range(0, num_train_timesteps, skip)
seq_next = [-1] + list(seq[:-1])
time_pairs = list(zip(reversed(seq), reversed(seq_next)))

# measurement
x0 = x_true * 2.0 - 1.0
y = physics(x0.to(device))

# initial sample from x_T
x = torch.randn_like(x0)

xs = [x]
x0_preds = []

for i, j in tqdm(time_pairs):
    t = (torch.ones(batch_size) * i).to(device)
    next_t = (torch.ones(batch_size) * j).to(device)

    at = compute_alpha(betas, t.long())
    at_next = compute_alpha(betas, next_t.long())

    xt = xs[-1].to(device)

    with torch.enable_grad():
        xt.requires_grad_()

        # 1. denoising step
        # we call the denoiser using standard deviation instead of the time step.
        aux_x = xt / 2 + 0.5
        x0_t = 2 * model(aux_x, (1 - at).sqrt() / at.sqrt() / 2) - 1
        x0_t = torch.clip(x0_t, -1.0, 1.0)  # optional

        # 2. likelihood gradient approximation
        l2_loss = data_fidelity(x0_t, y, physics).sqrt().sum()

    norm_grad = torch.autograd.grad(outputs=l2_loss, inputs=xt)[0]
    norm_grad = norm_grad.detach()

    sigma_tilde = ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt() * eta
    c2 = ((1 - at_next) - sigma_tilde**2).sqrt()

    # 3. noise step
    epsilon = torch.randn_like(xt)

    # 4. DDPM(IM) step
    xt_next = (
        (at_next.sqrt() - c2 * at.sqrt() / (1 - at).sqrt()) * x0_t
        + sigma_tilde * epsilon
        + c2 * xt / (1 - at).sqrt()
        - norm_grad
    )

    x0_preds.append(x0_t.to("cpu"))
    xs.append(xt_next.to("cpu"))

recon = xs[-1]

# plot the results
x = recon / 2 + 0.5
imgs = [y, x, x_true]
plot(imgs, titles=["measurement", "model output", "groundtruth"])


# %%
# Using DPS in your inverse problem
# ---------------------------------------------
# You can readily use this algorithm via the :meth:`deepinv.sampling.DPS` class.
#
# ::
#
#       y = physics(x)
#       model = dinv.sampling.DPS(dinv.models.DiffUNet(), data_fidelity=dinv.optim.L2())
#       xhat = model(y, physics)
#
