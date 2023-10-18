r"""
Implementing DPS
====================

In this tutorial, we will go over the steps in the Diffusion Posterior Sampling (DPS) algorithm introduced in
[`Chung et al. <https://arxiv.org/abs/2209.14687>`_]. The full algorithm is implemented in
:class:`deepinv.sampling.diffusion.DPS`.
"""

# %% Installing dependencies
# -----------------------------
# Let us ``import`` the relevant packages, and load a sample
# image of size 256x256. This will be used as our ground truth image.

import numpy as np
import torch

import deepinv as dinv
from deepinv.utils.plotting import plot
from deepinv.optim.data_fidelity import L2
from deepinv.utils.demo import load_url_image
import tqdm  # to visualize progress

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

url = (
    "https://mycore.core-cloud.net/index.php/s/9EzDqcJxQUJKYul/"
    "download?path=%2Fdatasets&files=butterfly.png"
)
x_true = load_url_image(url=url, img_size=256).to(device)
x = x_true.clone()

# %%
# In this tutorial we consider random inpainting as the inverse problem, where the forward operator is implemented
# in :meth:`dinv.physics.Inpainting`. In the example that we use, 90% of the pixels will be masked out randomly,
# and we will additionally have Additive White Gaussian Noise (AWGN) of standard deviation  12.75/255.

sigma = 12.75 / 255.0  # noise level

physics = dinv.physics.Inpainting(
    tensor_size=(3, x.shape[-2], x.shape[-1]),
    mask=0.1,
    pixelwise=True,
    device=device,
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


model = dinv.models.DiffUNet(image_size=256, large_model=True).to(device)

# %%
# Define diffusion schedule
# ----------------------------
#
# We will use the standard linear diffusion noise schedule. Once $\beta_t$ is defined to follow a linear schedule
# that interpolates between :math:`\beta_{\rm min}` and :math:`\beta_{\rm max}`,
# we have the following additional definitions:
# :math:`\alpha_t := 1 - \beta_t`, :math:`\bar\alpha_t := \prod_{i=1}^t \alpha_t`.
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


# %%
# Tweedie's formula
# ----------------------------
#
# Given a noisy image :math:`\mathbf{x}_t`, Tweedie's formula lets us compute the posterior mean
# :math:`\hat{\mathbf{x}}_{0|t} := \mathbb{E}[\mathbf{x}_0|\mathbf{x}_t]$
# when we have access to the __score function__ :math:`\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)`.
# In the context of variance-preserving (VP) diffusion, the formula reads
#
# .. math::
#          \hat{\mathbf{x}}_{0|t} = \frac{1}{\sqrt{\bar\alpha_t}}
#          (\mathbf{x}_t + (1 - \bar\alpha(t))\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t))
#
# Recall that we can approximate the true score function with our network. Hence, we can use the following to compute
# the posterior mean
#
# .. math::
#           \hat{\mathbf{x}}_{0|t} = \frac{1}{\sqrt{\bar\alpha_t}}(\mathbf{x}_t - \sqrt{1 - \bar\alpha(t)} \mathbf{
#           \epsilon}_\theta(\mathbf{x}_t))
#
# Let us see this in effect by sampling :math:`\mathbf{x}_t \sim q(\mathbf{x}_t|\mathbf{x}_0)`, and then computing the
# posterior mean.

# Utility function to let us easily retrieve \bar\alpha_t
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


x0 = x_true
xt = x0 + .1 * torch.randn_like(x0)

sigma_t = .1
# apply denoiser
x0_t = model(xt, sigma_t)

# Visualize
imgs = [x0, xt, x0_t]
plot(
    imgs,
    titles=["ground-truth", "noisy", "posterior mean"],
)

# %%
# DPS approximation
# ----------------------------
#
# In order to perform gradient-based __posterior sampling__ with diffusion models, we have to be able to compute
# :math:`\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t|\mathbf{y})`. Applying Bayes rule, we have
#
# .. math::
#
#           \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t|\mathbf{y}) = \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)
#           + \nabla_{\mathbf{x}_t} \log p(\mathbf{y}|\mathbf{x}_t)
#
# For the former term, we can simply plug-in our estimated score function as in Tweedie's formula. As the latter term
# is intractable, DPS proposes the following approximation (for details, see Theorem 1 of
# [`Chung et al. <https://arxiv.org/abs/2209.14687>`_]
#
# .. math::
#
#           \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t|\mathbf{y}) = \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)
#           + \nabla_{\mathbf{x}_t} \log p(\mathbf{y}|\hat{\mathbf{x}}_{0|t})$
#
# Remarkably, we can now compute the latter term when we have Gaussian noise, as
#
# .. math::
#
#       \log p(\mathbf{y}|\hat{\mathbf{x}}_{0|t}) =
#       -\frac{\|\mathbf{y} - A\hat{\mathbf{x}}_{0|t}\|_2^2}{2\sigma_y^2}.
#
# Moreover, taking the gradient w.r.t. :math:`\mathbf{x}_t` can be performed through automatic differentiation.
# Let's see how this can be done in PyTorch. Note that when we are taking the gradient w.r.t. a tensor,
# we first have to enable the gradient computation by `tensor.requires_grad_()`


# [0, 1] -> [-1, 1]
x0 = x_true * 2. - 1.
x0 = x0.view(1, 3, 256, 256).to(device)

data_fidelity = L2()

# xt ~ q(xt|x0)
i = 200  # noise level (discretized)
t = (torch.ones(1) * i).to(device)
at = compute_alpha(betas, t.long())
xt = at.sqrt() * x0 + (1 - at).sqrt() * torch.randn_like(x0)

# DPS
with torch.enable_grad():
    # Turn on gradient
    xt.requires_grad_()
    et = model(xt, t, type_t='timestep')
    et = et[:, :3]
    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
    # Log-likelihood
    ll = data_fidelity(x0_t, y, physics)
    # Take gradient w.r.t. xt
    grad_ll = torch.autograd.grad(outputs=ll, inputs=xt)[0]

# Visualize
imgs = [x0, xt, x0_t, grad_ll]
plot(
    imgs,
    titles=["groundtruth", "noisy", "posterior mean", "gradient"],
)

# %%
# DPS Algorithm
# --------------
#
# As we visited all the key components of DPS, we are now ready to define the algorithm. For every denoising
# timestep, the algorithm iterates the following
#
# 1. Get :math:`\hat{\mathbf{x}}_{0|t}` from Tweedie's formula.
# 2. Compute :math:`\nabla_{\mathbf{x}_t} \log p(\mathbf{y}|\mathbf{x}_t)` through backpropagation.
# 3. Perform reverse diffusion sampling with DDPM(IM), corresponding to an
# update with :math:`\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)`.
# 4. Take a gradient step with $\nabla_{\mathbf{x}_t} \log p(\mathbf{y}|\mathbf{x}_t)` computed from step 2.
#
# There are two caveats here. First, in the original work, DPS used DDPM ancestral sampling. As the [DDIM sampler](
# https://arxiv.org/abs/2010.02502) is a generalization of DDPM in a sense that it retrieves DDPM when
# :math:`\eta = 1.0`, here we consider DDIM sampling.
# One can freely choose the $\eta$ parameter here, but since we will consider 1000
# NFEs, it is advisable to keep it :math:`\eta = 1.0`. Second, when taking the log-likelihood gradient step in 4,
# the gradient is weighted so that the actual implementation is a static step size times the l2 norm of the residual.
#
# .. math::
#
#           \nabla_{\mathbf{x}_t} \log p(\mathbf{y}|\hat{\mathbf{x}}_{0|t}) \simeq
#           \rho \|\mathbf{y} - \mathbf{A}\hat{\mathbf{x}}_{0|t}\|_2
#
# With these in mind, let us solve the inverse problem with DPS!


# %%
# .. note::
#
#   We only use 30 steps to reduce the computational time of this example. As suggested by the authors of DPS, the
#   algorithm works best with ``num_steps = 1000``.
#

num_steps = 1000

skip = num_train_timesteps // num_steps

batch_size = 1
eta = 1.0

seq = range(0, num_train_timesteps, skip)
seq_next = [-1] + list(seq[:-1])
time_pairs = list(zip(reversed(seq), reversed(seq_next)))

# measurement
x0 = x_true * 2. - 1.
y = physics(x0.to(device))

# initial sample from x_T
x = torch.randn(
    y.shape[0],
    3,
    256,
    256,
    device=device,
)

xs = [x]
x0_preds = []

for i, j in tqdm.tqdm(time_pairs):
    t = (torch.ones(batch_size) * i).to(device)
    next_t = (torch.ones(batch_size) * j).to(device)

    at = compute_alpha(betas, t.long())
    at_next = compute_alpha(betas, next_t.long())

    c1 = ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt() * eta
    c2 = ((1 - at_next) - c1 ** 2).sqrt()

    xt = xs[-1].to(device)

    # 1. NFE
    with torch.enable_grad():
        xt.requires_grad_()

        # we call the denoiser using standard deviation instead of the time step.
        aux_x = xt / 2 + 0.5
        x0_t = 2 * model(aux_x, (1 - at).sqrt() / at.sqrt() / 2) - 1
        x0_t = torch.clip(x0_t, -1., 1.)  # optional

        # 3. DPS
        l2_loss = data_fidelity(x0_t, y, physics).sqrt().sum()

    # Tweedie
    et = (xt - at.sqrt() * x0_t) / (1 - at).sqrt()

    norm_grad = torch.autograd.grad(outputs=l2_loss, inputs=xt)[0]
    norm_grad = norm_grad.detach()

    # 4. DDPM(IM) step
    xt_next = (
            at_next.sqrt() * x0_t
            + c1 * torch.randn_like(x0_t)
            + c2 * et.detach()
            - norm_grad
    )

    # 5. clear out memory
    del et
    x0_preds.append(x0_t.to('cpu'))
    xs.append(xt_next.to('cpu'))

recon = xs[-1]

# plot the results
x = recon / 2 + 0.5
imgs = [y, x, x_true]
plot(
    imgs,
    titles=["measurement", "model output", "groundtruth"]
)
