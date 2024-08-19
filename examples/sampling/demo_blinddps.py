r"""
Implementing BlindDPS
====================

In this tutorial, we will go over the different steps in the Blind Diffusion posterior Sampling (BlindDPS)
algorithm introduced in `Chung et al. <https://arxiv.org/abs/2211.10656>`_ The full algorithm is implemented 
in :meth:`deepinv.sampling.BlindDPS`.
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
# In this tutorial we consider Gaussian deblurring as the inverse problem, where the forward operator is implemented
# in :meth:`deepinv.physics.Blur`. The kernel is generated using :meth:`deepinv.physics.blur.gaussian_blur``.
# In this example we also add a small amount of Gaussian noise to the image.

blur_std = 3.0
sigma = 5.1 / 255.0

k_true = dinv.physics.blur.gaussian_blur(sigma=blur_std, angle=0.0).to(device)
k_true = torch.nn.functional.pad(k_true, (21, 20, 21, 20))

physics = dinv.physics.Blur(
    filter=k_true,
    padding="reflect",
    noise_model=dinv.physics.GaussianNoise(sigma=sigma),
    device=device,
)

y = physics(x_true)

imgs = [y, x_true, k_true]
plot(
    imgs,
    titles=["measurement", "groundtruth", "kernel"],
)


# %%
# Diffusion model loading
# ----------------------------
#
# For the image prior will take a pre-trained diffusion model, trained on the FFHQ 256x256 dataset.
# Note that this means that the model was trained with human face images,
# which is very different from the image that we consider in our example.
# For the kernel prior, we take the model provided by `Chung et al. <https://arxiv.org/abs/2209.14687>`_
# which is trained on kernels of size 64x64 with both Gaussian and motion blur.

model_x = dinv.models.DiffUNet(in_channels=3, out_channels=3).to(device)
model_k = dinv.models.DiffUNet(in_channels=1, out_channels=1).to(device)

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
#           \mathbf{k}_t = \sqrt{\beta_t}\mathbf{k}_{t-1} + \sqrt{1 - \beta_t}\mathbf{\epsilon}
#
#           \mathbf{k}_t = \sqrt{\bar\alpha_t}\mathbf{k}_0 + \sqrt{1 - \bar\alpha_t}\mathbf{\epsilon}
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
# The BlindDPS algorithm
# ---------------------
#
# Now that the inverse problem is defined, we can apply the BlindDPS algorithm to solve it.
# Similar to DPS, the BlindDPS algorithm is a diffusion algorithm that alternates between a denoising step,
# a gradient step and a reverse diffusion step.
# The key difference is that the denoised kernel produced by the diffusion model is normalized to have unit norm.
# The authors also add a regularization term to the likelihood gradient step of the kernel to promote sparsity.
# The algorithm writes as follows, for :math:`t` decreasing from :math:`T` to :math:`1`:
#
# .. math::
#             \begin{equation*}
#             \begin{aligned}
#             \widehat{\mathbf{x}}_{t} &= D_{\theta_x}(\mathbf{x}_t, \sqrt{1-\overline{\alpha}_t}/\sqrt{\overline{\alpha}_t})
#             \\
#             \widehat{\mathbf{k}}_{t} &= D_{\theta_k}(\mathbf{k}_t, \sqrt{1-\overline{\alpha}_t}/\sqrt{\overline{\alpha}_t})
#             \\
#             \mathbf{g}_t^x &= \nabla_{\mathbf{x}_t} \log p( \widehat{\mathbf{x}}_{t}(\mathbf{x}_t) | \mathbf{y} ) \\
#             \mathbf{g}_t^k &= \nabla_{\mathbf{k}_t} \log p( \widehat{\mathbf{k}}_{t}(\mathbf{k}_t) | \mathbf{y} ) \\
#             \mathbf{\varepsilon}_t^x &= \mathcal{N}(0, \mathbf{I}) \\
#             \mathbf{\varepsilon}_t^k &= \mathcal{N}(0, \mathbf{I}) \\
#             \mathbf{x}_{t-1} &= a_t \,\, \mathbf{x}_t + b_t \, \, \widehat{\mathbf{x}}_t + \tilde{\sigma}_t \, \, \mathbf{\varepsilon}_t^x + \, \mathbf{g}_t^x, \\
#             \mathbf{k}_{t-1} &= a_t \,\, \mathbf{k}_t + b_t \, \, \widehat{\mathbf{k}}_t + \tilde{\sigma}_t \, \, \mathbf{\varepsilon}_t^k + \, \mathbf{g}_t^k,
#             \end{aligned}
#             \end{equation*}
#
#     where :math:`D_{\theta_x}(\cdot)` is an image denoising network for noise level :math:`\sigma`,
#     :math:`D_{\theta_k}(\cdot)` is a kernel denoising network for noise level :math:`\sigma`
#     :math:`\eta` is a hyperparameter, and the constants :math:`\tilde{\sigma}_t, a_t, b_t` are defined as
#
# .. math::
#             \begin{equation*}
#             \begin{aligned}
#               \tilde{\sigma}_t &= \eta \sqrt{ (1 - \frac{\overline{\alpha}_t}{\overline{\alpha}_{t-1}})
#               \frac{1 - \overline{\alpha}_{t-1}}{1 - \overline{\alpha}_t}} \\
#               a_t &= \sqrt{1 - \overline{\alpha}_{t-1} - \tilde{\sigma}_t^2}/\sqrt{1-\overline{\alpha}_t} \\
#               b_t &= \sqrt{\overline{\alpha}_{t-1}} - \sqrt{1 - \overline{\alpha}_{t-1} - \tilde{\sigma}_t^2}
#               \frac{\sqrt{\overline{\alpha}_{t}}}{\sqrt{1 - \overline{\alpha}_{t}}}.
#             \end{aligned}
#             \end{equation*}
#


# %%
# Denoising step
# ----------------------------
#
# The first step of BlindDPS consists of applying a denoiser function to the current image :math:`\mathbf{x}_t`
# and kernel :math:`\mathbf{k}_t`, with standard deviation :math:`\sigma_t = \sqrt{1 - \overline{\alpha}_t}/\sqrt{\overline{\alpha}_t}`.
# This is equivalent to sampling :math:`\mathbf{x}_t \sim q(\mathbf{x}_t|\mathbf{x}_0)`, and then computing the
# posterior mean. The same holds for the kernel.
#


t = torch.ones(1, device=device) * 50  # choose some arbitrary timestep
at = compute_alpha(betas, t.long())
sigmat = (1 - at).sqrt() / at.sqrt()

x0 = x_true
xt = x0 + sigmat * torch.randn_like(x0)

k0 = k_true
kt = k0 + sigmat * torch.randn_like(k0)

# apply denoisers
x0_t = model_x(xt, sigmat)
k0_t = model_k(kt, sigmat)

# Visualize
imgs = [x0, xt, x0_t]
plot(
    imgs,
    titles=["ground-truth", "noisy", "posterior mean"],
)
kernels = [k0, kt, k0_t]
plot(
    kernels,
    titles=["ground-truth", "noisy", "posterior mean"],
)

# %%
# DPS approximation
# ----------------------------
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
# is intractable, DPS proposes the following approximation (for details, see Theorem 1 of
# `Chung et al. <https://arxiv.org/abs/2209.14687>`_
#
# .. math::
#
#           \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t|\mathbf{y}) \approx \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)
#           + \nabla_{\mathbf{x}_t} \log p(\mathbf{y}|\widehat{\mathbf{x}}_{t})
#
# Remarkably, we can now compute the latter term when we have Gaussian noise, as
#
# .. math::
#
#       \log p(\mathbf{y}|\hat{\mathbf{x}}_{t}) =
#       -\frac{\|\mathbf{y} - A\widehat{\mathbf{x}}_{t}\|_2^2}{2\sigma_y^2}.
#
# Moreover, taking the gradient w.r.t. :math:`\mathbf{x}_t` can be performed through automatic differentiation.
# Let's see how this can be done in PyTorch. Note that when we are taking the gradient w.r.t. a tensor,
# we first have to enable the gradient computation by ``tensor.requires_grad_()``. Since we will backpropagate
# the gradient twice, once with respect to the image and once with respect to the kernel,
# we have to retain the compute graph with ``retain_graph=True``.
#
# .. note::
#           The DPS algorithm assumes that the images are in the range [-1, 1], whereas standard denoisers
#           usually output images in the range [0, 1]. This is why we rescale the images before applying the steps.
from deepinv.physics.functional.convolution import conv2d

k0 = k_true / (k_true.max() - k_true.min())  # sum = 1 -> [0, 1]
k0 = k0 * 2.0 - 1.0  # [0, 1] -> [-1, 1]
x0 = x_true * 2.0 - 1.0  # [0, 1] -> [-1, 1]

data_fidelity = L2()

i = 200  # choose some arbitrary timestep
t = (torch.ones(1) * i).to(device)
at = compute_alpha(betas, t.long())
kt = at.sqrt() * k0 + (1 - at).sqrt() * torch.randn_like(k0)
xt = at.sqrt() * x0 + (1 - at).sqrt() * torch.randn_like(x0)

regularization_k = dinv.optim.L1Prior()

with torch.enable_grad():
    # Turn on gradient
    kt.requires_grad_()
    xt.requires_grad_()

    # normalize to [0,1], denoise, and rescale to [-1, 1]
    k0_t = model_k(kt / 2 + 0.5, (1 - at).sqrt() / at.sqrt() / 2) * 2 - 1
    x0_t = model_x(xt / 2 + 0.5, (1 - at).sqrt() / at.sqrt() / 2) * 2 - 1

    # Apply the normalized kernel to the image
    k0_t = (k0_t + 1.0) / 2.0
    k0_t_norm = k0_t / k0_t.sum()
    y0_t = conv2d(x0_t, k0_t_norm, padding="reflect")

    # Log-likelihood
    ll_x = torch.linalg.norm(y0_t - y)
    ll_k = torch.linalg.norm(y0_t - y) + 0.1 * torch.linalg.vector_norm(
        k0_t_norm.flatten(), ord=0
    )

# Take gradient w.r.t. xt
grad_ll = torch.autograd.grad(outputs=ll_k, inputs=kt, retain_graph=True)[0]
grad_ll_x = torch.autograd.grad(outputs=ll_x, inputs=xt)[0]

plot(
    [x_true, xt, x0_t, grad_ll_x],
    titles=["groundtruth", "noisy", "denoised", "gradient"],
)
plot(
    [k_true, kt, k0_t, grad_ll], titles=["groundtruth", "noisy", "denoised", "gradient"]
)

# %%
# BlindDPS Algorithm
# --------------
#
# Now let's assemble all the components into the final algorithm.
# For every denoising timestep, the algorithm iterates the following
#
# 1. Get :math:`\hat{\mathbf{x}}` and :math:`\hat{\mathbf{k}}` using each of the denoiser networks.
# 2. Compute :math:`\nabla_{\mathbf{x}_t} \log p(\mathbf{y}|\hat{\mathbf{x}}_t)` and :math:`\nabla_{\mathbf{k}_t} \log p(\mathbf{y}|\hat{\mathbf{k}}_t)` through backpropagation.
# 3. Perform reverse diffusion sampling with DDPM(IM), corresponding to an update with :math:`\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)` and :math:`\nabla_{\mathbf{k}_t} \log p(\mathbf{k}_t)`.
# 4. Take a gradient step with :math:`\nabla_{\mathbf{x}_t} \log p(\mathbf{y}|\hat{\mathbf{x}}_t)` and :math:`\nabla_{\mathbf{k}_t} \log p(\mathbf{y}|\hat{\mathbf{k}}_t)`.
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
#           \rho \nabla_{\mathbf{x}_t} \|\mathbf{y} - \mathbf{A}\hat{\mathbf{x}}_{t}\|^2_2
#
# With these in mind, let us solve the inverse problem with DPS!


num_steps = 1000

skip = num_train_timesteps // num_steps

batch_size = 1
eta = 1.0

seq = range(0, num_train_timesteps, skip)
seq_next = [-1] + list(seq[:-1])
time_pairs = list(zip(reversed(seq), reversed(seq_next)))

# measurement
x0 = x_true * 2.0 - 1.0
y = physics(x0.to(device))

# initial sample from x_T and k_T
x = torch.randn_like(x0)
k = torch.randn_like(k_true)

xs = [x]
ks = [k]
x0_preds = []
k0_preds = []

# Parameters reported in the paper
reg_factor = 1.0
grad_factor = 0.3

for i, j in tqdm(time_pairs):
    t = (torch.ones(batch_size) * i).to(device)
    next_t = (torch.ones(batch_size) * j).to(device)

    at = compute_alpha(betas, t.long())
    at_next = compute_alpha(betas, next_t.long())

    xt = xs[-1].to(device)
    kt = ks[-1].to(device)

    with torch.enable_grad():
        xt.requires_grad_()
        kt.requires_grad_()

        # 1. denoising step
        # we call the denoiser using standard deviation instead of the time step.

        # Image
        aux_x = xt / 2 + 0.5
        x0_t = 2 * model_x(aux_x, (1 - at).sqrt() / at.sqrt() / 2) - 1
        # x0_t = torch.clip(x0_t, -1.0, 1.0)  # optional

        # Kernel
        aux_k = kt / 2 + 0.5
        k0_t = 2 * model_k(aux_k, (1 - at).sqrt() / at.sqrt() / 2) - 1
        # k0_t = torch.clip(k0_t, -1.0, 1.0) # optional

        # This mean kernel estimate is for the DDPM step
        k0_t = (k0_t + 1.0) / 2.0
        # This one is for the gradient step
        k0_t_norm = k0_t / k0_t.sum()

        # Here need to redefine the physics model otherwise the compute graph is broken
        physics_cur = Blur(
            filter=torch.ones_like(k0_t_norm),
            padding="circular",
            noise_model=dinv.physics.GaussianNoise(sigma=0.0),
            device=device,
        )

        y0_t = physics_cur.A(x0_t, filter=k0_t_norm)

        # 2. likelihood gradient approximation
        ll_x = torch.linalg.norm(y0_t - y)
        ll_k = torch.linalg.norm(y0_t - y) + 0.1 * torch.linalg.vector_norm(
            k0_t_norm.flatten(), ord=0
        )

    norm_grad_x = torch.autograd.grad(outputs=ll_x, inputs=xt, retain_graph=True)[0]
    norm_grad_x = norm_grad_x.detach()

    norm_grad_k = torch.autograd.grad(outputs=ll_k, inputs=kt)[0]
    norm_grad_k = norm_grad_k.detach()

    sigma_tilde = ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt() * eta
    c2 = ((1 - at_next) - sigma_tilde**2).sqrt()

    # 3. noise step
    epsilon_x = torch.randn_like(xt)
    espilon_k = torch.randn_like(kt)

    # 4. DDPM(IM) step
    xt_next = (
        (at_next.sqrt() - c2 * at.sqrt() / (1 - at).sqrt()) * x0_t
        + sigma_tilde * epsilon_x
        + c2 * xt / (1 - at).sqrt()
        - grad_factor * norm_grad_x
    )

    kt_next = (
        (at_next.sqrt() - c2 * at.sqrt() / (1 - at).sqrt()) * (2 * k0_t - 1)
        + sigma_tilde * espilon_k
        + c2 * kt / (1 - at).sqrt()
        - grad_factor * norm_grad_k
    )

    x0_preds.append(x0_t.detach().to("cpu"))
    k0_preds.append(k0_t.detach().to("cpu"))
    xs.append(xt_next.detach().to("cpu"))
    ks.append(kt_next.detach().to("cpu"))

    del ll_x, ll_k
    torch.cuda.empty_cache()

recon_x = xs[-1]
recon_k = ks[-1]

# plot the results
x = recon_x / 2 + 0.5
k = recon_k / 2 + 0.5
k = k / k.sum()

plot([y, x, x_true], titles=["measurement", "model output", "groundtruth"])
plot([recon_k, k_true], titles=["model output", "groundtruth"], figsize=(12, 3))


# %%
# Using BlindDPS in your inverse problem
# ---------------------------------------------
# You can readily use this algorithm via the :meth:`deepinv.sampling.BlindDPS` class.
#
# ::
#
#       y = physics(x)
#       model = dinv.sampling.BlindDPS(dinv.models.DiffUNet(),
#                                      dinv.models.DiffUNet(in_channels=1, out_channels=1),
#                                      data_fidelity=dinv.optim.L2())
#       xhat, khat = model(y, physics)
#
