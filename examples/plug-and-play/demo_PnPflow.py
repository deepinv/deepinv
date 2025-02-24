r"""
Flow matching based PnP: PnP-Flow
=================================

The example implements the PnP-Flow flow matching algorithm for image reconstruction from `Martin et al. (ICLR 2025) <https://arxiv.org/abs/2410.02423>`_.
The full algorithm is implemented in :class:`deepinv.optim.pnpflow.PnPFlow`.

PnP-Flow alternates between 1)
a gradient step on the data fidelity, 2) an interpolation step and  3)
a denoising step.
With a datafit term :math:`f`, the iterations are:

.. math::

    \begin{equation*}
    \begin{cases}
    z_k = x_{k-1} - \eta_k \nabla f(x_{k-1}) \\
    \tilde z_k = (1-t_k) z_k + t_k \varepsilon_k \\
    x_{k+1} = D_\theta(\tilde z_k, t_k)
    \end{cases}
    \end{equation*}

where

- :math:`\varepsilon_k \sim \mathcal N(0, \mathrm{Id})`
- :math:`t_k` is a sequence of timesteps with values in :math:`[0, 1]`, typically
  :math:`t_k = k / n_\mathrm{iter}`
- the denoiser :math:`D_\theta` builds upon the velocity field :math:`v_\theta` of a
  pretrained flow matching model:

.. math::

    D_\theta(x, t) = x + (1-t) v_\theta(x, t) \, .

"""

# %%
# We now implement PnP-FLow explicitly and apply it to the problems of
# inpainting and deblurring.

import torch
from tqdm import tqdm

import deepinv as dinv
from deepinv.utils.plotting import plot
from deepinv.optim.data_fidelity import L2
from deepinv.utils.plotting import config_matplotlib
from deepinv.utils.demo import load_url_image, get_image_url

from deepinv.optim.pnpflow import PnPFlow
from deepinv.models.flowunet import FlowUNet


config_matplotlib()
# Set the global random seed from pytorch to ensure
# the reproducibility of the example.
torch.manual_seed(0)
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# Load the Flow Matching generative model
#  --------------------------------------
# PnPFlow uses a denoiser which is built on the velocity field a Flow Matching
# model. In this example, we will use a Flow Matching model trained with
# minibatch OT on the CelebA dataset.

denoiser = FlowUNet(
    input_channels=3,
    input_height=128,
    pretrained="download",
    dataset_name="celeba",
    device=device,
)


# %%
# First, we consider the problem of mask inpainting. The forward operator is
# implemented in :class:`deepinv.physics.Inpainting`. The mask we use is
# a centered black square with size 40x40. We additionally use Additive
# White Gaussian Noise of standard deviation 25/255.

url = get_image_url("celeba_example.jpg")
x_true = load_url_image(url=url, img_size=128, resize_mode="resize", device=device)
# scale ground truth, as Flow Matching model was trained on images with pixel
# values in [-1, 1]:
x_true = 2 * x_true - 1
mask = torch.ones_like(x_true)
mask[:, :, 44:84, 44:84] = 0
sigma_noise = 25 / 255  # noise level

physics = dinv.physics.Inpainting(
    mask=mask,
    tensor_size=x_true.shape[1:],
    noise_model=dinv.physics.GaussianNoise(sigma=sigma_noise),
    device=device,
)
y = physics(x_true)

# %%
# Implementing Pnp-Flow
# ---------------------
# First we define the parameters of the algorithm: iteration number, and learning
# rate related hyperparameters `lr` and `lr_exp`.
max_iter = 100
lr = 1.0
lr_exp = 0.5
n_avg = 1
data_fidelity = L2()


def interpolation_step(x, t):
    """Interpolate between `x` and white gaussian noise."""
    tv = t.view(-1, 1, 1, 1)
    return tv * x + (1 - tv) * torch.randn_like(x)


def denoiser_step(x, t):
    """Denoiser based on velocity field of flow matching model."""
    return x + (1 - t.view(-1, 1, 1, 1)) * denoiser.forward_velocity(x, t)


x_hat = physics.A_adjoint(y)  # initialization

with torch.no_grad():
    for it in tqdm(range(max_iter)):
        t = torch.ones(len(x_hat), device=device) * it / max_iter
        lr_t = lr * (1 - t.view(-1, 1, 1, 1)) ** lr_exp
        z = x_hat - lr_t * data_fidelity.grad(x_hat, y, physics)
        x_new = torch.zeros_like(x_hat)
        for _ in range(n_avg):
            z_tilde = interpolation_step(z, t)
            x_new += denoiser_step(z_tilde, t)
        x_new /= n_avg
        x_hat = x_new
# %%
# Note: in some settings, the performance may be improved by using n_avg > 1,
# corresponding to multiple interpolation with different noise samples.
# For the gradient step `z = x_hat - lr_t * data_fidelity.grad(x_hat, y, physics)`,
# we have used an iteration-dependent learning rate.


# %% Plot results
imgs = [y, x_true, x_hat]
plot(
    imgs,
    titles=["measurement", "ground-truth", "reconstruction"],
    save_fn="celeba.png",
    save_dir=".",
)

psnr = dinv.metric.PSNR()
print(f"PSNR noisy image: {psnr(y, x_true).item():.2f}")
print(f"PSNR denoised image: {psnr(x_hat, x_true).item():.2f}")


# %%
# Deblurring example
# ------------------
# Next, we consider a deblurring problem, where the forward operator is implemented
# in :class:`deepinv.physics.BlurFFT`. In the example, we use Gaussian blur
# and we will additionally have Additive White Gaussian Noise (AWGN) of standard deviation  25/255.

url = get_image_url("celeba_example2.jpg")
x_true = load_url_image(url=url, img_size=128, resize_mode="resize", device=device)
x_true = 2 * x_true - 1
sigma_noise = 25 / 255.0

physics = dinv.physics.BlurFFT(
    img_size=x_true.shape[1:],
    filter=dinv.physics.blur.gaussian_blur(sigma=1.0),
    noise_model=dinv.physics.GaussianNoise(sigma=sigma_noise),
    device=device,
)
y = physics(x_true)

# %%
# We can directly use the :class:`deepinv.optim.pnpflow.PnPFlow` class:
pnpflow = PnPFlow(
    denoiser,
    data_fidelity=L2(),
    verbose=True,
    max_iter=100,
    device=device,
    lr=1.0,
    lr_exp=0.01,
)

x_hat = pnpflow(y, physics)

imgs = [y, x_true, x_hat]
plot(
    imgs,
    titles=["measurement", "ground-truth", "reconstruction"],
    save_fn="celeba2.png",
    save_dir=".",
)

print(f"PSNR noisy image: {psnr(y, x_true).item():.2f}")
print(f"PSNR denoised image: {psnr(x_hat, x_true).item():.2f}")
