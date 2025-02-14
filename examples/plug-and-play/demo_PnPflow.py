r"""
Implementing PnP-Flow
=====================

The example implements the PnP-Flow flow matching algorithm for image reconstruction from `Martin et al. (ICLR 2025) <https://arxiv.org/abs/2410.02423>`_. 
 
The full algorithm is implemented in :class:`deepinv.optim.pnpflow.PnPFlow`.

PnPFlow iterates on timesteps :math:`t \in [0,1]`. It alternates between 1) 
a gradient step on the data fidelity, 2) an interpolation step and  3) 
a denoising step.

With a datafit term :math:`f`, the iterations are:

.. math::

    \begin{equation*}
    \begin{aligned}
    &x_t = x_{t-1} - \eta \nabla f(x_{t-1}) \\
    &z_t = (1-t) x_t + t \varepsilon_t \\
    &x_{t+1} = D_\theta(z_t, t)
    \end{aligned}
    \end{equation*}

where :math:`\varepsilon_t \sim \mathcal N(0, \mathrm{Id})`, and the denoiser
:math:`D_\theta` builds upon the velocity field :math:`v_\theta` of a 
pretrained flow matching model:

.. math::

    D_\theta(x, t) = x + (1-t) v_\theta(x,t),

"""

# %%
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import deepinv as dinv
from deepinv.utils.plotting import plot
from deepinv.optim.data_fidelity import L2
from deepinv.utils.plotting import config_matplotlib
from deepinv.utils.demo import load_url_image, get_image_url

from deepinv.models.flowunet import FlowUNet
from deepinv.optim.pnpflow import PnPFlow


config_matplotlib()


# Set the global random seed from pytorch to ensure
# the reproducibility of the example.
torch.manual_seed(0)
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# Load the Flow Matching generative model
#  --------------------------------------
# PnPFlow uses a denoiser which is built on a flow matching velocity.
# In this example, we will use a Flow Matching model trained with minibatch OT
# on the CelebA dataset.

velocity = FlowUNet(input_channels=3, input_height=128,
                    pretrained="download", device=device)


# %%
# First, we consider the problem of mask inpainting. The forward operator is
# # implemented in :class:`deepinv.physics.Inpainting`. The mask we use is
# a centered black square with size 60x60, and We additionally use Additive
# White Gaussian Noise of standard deviation  12.75/255.

url = get_image_url("celeba_example.jpg")
x_true = load_url_image(url=url, img_size=128,
                        resize_mode="resize", device=device)
x = 2 * x_true.clone() - 1  # values in [-1, 1]
mask = torch.ones_like(x)
mask[:, :, 32:96, 32:96] = 0
sigma_noise = 12.5 / 255.0  # noise level

physics = dinv.physics.Inpainting(
    mask=mask,
    tensor_size=x.shape[1:],
    noise_model=dinv.physics.GaussianNoise(sigma=sigma_noise),
    device=device,
)
y = physics(x)

# %%
# Run the PnP method
# ------------------
max_iter = 20
delta = 1 / max_iter
lr = 1.0
lr_exp = 0.6
n_avg = 1
data_fidelity = L2()
x_hat = physics.A_adjoint(y)


def interpolation_step(x, t):
    tv = t.view(-1, 1, 1, 1)
    return tv * x + (1 - tv) * torch.randn_like(x)


def denoiser(x, t):
    return x + (1 - t.view(-1, 1, 1, 1)) * velocity(x, t)


with torch.no_grad():
    for it in tqdm(range(max_iter)):
        t = torch.ones(len(x), device=device) * delta * it
        lr_t = lr * (1 - t.view(-1, 1, 1, 1)) ** lr_exp
        z = x - lr_t * data_fidelity.grad(x, y, physics)
        x_new = torch.zeros_like(x)
        for _ in range(n_avg):
            z_tilde = interpolation_step(x, t)
            x_new += denoiser(z_tilde, t)
        x_new /= n_avg
        x = x_new


# %% Plot results
imgs = [y, x_true, x_hat]
plot(
    imgs,
    titles=["measurement", "ground-truth", "reconstruction"],
    save_fn="res_inpainting.png",
    save_dir=".",
)

print("PSNR noisy image :", dinv.metric.PSNR()((y + 1.0) * 0.5, x_true).item())
print("PSNR restored image :", dinv.metric.PSNR()
      ((x_hat + 1.0) * 0.5, x_true).item())

# %%
# Next, we consider a deblurring problme, where the forward operator is implemented
# in :class:`deepinv.physics.BlurFFT`. In the example, we use Gaussian blur
# and we will additionally have Additive White Gaussian Noise (AWGN) of standard deviation  12.75/255.

print("Running deblurring example")
pnpflow = PnPFlow(
    velocity,
    data_fidelity=L2(),
    verbose=True,
    max_iter=100,
    device=device,
    lr=1.0,
    lr_exp=0.01,
)

url = get_image_url("celeba_example2.jpg")
x_true = load_url_image(url=url, img_size=128,
                        resize_mode="resize", device=device)
x = x_true.clone()
sigma_noise = 12.75 / 255.0  # noise level

physics = dinv.physics.BlurFFT(
    img_size=x.shape[1:],
    filter=dinv.physics.blur.gaussian_blur(sigma=1.0),
    noise_model=dinv.physics.GaussianNoise(sigma=sigma_noise),
    device=device,
)
y = physics(2 * x - 1)

x_hat = pnpflow(y, physics)

imgs = [y, x_true, (x_hat + 1.0) * 0.5]
plot(
    imgs,
    titles=["measurement", "ground-truth", "reconstruction"],
    save_fn="res_blurfft.png",
    save_dir=".",
)

print("PSNR noisy image :", dinv.metric.PSNR()((y + 1.0) * 0.5, x_true).item())
print("PSNR restored image :", dinv.metric.PSNR()
      ((x_hat + 1.0) * 0.5, x_true).item())


##
# To run the algorithm directly, one can do:
pnpflow = PnPFlow(
    velocity,
    data_fidelity=L2(),
    verbose=True,
    max_iter=100,
    device=device,
    lr=1.0,
    lr_exp=0.6,
)
pnpflow(y, physics)
