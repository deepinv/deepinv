r"""
Implementing PnP-Flow
=====================

In this tutorial, we revisit the implementation of the PnP-Flow flow matching
algorithm for image reconstruction from
`Martin et al. (ICLR 2025) <https://arxiv.org/abs/2410.02423>`_. The full 
algorithm is implemented in :class:`deepinv.optim.pnpflow.PnPFlow`.
"""


import numpy as np
import torch
import matplotlib.pyplot as plt
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
# Load the Flow Matching generative model.
#  --------------------------------------------------------------------------------
# This generative model is an Optimal Transport Flow Matching model trained on the CelebA dataset.

mynet = FlowUNet(input_channels=3, input_height=128,
                 pretrained="download", device=device)

# %%
# Set up the PnP Flow algorithm to solve inverse problems.
#  --------------------------------------------------------------------------------
# This method iterates on timesteps :math:`t \in [0,1]`.
# It alternates between a gradient step on the data fidelity, an interpolation step and a denoising step.
# The interpolation step at time :math:`t` writes as
#
# .. math::
#
#           z_t = (1-t) x_t + t \epsilon
#
# where :math:`\epsilon \sim \mathcal N(0, \mathrm{Id})`.
#
# The denoiser is defined as
#
# .. math::
#
#     D_\theta(x, t) = x + (1-t) v_\theta(x,t),
#
# where :math:`v_\theta` is the velocity field given by the pretrained flow mathching model.


### INPAINTING EXAMPLE ###
pnpflow = PnPFlow(
    mynet,
    data_fidelity=L2(),
    verbose=True,
    max_iter=100,
    device=device,
    lr=1.0,
    lr_exp=0.6,
)

# %%
# In this tutorial we first consider mask inpainting as the inverse problem, where the forward operator is implemented
# in :class:`deepinv.physics.Inpainting`. In the example that we use, the mask is a centered black square with size 60x60,
# and we will additionally have Additive White Gaussian Noise (AWGN) of standard deviation  12.75/255.

print("Running inpainting example")


url = get_image_url("celeba_example.jpg")
x_true = load_url_image(url=url, img_size=128,
                        resize_mode="resize", device=device)
x = x_true.clone()
mask = torch.ones_like(x)
mask[:, :, 32:96, 32:96] = 0
sigma_noise = 12.5 / 255.0  # noise level

physics = dinv.physics.Inpainting(
    mask=mask,
    tensor_size=x.shape[1:],
    noise_model=dinv.physics.GaussianNoise(sigma=sigma_noise),
    device=device,
)
y = physics(2 * x - 1)

# %%
# Run the PnP method
# ------------------
x_hat = pnpflow(y, physics)


imgs = [y, x_true, (x_hat + 1.0) * 0.5]
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
    mynet,
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
