r"""
Demo of spatial unwrapping
===================================================

In this example we show how to use the :class:`deepinv.physics.SpatialUnwrapping` forward model.

"""

import numpy as np
import torch
from deepinv.utils.plotting import plot

# set random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)


from deepinv.physics.spatial_unwrapping import SpatialUnwrapping

import deepinv as dinv
from deepinv.utils.demo import load_example

import torchvision.transforms as transforms


def channel_norm(x):
    x = (
        x - x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    )  # center the image around zero
    x = (
        x / x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    )  # normalize the image to [0, 1]
    return x


size = 256
dr = 2  # dynamic range
dtype = torch.float32
device = "cpu"
img_size = (size, size)
mode = "floor"  # available modes: "round", "floor"

x_rgb = load_example(
    "CBSD_0010.png", grayscale=False, device=device, dtype=dtype, img_size=img_size
)

x_rgb = channel_norm(x_rgb)  # normalize the image to [0, 1]
x_rgb = x_rgb * dr

# upscale the image to the desired size
factor = 3
resize = transforms.Resize(
    size=(img_size[0] * factor, img_size[1] * factor),
)

x_rgb = resize(x_rgb)

if mode == "round":
    # center the phase map around zero
    x_rgb = x_rgb - dr / 2

# apply blur filter
filter_0 = dinv.physics.blur.gaussian_blur(sigma=(1, 1), angle=0.0)
blur_op = dinv.physics.Blur(filter_0, device=device)
x_rgb = blur_op(x_rgb)


noise_model = dinv.physics.GaussianNoise(sigma=0.1)
physics = SpatialUnwrapping(threshold=1.0, mode=mode, noise_model=noise_model)

phase_map = x_rgb  # add batch and channel dimensions
wrapped_phase = physics(phase_map)
wrapped_phase = wrapped_phase

x_est = physics.A_dagger(wrapped_phase)
stepsize = 1e-4
lam = 2.0 / stepsize
x_models = []

prior = dinv.optim.TVPrior(n_it_max=10)
fidelity = dinv.optim.ItohFidelity()

psnr_fn = dinv.metric.PSNR()
ssim_fn = dinv.metric.SSIM()

params_algo = {"stepsize": stepsize, "lambda": lam, "g_param": 1.0}

model = dinv.optim.optim_builder(
    iteration="ADMM",
    prior=prior,
    data_fidelity=fidelity,
    max_iter=10,
    verbose=False,
    params_algo=params_algo,
)
x_model = model(wrapped_phase, physics, compute_metrics=False)


x_est = channel_norm(x_est)
x_model = channel_norm(x_model)
phase_map = channel_norm(phase_map)

# include PSNR for ADMM Inversion
psnr_admm = psnr_fn(phase_map, x_model).item()
psnr_dct = psnr_fn(phase_map, x_est).item()

ssim_admm = ssim_fn(phase_map, x_model).item()
ssim_dct = ssim_fn(phase_map, x_est).item()

imgs = [wrapped_phase[0], phase_map[0], x_est[0], x_model[0]]
titles = [
    "Wrapped Phase",
    "Original Phase",
    f"DCT Inversion\n PSNR={psnr_dct:.2f} SSIM={ssim_dct:.2f}",
    f"ADMM Inversion\n PSNR={psnr_admm:.2f} SSIM={ssim_admm:.2f}",
]

plot(imgs, titles=titles, cmap="gray", figsize=(20, 10))
