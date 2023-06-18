r"""
A tour of forward sensing operators
===================================================



"""

import deepinv as dinv
from deepinv.utils.plotting import plot
import torch
import torchvision
import requests
from imageio.v2 import imread
from io import BytesIO
from pathlib import Path

# %%
# Load image from the internet
# ----------------------------
#
# This example uses an image of Lionel Messi from Wikipedia.

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

url = (
    "https://upload.wikimedia.org/wikipedia/commons/b/b4/"
    "Lionel-Messi-Argentina-2022-FIFA-World-Cup_%28cropped%29.jpg"
)
res = requests.get(url)
x = imread(BytesIO(res.content)) / 255.0

x = torch.tensor(x, device=device, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
x = torch.nn.functional.interpolate(
    x, scale_factor=0.5
)  # reduce the image size for faster eval
x = torchvision.transforms.functional.center_crop(x, 32)
img_size = x.shape[1:]
# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)


# %%
# Denoising
# ---------------------------------------
#

physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(0.1))

y = physics(x)

# plot results
plot(
    [x, y],
    titles=["signal", "measurement"],
)

# %%
# Inpainting
# ---------------------------------------
#

sigma = 0.1  # noise level
physics = dinv.physics.Inpainting(mask=0.5, tensor_size=x.shape[1:], noise_model=dinv.physics.GaussianNoise(sigma=sigma),
                                  device=device)

y = physics(x)

# plot results
plot(
    [x, y],
    titles=["signal", "measurement"],
)

# %%
# Compressed Sensing
# ---------------------------------------
#

physics = dinv.physics.CompressedSensing(
    m=256, fast=False, channelwise=True, img_shape=img_size, device=device
)

y = physics(x)

# plot results
plot(
    [x, physics.A_dagger(y)],
    titles=["signal", "linear inverse"],
)

# %%
# Computed Tomography
# ---------------------------------------
#

physics = dinv.physics.Tomography(img_width=img_size[-1], angles=20, device=device,
                                  noise_model=dinv.physics.PoissonGaussianNoise(gain=0.1, sigma=0.05))

y = physics(x)

# plot results
plot(
    [x, (y-y.min())/y.max(), physics.A_dagger(y)],
    titles=["signal", "sinogram", "filtered backprojection"],
)

# %%
# MRI
# ---------------------------------------
#

mask = torch.rand((1, img_size[-1]), device=device) > 0.75
mask = torch.ones((img_size[-2], 1), device=device) * mask
mask[:, int(img_size[-1]/2)-2:int(img_size[-1]/2)+2] = 1

physics = dinv.physics.MRI(mask=mask, device=device, noise_model=dinv.physics.GaussianNoise(sigma=0.05))

x2 = torch.cat([x[:, 0, :, :].unsqueeze(1), torch.zeros_like(x[:, 0, :, :].unsqueeze(1))], dim=1)
y = physics(x2)

# plot results
plot(
    [x2, mask.unsqueeze(0).unsqueeze(0), physics.A_adjoint(y)],
    titles=["signal", "k-space mask", "linear inverse"],
)

# %%
# Decolorize
# ---------------------------------------
#

physics = dinv.physics.Decolorize()

y = physics(x)

# plot results
plot(
    [x, y],
    titles=["signal", "measurement"],
)

# %%
# Pan-sharpening
# ---------------------------------------
#

physics = dinv.physics.Pansharpen(img_size=img_size, device=device)

y = physics(x)

# plot results
plot(
    [x, y[0], y[1]],
    titles=["signal", "high res gray", "low res rgb"],
)

# %%
# Single-Pixel Camera
# ---------------------------------------
#

physics = dinv.physics.SinglePixelCamera(
        m=20, fast=True, img_shape=img_size, device=device
    )

y = physics(x)

# plot results
plot(
    [x, physics.A_adjoint(y)],
    titles=["signal", "linear inverse"],
)


# %%
# Blur
# ---------------------------------------
#

physics = dinv.physics.Blur(
    dinv.physics.blur.gaussian_blur(sigma=(2, 0.1), angle=45.0), device=device
)

y = physics(x)

# plot results
plot(
    [x, y],
    titles=["signal", "measurement"],
)

# %%
# Super-Resolution
# ---------------------------------------
#

physics = dinv.physics.Downsampling(img_size=img_size, factor=2, device=device)


y = physics(x)

# plot results
plot(
    [x, y],
    titles=["signal", "measurement"],
)
