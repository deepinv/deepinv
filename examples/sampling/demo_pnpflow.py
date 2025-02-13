r"""
Implementing PnP-Flow
=====================

In this tutorial, we revisit the implementation of the PnP-Flow flow matching
algorithm for image reconstruction from
`Martin et al. <https://arxiv.org/pdf/2410.02423>`_. The full algorithm is
implemented in :class:`deepinv.sampling.pnpflow`.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import deepinv as dinv
from deepinv.utils.plotting import plot
from deepinv.optim.data_fidelity import L2
from deepinv.utils.demo import load_url_image, get_image_url
from deepinv.models.flowunet import FlowUNet
from deepinv.sampling.pnpflow import PnPFlow

# Use matplotlib config from deepinv to get nice plots
from deepinv.utils.plotting import config_matplotlib

config_matplotlib()


device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
torch.manual_seed(1)

mynet = FlowUNet(input_channels=3,
                 input_height=128, pretrained='True', device=device)


### INPAINTING EXAMPLE ###
pnpflow = PnPFlow(mynet, data_fidelity=L2(),
                  verbose=True, max_iter=100, device=device, lr=1.0, lr_exp=0.6)
print("Running inpainting example")

url = get_image_url("celeba_example.jpg")
# url = get_image_url("69037.png")
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
y = physics(2*x - 1)
x_hat = pnpflow.forward(y, physics)


imgs = [y, x_true, (x_hat + 1.0)*0.5]
plot(
    imgs,
    titles=["measurement", "ground-truth", "reconstruction"],
    save_fn='res_inpainting.png', save_dir='.'
)

print('PSNR noisy image :', dinv.metric.PSNR()(y, x_true).item())
print('PSNR restored image :', dinv.metric.PSNR()
      ((x_hat + 1.0)*0.5, x_true).item())

### DEBLURRING EXAMPLE ###
print("Running deblurring example")
pnpflow = PnPFlow(mynet, data_fidelity=L2(),
                  verbose=True, max_iter=100, device=device, lr=1.0, lr_exp=0.01)

url = get_image_url("celeba_example2.jpg")
x_true = load_url_image(url=url, img_size=128,
                        resize_mode="resize", device=device)
x = x_true.clone()
sigma_noise = 12.75 / 255.0  # noise level

physics = dinv.physics.BlurFFT(
    img_size=x.shape[1:], filter=dinv.physics.blur.gaussian_blur(sigma=1.),
    noise_model=dinv.physics.GaussianNoise(sigma=sigma_noise),
    device=device)
y = physics(2*x-1)

x_hat = pnpflow.forward(y, physics)

imgs = [y, x_true, (x_hat + 1.0)*0.5]
plot(
    imgs,
    titles=["measurement", "ground-truth", "reconstruction"],
    save_fn='res_blurfft.png', save_dir='.'
)

print('PSNR noisy image :', dinv.metric.PSNR()(y, x_true).item())
print('PSNR restored image :', dinv.metric.PSNR()
      ((x_hat + 1.0)*0.5, x_true).item())
