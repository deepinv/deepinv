# %%
import torch
import sys
import torch.nn as nn
from sde import EDMSDE, DiffusionSDE
import deepinv as dinv
from deepinv.utils.demo import load_url_image, get_image_url
from edm import load_model

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

denoiser = load_model("edm-afhqv2-64x64-uncond-ve.pkl").to(device)
prior = dinv.optim.prior.ScorePrior(denoiser=denoiser)
url = get_image_url("CBSD_0010.png")
x = load_url_image(url=url, img_size=64, device=device)

with torch.no_grad():
    x_noisy = x + torch.randn_like(x) * 0.5
    x_denoised = denoiser(x_noisy, 0.5)
dinv.utils.plot([x, x_noisy, x_denoised])
