# %%
import torch
from dnnlib.util import open_url
import pickle
import sys
import torch.nn as nn

sys.path.append("../")
from sde import EDMSDE, DiffusionSDE
import deepinv as dinv
from deepinv.utils.demo import load_url_image, get_image_url


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network_pkl = (
    "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-ve.pkl"
)
with open_url(network_pkl) as f:
    net = pickle.load(f)["ema"].to(device)
    print(
        "Number of parameters: ",
        sum(p.numel() for p in net.model.parameters()),
    )


class ModelWrapper(nn.Module):
    def __init__(self, edm_model: nn.Module):
        super().__init__()
        self.edm_model = edm_model

    def forward(self, x: torch.Tensor, t: float):
        if isinstance(t, float):
            t = torch.tensor([t] * x.size(0), device=x.device)
        return self.edm_model.forward(x, noise_labels=t, class_labels=None)


denoiser = ModelWrapper(net.model)
prior = dinv.optim.prior.ScorePrior(denoiser=denoiser)
url = get_image_url("CBSD_0010.png")
x = load_url_image(url=url, img_size=64, device=device)

with torch.no_grad():
    x_noisy = x + torch.randn_like(x) * 0.1
    x_denoised = denoiser(x_noisy, 0.1)
dinv.utils.plot([x, x_noisy, x_denoised])
