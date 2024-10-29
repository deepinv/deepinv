import deepinv as dinv
from deepinv.utils.demo import load_url_image, get_image_url
import torch
import numpy as np
from sde import EDMSDE

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
url = get_image_url("CBSD_0010.png")
x = load_url_image(url=url, img_size=64, device=device)
x = x * 2 - 1 
class my_drunet(dinv.models.DRUNet):
    def __init__(self):
        super().__init__(pretrained = 'download', device = device)
    
    def forward(self, x, sigma):
        x = (x + 1) * 0.5
        out = super().forward(x, 0.5 * sigma)
        out = out * 2 - 1
        return out
denoiser = my_drunet()
prior = dinv.optim.prior.ScorePrior(denoiser=denoiser)

#VP 
timesteps = torch.linspace(0.001, 1., 100)
sigma = lambda t : ((np.exp( 0.5 * 19.9 * (t ** 2) + 0.1 * t - 1)) ** 0.5)
s = lambda t : 1 / (np.exp( 0.5 * 19.9 * (t ** 2) + 0.1 * t)) ** 0.5

OUSDE = EDMSDE(prior=prior, sigma = sigma, s = s, backward_ode = True)
with torch.no_grad():
    noise = torch.randn_like(x)
    sample = OUSDE.backward_sde.sample(noise.clone(), timesteps=timesteps.flip(dims=[0]))
    print(sample.min(), sample.max())
dinv.utils.plot([x, noise, sample])