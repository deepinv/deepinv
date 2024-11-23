# %%
import torch
import torch.nn as nn
from sde import DiffusionSDE
from sde_solver import EulerSolver, HeunSolver
import deepinv as dinv
from deepinv.utils.demo import load_url_image, get_image_url
from edm import load_model
import numpy as np
from utils import get_edm_parameters
from deepinv.models.edm import EDMPrecond, SongUNet, VEPrecond, VPPrecond, iDDPMPrecond

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet = SongUNet(
    img_resolution=64,
    in_channels=3,
    out_channels=3,
    augment_dim=9,
    model_channels=128,
    channel_mult=[1, 2, 2, 2],
    channel_mult_noise=2,
    embedding_type="fourier",
    encoder_type="residual",
    decoder_type="standard",
    resample_filter=[1, 3, 3, 1],
)
unet.load_state_dict(
    torch.load(
        "/home/minhhai/Works/dev/deepinv_folk/weights/edm/edm-ffhq-64x64-uncond-ve.pt"
    ),
    strict=True,
)
print(unet)
print(sum(p.numel() for p in unet.parameters()))
for p in unet.parameters():
    p.requires_grad_(False)
# %%
# model = EDMPrecond(model=unet).to(device)
# denoiser = lambda x, t: model(x.to(torch.float32), t).to(torch.float64)
denoiser = dinv.models.DRUNet(pretrained="download").to(device)
prior = dinv.optim.prior.ScorePrior(denoiser=denoiser)
url = get_image_url("CBSD_0010.png")
x = load_url_image(url=url, img_size=64, device=device)

x_noisy = x + torch.randn_like(x) * 1.
dinv.utils.plot([x, x_noisy, denoiser(x_noisy, 1.)])

dinv.utils.plot(prior.score(x_noisy, 1))


# %%
def edm_sampler(
    model: nn.Module,
    latents: torch.Tensor,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
):
    # Time step discretization.
    step_indices = np.arange(num_steps)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = np.concatenate([t_steps, np.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    # 0, ..., N-1
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        # Increase noise temporarily.

        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        )
        t_hat = t_cur + gamma * t_cur
        x_hat = x_cur + np.sqrt(t_hat**2 - t_cur**2) * S_noise * torch.randn_like(x_cur)

        # Euler step.
        denoised = model(x_hat.to(torch.float32), t_hat).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = model(x_next.to(torch.float32), t_next).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


# %%
with torch.no_grad():
    latents = torch.randn(2, 3, 64, 64, device=device)
    samples = edm_sampler(model, latents=latents, num_steps=20)
    dinv.utils.plot([latents, samples])

# %%
params = get_edm_parameters("ve")
sigma_min = 0.01
sigma_max = 100
# timesteps_fn = params["timesteps_fn"]
drift = lambda x, t: 0.0
diffusion = (
    lambda t: sigma_min
    * (sigma_max / sigma_min) ** t
    * np.sqrt(2 * (np.log(sigma_max) - np.log(sigma_min)))
)
sde = DiffusionSDE(drift=drift, diffusion=diffusion, prior=prior, device=device)
x_init = torch.randn((1, 3, 64, 64), device=device)
timesteps_fn = lambda n: np.linspace(0.001, 1, n)[::-1]

import matplotlib.pyplot as plt

plt.plot(timesteps_fn(100))
plt.show()
plt.plot([diffusion(t) for t in timesteps_fn(1000)])

plt.plot(np.exp(np.linspace(np.log(sigma_min), np.log(sigma_max), 1000))[::-1])
# %%
num_steps = 200
with torch.no_grad():
    noise = torch.randn(1, 3, 64, 64, device=device) * sigma_max
    samples = sde.backward_sde.sample(noise, timesteps=timesteps_fn(num_steps))
dinv.utils.plot(samples)

# %%
num_steps = 100
euler_solver = EulerSolver(drift=sde.drift_back, diffusion=sde.diff_back)
with torch.no_grad():
    noise = torch.randn(2, 3, 64, 64, device=device) * sigma_max
    samples = euler_solver.sample(noise, timesteps=timesteps_fn(num_steps))
dinv.utils.plot(samples)

# %%
num_steps = 100
heun_solver = HeunSolver(drift=sde.drift_back, diffusion=sde.diff_back)
with torch.no_grad():
    noise = torch.randn(2, 3, 64, 64, device=device) * sigma_max
    samples = heun_solver.sample(noise, timesteps=timesteps_fn(num_steps))
dinv.utils.plot(samples)
