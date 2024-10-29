# %%
import torch
import sys
import torch.nn as nn
from sde import EDMSDE, DiffusionSDE
import deepinv as dinv
from deepinv.utils.demo import load_url_image, get_image_url
from edm import load_model
import numpy as np

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model("edm-ffhq-64x64-uncond-ve.pkl").to(device)
denoiser = lambda x, t: model(x.to(torch.float32), t).to(torch.float64)
prior = dinv.optim.prior.ScorePrior(denoiser=denoiser)
url = get_image_url("CBSD_0010.png")
x = load_url_image(url=url, img_size=64, device=device)


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
    samples = edm_sampler(model, latents=latents, num_steps=100)
    dinv.utils.plot([latents, samples])

# %%
ve_sigma = lambda t: t**0.5
ve_sigma_prime = lambda t: 1 / (2 * t**0.5)
ve_beta = lambda t: 0
ve_sigma_max = 100
ve_sigma_min = 0.02
num_steps = 400
ve_timesteps = ve_sigma_max**2 * (ve_sigma_min**2 / ve_sigma_max**2) ** (
    np.arange(num_steps) / (num_steps - 1)
)
sde = EDMSDE(prior=prior, beta=ve_beta, sigma=ve_sigma, sigma_prime=ve_sigma_prime)


with torch.no_grad():
    x_noisy = x + torch.randn_like(x) * 10.0
    x_denoised = denoiser(x_noisy, 10.0)
dinv.utils.plot([x, x_noisy, x_denoised])


# %%
with torch.no_grad():
    endpoint = sde.forward_sde.sample(x, ve_timesteps[::-1])
    print(f"End point std: {endpoint.std()}")
    dinv.utils.plot(endpoint)
    noise = torch.randn(2, 3, 64, 64, device=device) * ve_sigma_max
    samples = sde.backward_sde.sample(noise, timesteps=ve_timesteps)
dinv.utils.plot(samples)
