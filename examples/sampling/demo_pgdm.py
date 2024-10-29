# %%
# Load image and physics

import numpy as np
import torch

import deepinv as dinv
from deepinv.utils.plotting import plot
from deepinv.optim.data_fidelity import L2
from deepinv.utils.demo import load_url_image, get_image_url
from tqdm import tqdm  # to visualize progress

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

url = get_image_url("00000.png")

img_size = 128
x_true = load_url_image(url=url, img_size=img_size).to(device)
x_true = x_true[
    :,
    :3,
]

sigma = 5 / 255.0  # noise level

physics = dinv.physics.Inpainting(
    tensor_size=(3, img_size, img_size),
    mask=0.05,
    pixelwise=True,
    device=device,
)

y = physics(x_true)

plot(
    [y, x_true],
    titles=["measurement", "groundtruth"],
)

# %%
# Load diffusion model and compute coefficients

num_train_timesteps = 1000  # Number of timesteps used during training


def get_betas(
    beta_start=0.1 / 1000, beta_end=20 / 1000, num_train_timesteps=num_train_timesteps
):
    betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    betas = torch.from_numpy(betas).to(device)

    return betas


# Utility function to let us easily retrieve \bar\alpha_t
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


betas = get_betas()

model = dinv.models.DiffUNet(large_model=False).to(device)


# %%
# PGDM implementation using deepinv

num_steps = 1000
skip = num_train_timesteps // num_steps

eta = 1.0
grad_weight = 1.0
batch_size = 1

seq = range(0, num_train_timesteps, skip)
seq_next = [-1] + list(seq[:-1])
time_pairs = list(zip(reversed(seq), reversed(seq_next)))

# Initialize xt
# y_0 = y.clone()
# x_0 = physics.A_dagger(y_0)
# t = (torch.ones(1) * num_steps).to(device)
# alpha_t = compute_alpha(betas, t.long())
# xt = alpha_t.sqrt() * x_0 + (1 - alpha_t).sqrt() * torch.randn_like(x_0)
xt = torch.randn_like(y)

xs = [xt]
x0_preds = []

with torch.no_grad():
    for i, j in tqdm(time_pairs):
        t = (torch.ones(batch_size) * i).to(device)
        next_t = (torch.ones(batch_size) * j).to(device)

        at = compute_alpha(betas, t.long())
        at_next = compute_alpha(betas, next_t.long())

        c1 = ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt() * eta
        c2 = ((1 - at_next) - c1**2).sqrt()

        with torch.enable_grad():
            xt.requires_grad_(True)

            scale = 1.0
            sigma_t = (1 - at).sqrt() / at.sqrt()
            x0_t = model(xt / 2 + 0.5, sigma_t / 2) * 2 - 1
            et = (xt - x0_t * at.sqrt()) / (1 - at).sqrt()
            mat = (physics.A_dagger(y) - physics.A_dagger(physics.A(x0_t))).reshape(
                batch_size, -1
            )
            mat_x = (mat.detach() * x0_t.reshape(batch_size, -1)).sum()

        grad_term = torch.autograd.grad(mat_x, xt, retain_graph=True)[0]
        coeff = at_next.sqrt() * at.sqrt() * grad_weight

        x0_t = x0_t.detach()
        et = et.detach()

        xt_next = (
            at_next.sqrt() * x0_t
            + c1 * torch.randn_like(xt)
            + c2 * et
            + grad_term * coeff
        )

        xs.append(xt_next.detach().cpu())
        x0_preds.append(x0_t.detach().cpu())
        xt = xt_next

recon = xs[-1]

# plot the results
x = recon / 2 + 0.5
imgs = [y, x, x_true]
plot(imgs, titles=["measurement", "model output", "groundtruth"])

# %%
# PGDM implementation using PGDMDataFidelity


num_steps = 1000
skip = num_train_timesteps // num_steps

eta = 1.0
grad_weight = 1.0
batch_size = 1

seq = range(0, num_train_timesteps, skip)
seq_next = [-1] + list(seq[:-1])
time_pairs = list(zip(reversed(seq), reversed(seq_next)))

# Initialize xt
# y_0 = y.clone()
# x_0 = physics.A_dagger(y_0)
# t = (torch.ones(1) * num_steps).to(device)
# alpha_t = compute_alpha(betas, t.long())
# xt = alpha_t.sqrt() * x_0 + (1 - alpha_t).sqrt() * torch.randn_like(x_0)
xt = torch.randn_like(y)

xs = [xt]
x0_preds = []

data_fidelity = dinv.sampling.noisy_datafidelity.PGDMDataFidelity(
    physics=physics, denoiser=model
)

with torch.no_grad():
    for i, j in tqdm(time_pairs):
        t = (torch.ones(batch_size) * i).to(device)
        next_t = (torch.ones(batch_size) * j).to(device)

        at = compute_alpha(betas, t.long())
        at_next = compute_alpha(betas, next_t.long())
        sigma_t = (1 - at).sqrt() / at.sqrt()

        c1 = ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt() * eta
        c2 = ((1 - at_next) - c1**2).sqrt()

        xt = xs[-1].to(device)

        grad_term = data_fidelity.grad(xt, y, sigma_t)
        coeff = at_next.sqrt() * at.sqrt() * grad_weight

        x0_t = x0_t.detach()
        et = et.detach()

        xt_next = (
            at_next.sqrt() * x0_t
            + c1 * torch.randn_like(xt)
            + c2 * et
            + grad_term * coeff
        )

        xs.append(xt_next.detach().cpu())
        x0_preds.append(x0_t.detach().cpu())

recon = xs[-1]

# plot the results
x = recon / 2 + 0.5
imgs = [y, x, x_true]
plot(imgs, titles=["measurement", "model output", "groundtruth"])
