r"""
Image reconstruction with a diffusion model
====================================================================================================

This code shows you how to use the DDRM diffusion algorithm to reconstruct images and also compute the
uncertainty of a reconstruction from incomplete and noisy measurements.

The paper can be found at https://arxiv.org/pdf/2209.11888.pdf.

The DDRM method requires that:

* The operator has a singular value decomposition (i.e., the operator is a :class:`deepinv.physics.DecomposablePhysics`).
* The noise is Gaussian with known standard deviation (i.e., the noise model is :class:`deepinv.physics.GaussianNoise`).

"""

# %%
# Load example image from the internet
# --------------------------------------------------------------
#
# This example uses an image of Lionel Messi from Wikipedia.
import deepinv as dinv
from deepinv.utils.plotting import plot
import torch
import numpy as np
from deepinv.utils.demo import load_url_image, get_image_url
from tqdm import tqdm

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

url = get_image_url("00000.png")
x_true = load_url_image(url=url, img_size=128).to(device)
x_true = x_true[:, :3, :, :]
plot(x_true)

# %%
# Define forward operator and noise model
# --------------------------------------------------------------
#
# We use image inpainting as the forward operator and Gaussian noise as the noise model.

sigma = 5 / 255  # noise level
mask = torch.ones(1, 3, 128, 128).to(device)
mask[:, :, 32:96, 32:96] = 0

physics = dinv.physics.Inpainting(
    mask=0.8,
    tensor_size=x_true.shape[1:],
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=sigma),
)
y = physics(x_true)
plot(y)
# %%
# Define the MMSE denoiser
# --------------------------------------------------------------
#
# The diffusion method requires an MMSE denoiser that can be evaluated a various noise levels.
# Here we use a pretrained DRUNET denoiser from the :ref:`denoisers <denoisers>` module.

# denoiser = dinv.models.DRUNet(pretrained="download").to(device)
denoiser = dinv.models.DiffUNet(large_model=False).to(device)

# %%
# Create the Monte Carlo sampler
# --------------------------------------------------------------
#
# We can now reconstruct a noisy measurement using the diffusion method.
# We use the DDRM method from :class:`deepinv.sampling.DDRM`, which works with inverse problems that
# have a closed form singular value decomposition of the forward operator.
# The diffusion method requires a schedule of noise levels ``sigmas`` that are used to evaluate the denoiser.

sigmas = np.linspace(1, 0, 100) if torch.cuda.is_available() else np.linspace(1, 0, 10)

diff = dinv.sampling.DDRM(denoiser=denoiser, etab=1.0, sigmas=sigmas, verbose=True)

# %%
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

# %%
y = physics(x_true)

i = 100  # choose some arbitrary timestep
t = (torch.ones(1) * i).to(device)
at = compute_alpha(betas, t.long())
xt = at.sqrt() * y + (1 - at).sqrt() * torch.randn_like(y)

sigma = (1 - at).sqrt() / at.sqrt()

sigma_noise = physics.noise_model.sigma
x_bar = physics.V_adjoint(x_true)

y_bar = physics.U_adjoint(y)
case = physics.mask > sigma_noise
y_bar[case] = y_bar[case] / physics.mask[case]

loss = y_bar - physics.mask * x_bar

# %%
# DDRM
x0 = y.clone().to(device)
i = 10  # choose some arbitrary timestep
t = (torch.ones(1) * i).to(device)
at = compute_alpha(betas, t.long())
xt = at.sqrt() * x0 + (1 - at).sqrt() * torch.randn_like(x0)

sigma = (1 - at).sqrt() / at.sqrt()

Sigma = physics.mask
Sigma_T = torch.transpose(Sigma, -2, -1)

if hasattr(physics.noise_model, "sigma"):
    sigma_noise = physics.noise_model.sigma
else:
    sigma_noise = 0.01

identity = (
    torch.eye(n=Sigma.size(-2), m=Sigma.size(-1), device=x.device)
    .unsqueeze(0)
    .unsqueeze(0)
)
identity = torch.ones_like(Sigma)

tmp = torch.pinverse(torch.abs(sigma_noise**2 * identity - sigma**2 * Sigma * Sigma_T))

tmp = torch.abs(sigma_noise**2 * identity - sigma**2 * Sigma * Sigma_T)
tmp[tmp > 0] = 1 / tmp[tmp > 0]
tmp[tmp == 0] = 0

grad_norm_op = -Sigma * tmp

grad_norm = physics.V(grad_norm_op * loss)
plot([grad_norm_op[:, :, :20, :20], physics.mask[..., :20, :20]], figsize=(10, 5))
plot([grad_norm, physics.mask], cbar=True, figsize=(10, 5))

# %%
noisy_datafidelity = dinv.sampling.noisy_datafidelity.DDRMDataFidelity(
    physics, denoiser
)

num_steps = 1000

skip = num_train_timesteps // num_steps

batch_size = 1
eta = 1.0

seq = range(0, num_train_timesteps, skip)
seq_next = [-1] + list(seq[:-1])
time_pairs = list(zip(reversed(seq), reversed(seq_next)))

# measurement
x0 = y * 2.0 - 1.0

# initial sample from x_T
x = torch.randn_like(x0)

xs = [x]
x0_preds = []

for i, j in tqdm(time_pairs):
    t = (torch.ones(batch_size) * i).to(device)
    next_t = (torch.ones(batch_size) * j).to(device)

    at = compute_alpha(betas, t.long())
    at_next = compute_alpha(betas, next_t.long())
    sigma = (1 - at).sqrt() / at.sqrt()

    xt = xs[-1].to(device)
    norm_grad = noisy_datafidelity.grad(xt / 2 + 0.5, y, sigma / 2)
    x0_t = denoiser(xt / 2 + 0.5, sigma / 2)
    sigma_tilde = ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt() * eta
    c2 = ((1 - at_next) - sigma_tilde**2).sqrt()

    # 3. noise step
    epsilon = torch.randn_like(xt)

    # 4. DDPM(IM) step
    xt_next = (
        (at_next.sqrt() - c2 * at.sqrt() / (1 - at).sqrt()) * x0_t
        + sigma_tilde * epsilon
        + c2 * xt / (1 - at).sqrt()
        - norm_grad
    )

    norm_grad.detach()
    x0_preds.append(x0_t.detach().to("cpu"))
    xs.append(xt_next.detach().to("cpu"))

recon = xs[-1]

# plot the results
x = recon / 2 + 0.5
imgs = [y, x]
plot(imgs, titles=["measurement", "model output", "groundtruth"])

# %%
# Create a Monte Carlo sampler
# ---------------------------------------------------------------------------------
# Running the diffusion gives a single sample of the posterior distribution.
# In order to compute the posterior mean and variance, we can use multiple samples.
# This can be done using the :class:`deepinv.sampling.DiffusionSampler` class, which converts
# the diffusion algorithm into a fully fledged Monte Carlo sampler.
# We set the maximum number of iterations to 10, which means that the sampler will run the diffusion 10 times.

f = dinv.sampling.DiffusionSampler(diff, max_iter=10)


# %%
# Run sampling algorithm and plot results
# ---------------------------------------------------------------------------------
# The sampling algorithm returns the posterior mean and variance.
# We compare the posterior mean with a simple linear reconstruction.

mean, var = f(y, physics)

# compute PSNR
print(f"Linear reconstruction PSNR: {dinv.metric.PSNR()(x, x_lin).item():.2f} dB")
print(f"Posterior mean PSNR: {dinv.metric.PSNR()(x, mean).item():.2f} dB")

# plot results
error = (mean - x).abs().sum(dim=1).unsqueeze(1)  # per pixel average abs. error
std = var.sum(dim=1).unsqueeze(1).sqrt()  # per pixel average standard dev.
imgs = [
    x_lin,
    x,
    mean,
    std / std.flatten().max(),
    error / error.flatten().max(),
]
plot(
    imgs,
    titles=[
        "measurement",
        "ground truth",
        "post. mean",
        "post. std",
        "abs. error",
    ],
)
