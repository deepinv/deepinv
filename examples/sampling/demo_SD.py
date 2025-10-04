r"""
Latent DDIM and PSLD
================================

This example demonstrates:

1. **Unconditional / text-to-image sampling** in the **latent** space using a
   Stable-Diffusion–style model via a **DDIM** sampler
   :footcite:t:`song2020denoising`, where :math:`\eta \ge 0` controls the
   stochasticity (``eta=0`` → deterministic DDIM; larger ``eta`` injects noise).
2. A simple **forward (measurement) model**: inpainting + Gaussian noise.
3. **Posterior sampling** with **PSLD** (Posterior Sampling with Latent Diffusion)
   :footcite:t:`Rout2023SolvingLI`, conditioning on the measurement.

Requirements
------------
* A pretrained latent diffusion prior (UNet in latent space + VAE encoder/decoder).
* A forward operator :math:`A` and its adjoint :math:`A^*` (here, inpainting), and a measurement :math:`y`.
* Consistent data range across models/operators (this demo uses VAE images in ``[-1, 1]``).

DDIM in latent space (:math:`\eta \ge 0`)
-------------------------------------------------
Let :math:`\hat\epsilon = \epsilon_\theta(z_t, t)` be the predicted noise at step :math:`t`.
With cumulative schedule :math:`\bar\alpha_t`, the **proxy clean latent** is:

.. math::
    \hat z_0
    \;=\;
    \frac{z_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon}{\sqrt{\bar\alpha_t}}.

The **DDIM** update is:

.. math::
    z_{t-1}
    \;=\;
    \sqrt{\bar\alpha_{t-1}}\,\hat z_0
    \;+\;
    \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\hat\epsilon
    \;+\;
    \sigma_t\,\xi,\quad \xi\sim\mathcal{N}(0,I),

where the noise scale :math:`\sigma_t` depends on :math:`\eta`:

.. math::
    \sigma_t
    \;=\;
    \eta\,
    \sqrt{\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}}\,
    \sqrt{1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}}.

Special cases: ``eta=0`` → deterministic DDIM; larger ``eta`` increases stochasticity
(``eta≈1`` is close to DDPM-like behavior with the same schedule).

PSLD: Posterior Sampling with Latent Diffusion
----------------------------------------------
PSLD augments the DDIM proposal with a **data-consistency** correction computed by differentiating
a measurement loss **through the VAE** (latent :math:`\leftrightarrow` image):

1. Predict :math:`\hat z_0(z_t)` and decode: :math:`\hat x_0=\mathcal{D}(\hat z_0)`.
2. Data loss (example): :math:`\mathcal{L}_\text{data}=\|A(\hat x_0)-y\|_2`.
3. Optional latent "gluing" via an image-space projection
   :math:`\Pi(x)=A^*y + (I-A^*A)\,x`:

   .. math::
       \mathcal{L}_\text{glue}=\|\mathcal{E}(\Pi(\hat x_0))-\hat z_0\|_2.

4. Combine: :math:`\mathcal{L} = \omega\,\mathcal{L}_\text{data} + \gamma\,\mathcal{L}_\text{glue}`.
5. Take a gradient step **w.r.t. the current latent** :math:`z_t`:

   .. math::
       z_{t-1} \;=\; z'_{t-1} \;-\; \eta_t \,\nabla_{z_t}\mathcal{L}.

In words: **DDIM** proposes :math:`z'_{t-1}` (possibly with noise via ``eta``), then **PSLD**
pulls it toward the measurement by back-propagating the loss through :math:`\mathcal{D}` and
(optionally) :math:`\mathcal{E}`.

Range conventions
-----------------
Stable Diffusion’s VAE commonly uses images in ``[-1, 1]``. If your physics :math:`A` and
measurements :math:`y` are in ``[0, 1]``, map appropriately (e.g., :math:`y' = 2y-1`) so the loss
is numerically well-posed.

.. note::

   For speed, the demo limits the number of steps; in practice, larger schedules tend
   to improve quality. With fixed seeds, DDIM is deterministic at ``eta=0``; PSLD’s gradient
   step adds optimization dynamics independent of ``eta``.

"""

# %%
import torch
import random
import numpy as np
import deepinv as dinv
from deepinv.models import LatentDiffusion
from deepinv.sampling import DDIMDiffusion, PSLDDiffusionPosterior

# %% [markdown]
# Global configuration
# --------------------

# Fixed seed (CPU + GPU + NumPy + Python's RNG)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# Determinism hints (may reduce performance)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
figsize = 2.5

# %% [markdown]
# 1) Unconditional / Text-to-Image Sampling
# -----------------------------------------
# - Instantiate a Stable-Diffusion–style latent model via deepinv's wrapper.
# - Run a **generic** DDIM sampler in latent space for a fixed number of steps (set `eta`).
# - Decode the latent with the SD VAE and rescale to [0, 1] for plotting.

denoiser = LatentDiffusion().to(device)

# Number of denoising steps for DDIM (20–500 typical; 200 gives good quality)
num_steps = 200

# DDIM sampler configured with SD-like beta range; prompt is optional
sampler = DDIMDiffusion(
    model=denoiser,
    beta_min=0.00085,
    beta_max=0.012,
    num_inference_steps=num_steps,
    prompt=(
        "a photorealistic close-up portrait of a (golden retriever:1.2) "
        "sitting in a sunlit meadow, detailed fur texture, wet nose, "
        "catchlights in the eyes, shallow depth of field, creamy bokeh, "
        "85mm photo, f/2.0, golden hour lighting, natural colors, "
        "ultra-sharp focus, high detail"
    ),
)

# Decoded image size; SD1.5 latent spatial size is /8
height, width = 512, 512

# Initial latent ~ N(0, I). Use float16 for speed/memory on GPU.
latents = torch.randn(
    (1, denoiser.unet.config.in_channels, height // 8, width // 8),
    device=device,
    dtype=torch.float16,
)

# DDIM forward: produce a latent sample (z_T -> z_0).  eta=0 → deterministic.
z = sampler.forward(latents, eta=0.5)  # try 0.0, 0.5, 1.0

# Decode to [-1, 1], then rescale to [0, 1] for visualization
x = denoiser.decode(z.half()) * 0.5 + 0.5

# Save unconditional sample
dinv.utils.plot(
    x,
    titles="Unconditional generation",
    save_fn="sde_sample.png",
    figsize=(figsize, figsize),
)

# %% [markdown]
# 2) Forward (Measurement) Model: Inpainting + Noise
# --------------------------------------------------
# - Build an inpainting operator (mask a 100x100 square) + Gaussian noise.
# - Generate measurement `y = A(x) + n` on the image domain.

mask = torch.ones_like(x)
mask[..., 100:200, 100:200] = 0.0

noise_model = dinv.physics.GaussianNoise(sigma=0.01)
physics = dinv.physics.Inpainting(
    tensor_size=x.shape[1:],  # (C, H, W)
    mask=mask,
    device=device,
    noise_model=noise_model,
)

y = physics(x.float())

# Visualize original vs. measurement
dinv.utils.plot(
    [x, y],
    show=True,
    titles=["Original", "Measurement"],
    figsize=(figsize * 2, figsize),
)

# %% [markdown]
# 3) Posterior Sampling with PSLD
# -------------------------------
# - Reuse the same latent diffusion prior.
# - Use PSLD to sample from an approximate posterior consistent with `y` and `A`.
# - SD VAE works in `[-1, 1]`; map `y` to that range via `y' = 2*y - 1`.
# - You may set **two** step sizes:
#     * `DDIM_eta` → **DDIM** noise level (stochasticity).
#     * `dps_eta` → **PSLD** gradient step size.

# A longer schedule (e.g., 500 or 999) is common for SD-style timesteps
num_steps = 999
sampler = PSLDDiffusionPosterior(
    model=denoiser,
    beta_min=0.00085,
    beta_max=0.012,
    num_inference_steps=num_steps,
)

# Fresh initial latent for posterior sampling
latents = torch.randn(
    (1, denoiser.unet.config.in_channels, height // 8, width // 8),
    device=device,
    dtype=torch.float16,
)

# PSLD run.  forward_model=physics enforces data-consistency; gamma tunes strength.
# `DDIM_eta` controls the DDIM stochasticity inside PSLD; `dps_eta` the PSLD gradient step.
z = sampler.forward(
    latents,
    y=2 * y - 1,
    forward_model=physics,
    gamma=0.1,
    dps_eta=1.0,
    DDIM_eta=0.0,  # DDIM noise inside PSLD; set 0.0 for deterministic proposal
)

# Decode and rescale to [0, 1] for visualization
x_hat = denoiser.decode(z.half()) * 0.5 + 0.5

# Compare original, measurement, and posterior sample
dinv.utils.plot(
    [x, y, x_hat],
    show=True,
    titles=["Original", "Measurement", "Posterior sample"],
    save_fn="posterior_sample.png",
    figsize=(figsize * 3, figsize),
)

# %%
# :References:
#
# .. footbibliography::
