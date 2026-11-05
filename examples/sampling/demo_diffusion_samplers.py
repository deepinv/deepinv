r"""
Posterior sampling with DPS, DiffPIR and DDRM
=============================================

This example compares three diffusion posterior samplers on a noisy inpainting problem:

- Diffusion Posterior Sampling (DPS) :footcite:t:`chung2022diffusion`, implemented in :class:`deepinv.sampling.DPS`,
- Diffusion PnP Image Restoration (DiffPIR) :footcite:t:`zhu2023denoising`, implemented in :class:`deepinv.sampling.DiffPIR`,
- Denoising Diffusion Restoration Models (DDRM) :footcite:t:`kawar2022denoising`, implemented in :class:`deepinv.sampling.DDRM`.

All three are instances of :class:`deepinv.sampling.PosteriorDiffusion`. They solve the reverse-time SDE of a diffusion
model, in which the score of the prior is replaced by the score of the posterior

.. math::

    \nabla_{x_t} \log p_t(x_t \vert y) = \nabla_{x_t} \log p_t(x_t) + \nabla_{x_t} \log p_t(y \vert x_t).

The first term is given by a denoiser through Tweedie's formula. The second one is intractable, and each method
approximates it differently, with a :class:`deepinv.sampling.NoisyDataFidelity`. These approximations are compared in
:ref:`sphx_glr_auto_examples_sampling_demo_noisy_data_fidelity.py`, and
:ref:`sphx_glr_auto_examples_sampling_demo_diffusion_sde.py` shows how to assemble a custom posterior sampler.
"""

# %%
# Generate an inverse problem
# ---------------------------
#
# We consider random inpainting, implemented in :class:`deepinv.physics.Inpainting`, where 70% of the pixels are
# masked out, with additive white Gaussian noise of standard deviation :math:`\sigma_y = 12.75/255`.
#
# .. note::
#           We work with an image of size 64 x 64 to reduce the computational time of this example.

# sphinx_gallery_tags = ["Inpainting"]

import time

import torch

import deepinv as dinv
from deepinv.utils import load_example
from deepinv.utils.plotting import plot

device = dinv.utils.get_device()

x_true = load_example("butterfly.png", img_size=64, device=device)

physics = dinv.physics.Inpainting(
    img_size=x_true.shape[1:],
    mask=0.3,
    pixelwise=True,
    noise_model=dinv.physics.GaussianNoise(sigma=12.75 / 255.0),
    device=device,
    rng=torch.Generator(device).manual_seed(0),
)
y = physics(x_true, seed=0)

plot({"Measurement": y, "Ground truth": x_true})

# %%
# Load a pretrained denoiser
# --------------------------
#
# The three samplers rely on a denoiser :math:`\denoiser{x}{\sigma}` that can be evaluated at any noise level
# :math:`\sigma`. We use the same pretrained DRUNet for all of them, but any denoiser taking an image and a noise
# level as input works, such as the diffusion networks :class:`deepinv.models.DiffUNet` or
# :class:`deepinv.models.NCSNpp`.

denoiser = dinv.models.DRUNet(pretrained="download").to(device)

# %%
# DPS
# ---
#
# DPS approximates the posterior mean :math:`\mathbb{E}[x_0 \vert x_t]` by the denoised estimate
# :math:`\denoiser{x_t}{\sigma_t}`, and differentiates the measurement residual through the denoiser.
# With ``guidance="annealed"`` (see :class:`deepinv.sampling.DPSDataFidelity`), the likelihood score is
#
# .. math::
#
#     \nabla_{x_t} \log p_t(y \vert x_t) \approx
#     - \lambda \nabla_{x_t} \frac{\|y - A \denoiser{x_t}{\sigma_t}\|_2^2}{2 (\sigma_y^2 + \sigma_t^2)},
#
# where :math:`\lambda` is the ``weight`` parameter. It only requires a differentiable forward operator, so that
# DPS also handles nonlinear problems, at the cost of a backward pass through the denoiser on each step.
# The default solver is the DDPM sampler of the original paper, which works best with many steps
# (``num_steps=1000`` in the paper). We use 200 steps to reduce the computational time.
#
# .. note::
#
#     We pass a random number generator ``rng`` to each sampler, so that the samples are reproducible with
#     the ``seed`` argument.

dps = dinv.sampling.DPS(
    denoiser,
    num_steps=200,
    weight=5.0,
    guidance="annealed",
    device=device,
    rng=torch.Generator(device),
)

# %%
# DiffPIR
# -------
#
# DiffPIR does not differentiate through the denoiser. It applies the proximal operator of a data-fidelity term
# :math:`f(\cdot, y)` to the denoised estimate,
#
# .. math::
#
#     \hat{x}_0 = \operatorname{prox}_{\frac{\sigma_t^2}{\lambda \sigma_y^2} f(\cdot, y)}\left(\denoiser{x_t}{\sigma_t}\right),
#
# and plugs :math:`\hat{x}_0` in Tweedie's formula in place of the denoised estimate
# (see :class:`deepinv.sampling.DiffPIRDataFidelity`). Here :math:`f` is the squared :math:`\ell_2` distance
# :class:`deepinv.optim.L2`, but any data-fidelity term with a proximal operator works, such as
# :class:`deepinv.optim.PoissonLikelihood`. The ``weight`` parameter :math:`\lambda` trades off the denoiser and the
# measurements.
#
# Each step of DiffPIR then moves to :math:`\hat{x}_0` and renews a fraction ``zeta`` of the noise
# (:math:`\zeta = 1` by default), which is the step of :class:`deepinv.sampling.AncestralSolver` with ``zeta``.

diffpir = dinv.sampling.DiffPIR(
    denoiser,
    data_fidelity=dinv.optim.L2(),
    num_steps=100,
    weight=7.0,
    zeta=1.0,
    device=device,
    rng=torch.Generator(device),
)

# %%
# DDRM
# ----
#
# DDRM requires a linear operator with a singular value decomposition :math:`A = U \Sigma V^\top`
# (:class:`deepinv.physics.DecomposablePhysics`, as for inpainting) and Gaussian noise. It works in the spectral
# domain: the singular components of :math:`x_t` that are measured with a higher signal-to-noise ratio than the
# current noise level are moved toward the measurements before denoising, and the denoised estimate is then corrected
# in closed form (see :class:`deepinv.sampling.DDRMDataFidelity`). As DiffPIR, it does not differentiate
# through the denoiser.
#
# The parameter ``eta`` sets the stochasticity of the sampler: each step renews a fraction
# :math:`\zeta = \eta^2` of the noise of the unmeasured components.

ddrm = dinv.sampling.DDRM(
    denoiser,
    num_steps=100,
    eta=0.85,
    device=device,
    rng=torch.Generator(device),
)

# %%
# Run the samplers
# ----------------
#
# Each call returns one sample of the posterior distribution. DPS needs gradients through the denoiser, which it
# computes internally, so that all three samplers can be run in a ``torch.no_grad()`` context.

models = {"DPS": dps, "DiffPIR": diffpir, "DDRM": ddrm}
psnr = dinv.metric.PSNR()

samples = {}
for name, model in models.items():
    start = time.time()
    with torch.no_grad():
        samples[name] = model(y, physics, seed=1)
    print(
        f"{name:>8s}: PSNR {psnr(samples[name], x_true).item():.2f} dB, "
        f"{len(model.solver.timesteps)} steps in {time.time() - start:.1f} s"
    )

plot(
    {"Measurement": y, **samples, "Ground truth": x_true},
    subtitles=[f"{psnr(y, x_true).item():.2f} dB"]
    + [f"{psnr(sample, x_true).item():.2f} dB" for sample in samples.values()]
    + [""],
    rescale_mode="clip",
    figsize=(10, 2.8),
)

# %%
# The three samplers recover the image from 30% of its pixels. They differ in their requirements and cost:
#
# - DPS handles any differentiable forward operator, but every step needs a backward pass through the denoiser,
#   and it needs more steps than the other two.
# - DiffPIR handles any data-fidelity term with a proximal operator and needs only forward passes.
# - DDRM is restricted to :class:`deepinv.physics.DecomposablePhysics` with Gaussian noise, and needs only forward
#   passes.
#
# Their hyperparameters (``weight``, ``zeta``, ``eta`` and the number of steps) should be tuned for each problem.

# %%
# Posterior mean and uncertainty
# ------------------------------
#
# Each run of a sampler draws a new sample of the posterior distribution. Averaging several samples estimates the
# posterior mean, which is the minimum mean squared error estimator, and their spread estimates the uncertainty
# of the reconstruction. :class:`deepinv.sampling.DiffusionSampler` turns any of these samplers into a Monte Carlo
# sampler that runs the diffusion several times and computes these statistics. We use DDRM, the fastest of the three
# here, with 10 samples.

sampler = dinv.sampling.DiffusionSampler(ddrm, max_iter=10, verbose=False)
mean, var = sampler(y, physics)

print(f"Posterior mean PSNR: {psnr(mean, x_true).item():.2f} dB")

error = (mean - x_true).abs().sum(dim=1, keepdim=True)  # per pixel absolute error
std = var.sum(dim=1, keepdim=True).sqrt()  # per pixel standard deviation
plot(
    {
        "Measurement": y,
        "Posterior mean": mean,
        "Posterior std": std,
        "Absolute error": error,
        "Ground truth": x_true,
    },
    figsize=(10, 2.5),
)

# %%
# The posterior mean is smoother than a single sample and closer to the ground truth. The posterior standard
# deviation concentrates along the edges of the wings, where the error is also the largest.

# %%
# :References:
#
# .. footbibliography::
