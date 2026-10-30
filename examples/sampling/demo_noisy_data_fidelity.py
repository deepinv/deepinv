r"""
Noisy data-fidelity terms for diffusion posterior sampling
==========================================================

This example compares eight approximations of the measurement-matching term
used by diffusion posterior samplers.

Three of them measure the mismatch directly at the noisy iterate :math:`x_t`, and
therefore need no denoiser evaluation at all:

- Score-based Annealed Langevin Dynamics (Score-ALD) :footcite:t:`jalal2021robust`,
- Score-SDE :footcite:t:`song2020score`,
- Iterative Latent Variable Refinement (ILVR) :footcite:t:`choi2021ilvr`.

Two first denoise :math:`x_t` and enforce measurement consistency on the denoised
estimate :math:`D_{\sigma_t}(x_t)` in closed form (in the SVD domain or via a
proximal operator), without differentiating through the denoiser:

- Denoising Diffusion Restoration Models (DDRM) :footcite:t:`kawar2022denoising`,
- Diffusion PnP Image Restoration (DiffPIR) :footcite:t:`zhu2023denoising`.

The remaining three first denoise :math:`x_t` and backpropagate the residual
through the denoiser, differing in how much of the conditional uncertainty they
keep:

- Diffusion Posterior Sampling (DPS) :footcite:t:`chung2022diffusion`,
- Pseudoinverse-Guided Diffusion Models (PiGDM) :footcite:t:`song2023pseudoinverse`,
- Moment Matching :footcite:t:`rozet2024learning`.

We follow the presentation of *A Survey on Diffusion Models for Inverse Problems*
:footcite:t:`daras2024survey`, which reviews these and other ways of incorporating
measurements into diffusion models.

Here, we focus on the noisy data-fidelity term itself. For complete DPS, DiffPIR
and DDRM reconstructions, see :ref:`sphx_glr_auto_examples_sampling_demo_diffusion_samplers.py`.
For a tutorial on assembling a posterior sampler from an SDE, a solver, and a
data-fidelity term, see :ref:`sphx_glr_auto_examples_sampling_demo_diffusion_sde.py`.
"""

# %%
# Posterior sampling and the intractable likelihood
# -------------------------------------------------
#
# We use a Variance-Exploding (VE) diffusion, whose scaling is
# :math:`s(t)=1`:
#
# .. math::
#
#     x_t = x_0 + \sigma_t\omega,
#     \qquad \omega\sim\mathcal N(0,\mathrm I).
#
# By Bayes' rule, the conditional score is
#
# .. math::
#
#     \nabla_{x_t}\log p_t(x_t\mid y)
#     = \nabla_{x_t}\log p_t(x_t)
#     + \nabla_{x_t}\log p_t(y\mid x_t).
#
# A diffusion denoiser :math:`D_\sigma` estimates the unconditional part using
# Tweedie's formula. The second term is harder because
#
# .. math::
#
#     p_t(y\mid x_t)
#     = \int p(y\mid x_0)p(x_0\mid x_t)\,\mathrm d x_0
#
# is generally intractable. The goal of the methods below is to replace
# :math:`p(x_0\mid x_t)` by a Gaussian approximation
#
# .. math::
#
#     p(x_0\mid x_t)
#     \approx \mathcal N\!\left(
#         x_0;D_{\sigma_t}(x_t),\Sigma_t(x_t)
#     \right).
#
# For a linear forward model and Gaussian measurement noise, the
# integral of the two Gaussian densities is available in closed form:
#
# .. math::
#
#     p_t(y\mid x_t)
#     \approx \mathcal N\!\left(
#         y;A D_{\sigma_t}(x_t),
#         A\Sigma_t(x_t) A^\top + \mathrm I
#     \right).
#
# The approximations differ primarily in their choice of
# :math:`\Sigma_t(x_t)`.
# DPS is the degenerate, zero-covariance case; PiGDM uses an isotropic
# covariance; and Moment Matching estimates a structured covariance with the
# second-order Tweedie formula. A
# :class:`deepinv.sampling.NoisyDataFidelity` approximates the gradient of the
# resulting negative log-likelihood. Consequently,
# :class:`deepinv.sampling.PosteriorDiffusion` subtracts the gradient returned
# by the data-fidelity object from the unconditional score.
#
# The word *noisy* here refers to the diffusion-corrupted variable
# :math:`x_t`; the measurements themselves do not have to be noisy.

# %%
# Create one inverse problem and one noisy diffusion state
# --------------------------------------------------------
#
# We use linear inpainting here. DPS and DiffPIR (given a proximal operator)
# also support nonlinear forward operators, whereas PiGDM and Moment Matching
# require a linear operator, and DDRM requires a decomposable linear operator
# (:class:`deepinv.physics.DecomposablePhysics`).

import matplotlib.pyplot as plt
import torch

import deepinv as dinv

device = dinv.utils.get_device()
dtype = torch.float32 if "mps" in str(device) else torch.float64

x_true = dinv.utils.load_example(
    "FFHQ_example.png", img_size=64, resize_mode="resize", device=device
)
mask = torch.ones_like(x_true)
mask[..., 24:40, 24:40] = 0.0
measurement_noise = 0.05
physics = dinv.physics.Inpainting(
    img_size=x_true.shape[1:],
    mask=mask,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=measurement_noise),
)
y = physics(x_true)

sigma_t = 0.15
rng = torch.Generator(device=device).manual_seed(0)
x_t = x_true + sigma_t * torch.randn(
    x_true.shape, generator=rng, device=device, dtype=x_true.dtype
)

# We use the same FFHQ NCSNpp denoiser for every approximation.
denoiser = dinv.models.NCSNpp(pretrained="download").to(device)
with torch.no_grad():
    x_0_denoised = denoiser(x_t, sigma_t)

dinv.utils.plot(
    {
        "Ground truth": x_true,
        "Measurement": y,
        r"Noisy $x_t$": x_t,
        r"Denoised $D_{\sigma_t}(x_t)$": x_0_denoised,
    },
    figsize=(12, 3),
)

# %%
# Score-ALD: guide with the noisy iterate itself
# ----------------------------------------------
#
# The cheapest option is not to denoise at all. Score-ALD
# :footcite:t:`jalal2021robust` evaluates the measurement mismatch directly at
# :math:`x_t`, and compensates for the fact that :math:`x_t` is noisy by
# inflating the measurement noise variance with an annealing parameter
# :math:`\gamma_t`:
#
# .. math::
#
#     p_t(y\mid x_t)
#     \approx \mathcal N\!\left(
#         y; A x_t, (\sigma_y^2+\gamma_t^2)\mathrm I
#     \right),
#
# which gives the guidance
#
# .. math::
#
#     \nabla_{x_t}\log p_t(x_t\mid y)
#     \approx \nabla_{x_t}\log p_t(x_t)
#       - \lambda \frac{A^\top (A x_t - y)}{\sigma_y^2+\gamma_t^2}.
#
# By default :math:`\gamma_t=\sigma_t`, so the guidance is weak early in the
# diffusion, when :math:`x_t` is mostly noise, and strengthens as
# :math:`\sigma_t\to 0`. Because that denominator already sets the scale,
# ``weight=1`` is the natural choice here.

ald = dinv.sampling.ALDDataFidelity(weight=1.0)

# %%
# Score-SDE and ILVR: noise the measurements instead
# --------------------------------------------------
#
# Score-SDE :footcite:t:`song2020score` makes the same comparison, but first
# lifts the measurements to the current noise level,
#
# .. math::
#
#     y_t = y + \sigma_t\epsilon,
#     \qquad \epsilon\sim\mathcal N(0,\mathrm I),
#
# so that :math:`y_t` and :math:`A x_t` are corrupted by comparable amounts of
# noise. ILVR :footcite:t:`choi2021ilvr` differs only in how the measurement
# residual is lifted back to the image space: it uses the pseudo-inverse
# :math:`A^\dagger` rather than the adjoint :math:`A^\top`,
#
# .. math::
#
#     \nabla_{x_t}\log p_t(x_t\mid y)
#     \approx \nabla_{x_t}\log p_t(x_t)
#       - \lambda \frac{A^\dagger (A x_t - y_t)}{\sigma_y^2+\gamma_t^2}.
#

score_sde = dinv.sampling.ScoreSDEDataFidelity(weight=1.0)
ilvr = dinv.sampling.ILVRDataFidelity(weight=1.0)

# %%
# DDRM and DiffPIR: update the denoised estimate in closed form
# -------------------------------------------------------------
#
# Rather than evaluating the measurement residual at the noisy iterate
# :math:`x_t`, DDRM :footcite:t:`kawar2022denoising` and DiffPIR
# :footcite:t:`zhu2023denoising` first compute the denoised estimate
# :math:`D_{\sigma_t}(x_t)` and enforce measurement consistency on that clean
# estimate without differentiating through the denoiser.
#
# For a decomposable linear operator :math:`A = U \Sigma V^\top`
# (:class:`deepinv.physics.DecomposablePhysics`), DDRM
# (:class:`deepinv.sampling.DDRMDataFidelity`) works in the spectral domain:
# it first moves the measured singular components of :math:`x_t` toward the
# measurements, giving :math:`\tilde{x}_t`, then corrects the denoised estimate
# with a diagonal weight :math:`W`:
#
# .. math::
#
#     \nabla_{x_t}\log p_t(x_t\mid y)
#     \approx \nabla_{x_t}\log p_t(x_t)
#       - \lambda V \Sigma^\top W (\sigma_y^2 \mathrm I + \sigma_t^2 \Sigma \Sigma^\top)^{-1}
#       (\Sigma V^\top D_{\sigma_t}(\tilde{x}_t) - U^\top y),
#
# where the prior score is also evaluated from :math:`D_{\sigma_t}(\tilde{x}_t)`.
#
# DiffPIR (:class:`deepinv.sampling.DiffPIRDataFidelity`) instead applies the
# proximal operator of a measurement data-fidelity function :math:`f(\cdot, y)`
# to :math:`D_{\sigma_t}(x_t)` with step size
# :math:`\rho_t^{-1} = \sigma_t^2 / (\lambda \sigma_y^2)`, and plugs the
# resulting estimate back into Tweedie's formula:
#
# .. math::
#
#     \nabla_{x_t}\log p_t(x_t\mid y)
#     \approx \nabla_{x_t}\log p_t(x_t)
#       - \frac{D_{\sigma_t}(x_t) - \operatorname{prox}_{f(\cdot, y) / \rho_t}\!\left(D_{\sigma_t}(x_t)\right)}{\sigma_t^2}.
#
# When paired with :class:`deepinv.sampling.PosteriorDiffusion` and an
# :class:`deepinv.sampling.EDMDiffusionSDE`, both methods reuse the denoiser
# evaluation performed inside the data-fidelity term to compute the
# unconditional prior score, so each diffusion step still requires only one
# forward pass through the denoiser and no backward pass.

ddrm = dinv.sampling.DDRMDataFidelity(denoiser=denoiser, weight=1.0)
diffpir = dinv.sampling.DiffPIRDataFidelity(
    denoiser=denoiser, data_fidelity=dinv.optim.L2(), weight=1.0
)

# %%
# DPS: plug in the denoised posterior mean
# ----------------------------------------
#
# DPS :footcite:t:`chung2022diffusion` approximates the conditional
# distribution by a Dirac mass at the denoised posterior mean:
#
# .. math::
#
#     p(x_0\mid x_t)
#     \approx \delta\!\left(
#         x_0-D_{\sigma_t}(x_t)
#     \right),
#     \qquad
#     D_{\sigma_t}(x_t)
#     \simeq \mathbb E[x_0\mid x_t].
#
# This is the degenerate Gaussian approximation
# :math:`\Sigma_t(x_t)=0`. Inserting it into the integral gives
# :math:`p_t(y\mid x_t)\approx p(y\mid D_{\sigma_t}(x_t))`.
# Note that this is equivalent to differentiating the residual norm:
#
# .. math::
#
#     \nabla_{x_t}\log p_t(x_t\mid y)
#     \approx \nabla_{x_t}\log p_t(x_t)
#     - \lambda\nabla_{x_t}
#       \left\|A D_{\sigma_t}(x_t)-y\right\|_2.
#
# In :class:`deepinv.sampling.DPSDataFidelity`, :math:`\lambda` is the
# ``weight`` parameter. It controls the scale of the data-fidelity contribution
# relative to the unconditional prior score: a larger value enforces the
# measurements more strongly.
#
# The residual norm above carries no noise variance, so :math:`\lambda` has to
# absorb a factor of order :math:`\|A D_{\sigma_t}(x_t)-y\|/\sigma_y^2`, in the
# hundreds here. To keep ``weight`` on the same scale as the other terms, we use
# ``guidance="annealed"``, which differentiates the Gaussian negative
# log-likelihood with the annealed variance :math:`\sigma_y^2+\sigma_t^2` instead:
#
# .. math::
#
#     \nabla_{x_t}\log p_t(x_t\mid y)
#     \approx \nabla_{x_t}\log p_t(x_t)
#     - \lambda\nabla_{x_t}
#       \frac{\left\|A D_{\sigma_t}(x_t)-y\right\|_2^2}{2(\sigma_y^2+\sigma_t^2)}.
#
# Pass ``guidance="norm"`` (the default) for the residual norm of the original
# paper.

dps = dinv.sampling.DPSDataFidelity(denoiser=denoiser, weight=1.0, guidance="annealed")

# %%
# PiGDM: use an isotropic covariance approximation
# ------------------------------------------------
#
# PiGDM :footcite:t:`song2023pseudoinverse` retains conditional uncertainty but
# approximates it with an isotropic Gaussian:
#
# .. math::
#
#     p(x_0\mid x_t)
#     \approx \mathcal N\!\left(
#         x_0;D_{\sigma_t}(x_t),\Sigma_t(x_t)
#     \right),
#     \qquad
#     \Sigma_t(x_t)=r_t^2\mathrm I,
#     \qquad
#     r_t^2=\frac{\sigma_t^2}{1+\sigma_t^2}.
#
# Writing :math:`J_D` for the denoiser Jacobian, the resulting gradient is
#
# .. math::
#
#     \nabla_{x_t}\log p_t(x_t\mid y)
#     \approx \nabla_{x_t}\log p_t(x_t)
#       - \lambda J_D^\top A^\top
#       (r_t^2 A A^\top + \sigma_y^2\mathrm I)^{-1}
#       (A D_{\sigma_t}(x_t)-y),
#
# where :math:`\sigma_y` is the standard deviation of the measurement noise,
# read from ``physics.noise_model.sigma``.
#
# The inverse is evaluated exactly for
# :class:`deepinv.physics.DecomposablePhysics` operators
# and with conjugate gradient for other linear operators. Jacobian-vector
# products are computed automatically, without forming :math:`J_D` explicitly.
# The :math:`\lambda` factor is exposed as ``weight`` in
# :class:`deepinv.sampling.PiGDMDataFidelity`; increasing it strengthens the
# data-fidelity contribution relative to the unconditional prior score.

pigdm_weight = 1
pigdm = dinv.sampling.PiGDMDataFidelity(
    denoiser=denoiser,
    weight=pigdm_weight,
    cg_max_iter=10,
)

# %%
# Moment Matching: retain a structured covariance
# ------------------------------------------------
#
# Moment Matching :footcite:t:`rozet2024learning` uses the denoiser Jacobian to
# approximate the conditional covariance rather than replacing it by an
# isotropic scalar. In our additive Gaussian parametrization, the
# second-order Tweedie formulas give
#
# .. math::
#
#     \Sigma_t(x_t) = \operatorname{Cov}[x_0\mid x_t] = \sigma_t^2 J_D(x_t,\sigma_t)
#
# Moment Matching explicitly approximates the conditional distribution by the
# anisotropic Gaussian with this covariance:
#
# .. math::
#
#     p(x_0\mid x_t)
#     \approx \mathcal N\!\left(
#         x_0;D_{\sigma_t}(x_t),\Sigma_t(x_t)
#     \right).
#
# The resulting gradient is
#
# .. math::
#
#     \nabla_{x_t}\log p_t(x_t\mid y)
#     \approx \nabla_{x_t}\log p_t(x_t)
#       - \lambda J_D^\top A^\top
#       (\sigma_t^2 A J_D A^\top + \sigma_y^2\mathrm I)^{-1}
#       (A D_{\sigma_t}(x_t)-y).
#
# This can capture direction-dependent uncertainty, at the cost of solving a
# denoiser-dependent linear system. DeepInverse evaluates the system with
# conjugate gradient and uses vector-Jacobian products throughout.
# The :math:`\lambda` factor is the ``weight`` parameter of
# :class:`deepinv.sampling.MomentMatchingDataFidelity`; increasing it gives the
# data-fidelity contribution more influence relative to the unconditional
# prior score.

moment_matching_weight = 1
moment_matching = dinv.sampling.MomentMatchingDataFidelity(
    denoiser=denoiser,
    weight=moment_matching_weight,
    cg_max_iter=3,
)

# %%
# Compare the guidance terms
# --------------------------
#
# Calling ``grad`` is enough to compare the eight approximations at the same
# :math:`x_t`. Their scales are method-dependent, so ``weight`` should be
# tuned separately in a reconstruction. The plot independently normalizes the
# magnitude of each gradient to emphasize its spatial structure.
#
# The three families look different:
#
# Score-ALD, Score-SDE and ILVR compare the *noisy* iterate to the measurements.
# Here :math:`x_t = x_0 + \sigma_t\omega` is built from the ground truth, so
#
# .. math::
#
#     A x_t - y = A(x_0 + \sigma_t\omega) - (A x_0 + \sigma_y\eta)
#               = \sigma_t A\omega - \sigma_y\eta :
#
# the signal cancels exactly, and the residual is *pure noise*. Their guidance
# maps therefore look like noise, and they vanish on the masked pixels, where
# :math:`A^\top` and :math:`A^\dagger` are zero.
#
# DDRM and DiffPIR compare a *denoised* estimate to the measurements and map the
# residual back to image space via the SVD or the proximal operator, without
# differentiating through the denoiser. Their residual therefore reflects
# reconstruction error rather than diffusion noise, while still vanishing on the
# masked pixels at a single step.
#
# DPS, PiGDM and Moment Matching also compare the *denoised* estimate to the
# measurements, and backpropagate the residual through the denoiser Jacobian,
# which spreads the guidance over the whole image, including inside the mask.

data_fidelities = {
    "Score-ALD": ald,
    "Score-SDE": score_sde,
    "ILVR": ilvr,
    "DDRM": ddrm,
    "DiffPIR": diffpir,
    "DPS": dps,
    "PiGDM": pigdm,
    "Moment Matching": moment_matching,
}

gradients = {}
for name, data_fidelity in data_fidelities.items():
    gradient = data_fidelity.grad(x_t.clone(), y=y, physics=physics, sigma=sigma_t)
    gradients[name] = gradient
    norm = torch.linalg.vector_norm(gradient).item()
    print(f"{name:>15s} gradient norm: {norm:.3e}")

dinv.utils.plot(
    {f"{name}": gradient.abs() for name, gradient in gradients.items()},
    figsize=(18, 3),
    suptitle="Gradient magnitude",
)

# %%
# Posterior sampling experiment
# -----------------------------
#
# The noisy data-fidelity object is one interchangeable component of
# :class:`deepinv.sampling.PosteriorDiffusion`. We now hold the denoiser, VE
# diffusion, Euler solver, measurements, and random seed fixed, and change only
# the noisy data-fidelity approximation. Using the same seed gives every method
# the same initial noise and Brownian increments.
#
# All eight use ``weight=1``: every term is normalized by a guidance strength of
# order :math:`\sigma_y^2+\sigma_t^2`, so a single scale works for all of them.
#
# DDRM and DiffPIR are the exception to the shared solver. Their correction does
# not go through the denoiser Jacobian, so it also carries the measurement noise.
# Their samplers discard it: each step moves to the corrected estimate and renews
# a fraction :math:`\zeta` of the noise variance, so that the next denoising
# removes it. The Euler solver instead accumulates the correction along the
# trajectory, and the samples keep part of the measurement noise. We therefore
# sample them with :class:`deepinv.sampling.AncestralSolver`, with
# :math:`\zeta=\eta^2` for DDRM and :math:`\zeta=1` for DiffPIR, as
# :class:`deepinv.sampling.DDRM` and :class:`deepinv.sampling.DiffPIR` do.
#
# Moment Matching is considerably slower on CPU because every diffusion step
# contains an inner conjugate-gradient solve. On CPU, we therefore display a
# precomputed reconstruction; on GPU and MPS devices, we compute it normally.


num_steps = 50
timesteps = torch.linspace(
    1.0,
    0.001,
    num_steps,
    device=device,
    dtype=dtype,
)

sde = dinv.sampling.VarianceExplodingDiffusion(
    alpha=0.25,
    device=device,
    dtype=dtype,
)

ancestral_zeta = {"DDRM": ddrm.eta**2, "DiffPIR": 1.0}

posterior_samples = {}
for name, data_fidelity in data_fidelities.items():
    if name == "Moment Matching" and device.type == "cpu":
        precomputed_sample = dinv.utils.load_url_image(
            "https://huggingface.co/deepinv/demo/resolve/main/" "moment_matching.png",
            device=device,
            dtype=x_true.dtype,
        )
        posterior_samples[name] = precomputed_sample[:, : x_true.shape[1]]
        continue

    if name in ancestral_zeta:
        solver = dinv.sampling.AncestralSolver(
            timesteps=timesteps,
            zeta=ancestral_zeta[name],
            rng=torch.Generator(device=device),
        )
    else:
        solver = dinv.sampling.EulerSolver(
            timesteps=timesteps,
            rng=torch.Generator(device=device),
        )
    posterior_sampler = dinv.sampling.PosteriorDiffusion(
        data_fidelity=data_fidelity,
        denoiser=denoiser,
        sde=sde,
        solver=solver,
        device=device,
        dtype=dtype,
        verbose=False,
    )

    with torch.no_grad():
        posterior_samples[name] = posterior_sampler(
            y=y,
            physics=physics,
            seed=1,
            denoise_output=True,
        ).clip(0.0, 1.0)

# Two rows of five panels, each preceded by a reference image.
rows = [
    {
        "Ground truth": x_true,
        **{
            name: posterior_samples[name]
            for name in ("Score-ALD", "Score-SDE", "ILVR", "DDRM")
        },
    },
    {
        "Measurement": y,
        **{
            name: posterior_samples[name]
            for name in ("DiffPIR", "DPS", "PiGDM", "Moment Matching")
        },
    },
]

fig, axs = plt.subplots(
    len(rows), 5, figsize=(15, 6.5), squeeze=False, layout="compressed"
)
for row, images in zip(axs, rows, strict=True):
    dinv.utils.plot(images, fig=fig, axs=row[None, :], show=False)
plt.show()

# %%
# All eight recover the masked region, at very different costs: Score-ALD,
# Score-SDE and ILVR need no denoiser evaluation for the guidance term, DDRM
# and DiffPIR reuse the forward denoiser evaluation without differentiating
# through it, DPS needs one backward pass through the denoiser, and PiGDM and
# Moment Matching need a linear solve on top.

# %%
# :References:
#
# .. footbibliography::
