r"""
Noisy data-fidelity terms for diffusion posterior sampling
==========================================================

This example compares three approximations of the measurement-matching term
used by diffusion posterior samplers: Diffusion Posterior Sampling (DPS) :footcite:t:`chung2022diffusion`,
Pseudoinverse-Guided Diffusion Models (PiGDM) :footcite:t:`song2023pseudoinverse`, and Moment Matching :footcite:t:`rozet2024learning`. These
methods belong to the broader family of explicit posterior approximations reviewed in
*A Survey on Diffusion Models for Inverse Problems*
:footcite:t:`daras2024survey`, which also presents other ways of incorporating
measurements into diffusion models.

Here, we focus on the noisy data-fidelity term itself. For a complete DPS
reconstruction, including the diffusion schedule and reverse-time sampling
loop, see :ref:`sphx_glr_auto_examples_sampling_demo_dps.py`. For a tutorial on
assembling a posterior sampler from an SDE, a solver, and a data-fidelity term,
see :ref:`sphx_glr_auto_examples_sampling_demo_diffusion_sde.py`.
"""

# %%
# Posterior sampling and the intractable likelihood
# -------------------------------------------------
#
# For simplicity, we use a Variance-Exploding (VE) diffusion, whose scaling is
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
# For a linear forward model and whitened Gaussian measurement noise, the
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
# We use linear inpainting here. DPS also supports differentiable nonlinear
# forward operators, whereas the PiGDM and Moment Matching implementations
# below require a linear operator.

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
# :math:`p_t(y\mid x_t)\approx p(y\mid D_{\sigma_t}(x_t))`. In practice, DPS
# uses normalized measurement guidance by differentiating the residual norm:
#
# .. math::
#
#     \nabla_{x_t}\log p_t(x_t\mid y)
#     \approx \nabla_{x_t}\log p_t(x_t)
#     - \lambda\nabla_{x_t}
#       \left\|A D_{\sigma_t}(x_t)-y\right\|_2.
#
# By the chain rule, this is the denoiser-Jacobian pullback of the normalized
# measurement residual. Its magnitude is therefore not directly proportional
# to the residual magnitude, preventing large residuals at early, high-noise
# diffusion steps from producing disproportionately large guidance updates.
# This is the residual-norm guidance used in the original DPS implementation.
#
# Equivalently, the Dirac mass can be viewed as a Gaussian whose covariance
# tends to zero. This approximation only requires differentiating the residual
# through the denoiser and the forward operator. It is therefore broadly
# applicable, but it discards all conditional uncertainty of :math:`x_0` given
# :math:`x_t`.
#
# In :class:`deepinv.sampling.DPSDataFidelity`, :math:`\lambda` is the
# ``weight`` parameter. It controls the scale of the data-fidelity contribution
# relative to the unconditional prior score: a larger value enforces the
# measurements more strongly.

dps_weight = 200
dps = dinv.sampling.DPSDataFidelity(denoiser=denoiser, weight=dps_weight)

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
#       (r_t^2 A A^\top + \mathrm I)^{-1}
#       (A D_{\sigma_t}(x_t)-y).
#
# The inverse is evaluated exactly for
# :class:`deepinv.physics.DecomposablePhysics` operators, such as inpainting,
# and with conjugate gradient for other linear operators. Jacobian-vector
# products are computed automatically, without forming :math:`J_D` explicitly.
# The :math:`\lambda` factor is exposed as ``weight`` in
# :class:`deepinv.sampling.PiGDMDataFidelity`; increasing it strengthens the
# data-fidelity contribution relative to the unconditional prior score.

pigdm_weight = 20
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
# isotropic scalar. In our additive Gaussian parametrization, the first- and
# second-order Tweedie formulas give
#
# .. math::
#
#     \mu_t(x_t)
#     &= \mathbb E[x_0\mid x_t]
#      = D_{\sigma_t}(x_t) \\
#     &= x_t + \sigma_t^2
#        \nabla_{x_t}\log p_t(x_t), \\
#     \Sigma_t(x_t)
#     &= \operatorname{Cov}[x_0\mid x_t] \\
#     &= \sigma_t^2 J_D(x_t,\sigma_t) \\
#     &= \sigma_t^2\left(
#          \mathrm I + \sigma_t^2
#          \nabla_{x_t}^2\log p_t(x_t)
#        \right).
#
# Moment Matching explicitly approximates the conditional distribution by the
# anisotropic Gaussian with these two moments:
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
#       (A J_D^\top A^\top + \mathrm I)^{-1}
#       (A D_{\sigma_t}(x_t)-y).
#
# This can capture direction-dependent uncertainty, at the cost of solving a
# denoiser-dependent linear system. DeepInverse evaluates the system with
# conjugate gradient and uses vector-Jacobian products throughout.
# The :math:`\lambda` factor is the ``weight`` parameter of
# :class:`deepinv.sampling.MomentMatchingDataFidelity`; increasing it gives the
# data-fidelity contribution more influence relative to the unconditional
# prior score.

moment_matching_weight = 20
moment_matching = dinv.sampling.MomentMatchingDataFidelity(
    denoiser=denoiser,
    weight=moment_matching_weight,
    cg_max_iter=3,
)

# %%
# Compare the guidance terms
# --------------------------
#
# Calling ``grad`` is enough to compare the three approximations at the same
# :math:`x_t`. Their scales are method-dependent, so ``weight`` should be
# tuned separately in a reconstruction. The plot independently normalizes the
# magnitude of each gradient to emphasize its spatial structure.

data_fidelities = {
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
    {
        f"{name} gradient magnitude": gradient.abs()
        for name, gradient in gradients.items()
    },
    figsize=(9, 3),
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
# Moment Matching is considerably slower on CPU because every diffusion step
# contains an inner conjugate-gradient solve. On CPU, we therefore display a
# precomputed reconstruction; on GPU and MPS devices, we compute it normally.


num_steps = 100
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

posterior_samples = {}
for name, data_fidelity in data_fidelities.items():
    if name == "Moment Matching" and device.type == "cpu":
        precomputed_sample = dinv.utils.load_url_image(
            "https://huggingface.co/deepinv/demo/resolve/main/"
            "moment_matching.png",
            device=device,
            dtype=x_true.dtype,
        )
        posterior_samples[name] = precomputed_sample[:, : x_true.shape[1]]
        continue

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

dinv.utils.plot(
    {
        "Ground truth": x_true,
        "Measurement": y,
        **{
            f"{name} posterior sample": sample
            for name, sample in posterior_samples.items()
        },
    },
    figsize=(15, 3),
    save_dir='output'
)

# %%
# :References:
#
# .. footbibliography::
