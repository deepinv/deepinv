r"""
Poisson-Gaussian Denoising with the Generalized Anscombe Transform
==================================================================

This example demonstrates how to denoise images corrupted by
:class:`Poisson-Gaussian noise <deepinv.physics.PoissonGaussianNoise>` using the
:class:`Generalized Anscombe Transform (GAT) <deepinv.models.AnscombeDenoiserWrapper>`, which
converts any Gaussian denoiser into a Poisson-Gaussian denoiser.

We compare two approaches on a butterfly image:

* **DRUNet baseline** — the pretrained :class:`DRUNet <deepinv.models.DRUNet>` Gaussian
  denoiser applied directly to the noisy measurement, using a global noise-level heuristic.
* **Anscombe + DRUNet** — the same DRUNet wrapped with
  :class:`AnscombeDenoiserWrapper <deepinv.models.AnscombeDenoiserWrapper>`, which first
  variance-stabilizes the heteroscedastic Poisson-Gaussian noise via the GAT.

Background
----------

The **Poisson-Gaussian noise model** arises naturally in photon-counting imaging systems
(fluorescence microscopy, astronomy, low-light photography), where the observed
measurement :math:`y` satisfies

.. math::

    y = \gamma \, z + \varepsilon, \qquad z \sim \mathcal{P}\!\left(\tfrac{x}{\gamma}\right),\;
    \varepsilon \sim \mathcal{N}(0, \sigma^2 I),

with photon **gain** :math:`\gamma > 0` and read-out noise level :math:`\sigma \geq 0`.

The **Generalized Anscombe Transform (GAT)** :footcite:t:`Makitalo2012` stabilizes the
variance of the Poisson-Gaussian noise to approximately :math:`\gamma^2`:

.. math::

    z = 2\sqrt{\gamma y + \tfrac{3}{8}\gamma^2 + \sigma^2}.

After applying the GAT the signal approximately follows :math:`\mathcal{N}(\cdot, \gamma^2)`,
so any Gaussian denoiser trained at noise level :math:`\sigma_d` can be applied after
rescaling :math:`z` by :math:`\sigma_d / \gamma`.  The **inverse GAT** maps back to the
original domain.  Setting :math:`u = z / \gamma`, it reads:

.. math::

    \hat{y} = \gamma \left(
              \frac{1}{4}u^2
              + \frac{1}{4}\sqrt{\tfrac{3}{2}}\,u^{-1}
              - \frac{11}{8}u^{-2}
              + \frac{5}{8}\sqrt{\tfrac{3}{2}}\,u^{-3}
              - \frac{1}{8}
              + \frac{\sigma^2}{\gamma^2}
              \right), \qquad u = \frac{z}{\gamma}.

The full pipeline of :class:`AnscombeDenoiserWrapper <deepinv.models.AnscombeDenoiserWrapper>`
reads:

.. math::

    \hat{x} = \mathrm{IGAT}\!\left(\denoisername\!\left(z,\; \gamma\right)\right), \qquad z = \mathrm{GAT}(y).

"""

# %%
import torch
import deepinv as dinv
from deepinv.models import AnscombeDenoiserWrapper, DRUNet, PatchCovarianceNoiseEstimator
from deepinv.utils import load_example

device = dinv.utils.get_device()
torch.manual_seed(0)

# %%
# Load a clean butterfly image
# ----------------------------

x = load_example("butterfly.png", device=device)

# %%
# Define the noise parameters
# ---------------------------
#
# We consider two scenarios:
#
# 1. **Pure Poisson noise**: :math:`\sigma \approx 0`, moderate photon gain :math:`\gamma`.
# 2. **Mixed Poisson-Gaussian noise**: both :math:`\gamma > 0` and :math:`\sigma > 0`.

gain = 0.1       # photon gain γ
sigma_pg = 0.01  # Gaussian read-out noise σ

# Scenario 1 – pure Poisson (σ set to a small positive value for numerical stability)
physics_poisson = dinv.physics.Denoising(
    dinv.physics.PoissonGaussianNoise(gain=gain, sigma=1e-6, clip_positive=True),
    device=device,
)

# Scenario 2 – mixed Poisson-Gaussian
physics_pg = dinv.physics.Denoising(
    dinv.physics.PoissonGaussianNoise(gain=gain, sigma=sigma_pg, clip_positive=True),
    device=device,
)

y_poisson = physics_poisson(x)
y_pg = physics_pg(x)

with torch.no_grad():
    z = dinv.models.anscombe.generalized_anscombe_transform(y_pg, gain=gain, sigma=sigma_pg)
    sigma_est = PatchCovarianceNoiseEstimator()(z)
    print(f"Estimated noise level after Anscombe transform: {sigma_est.item():.4f}. Gain = {gain:.4f}")

# %%
# Set up the Anscombe-wrapped DRUNet denoiser
# -------------------------------------------
#
# :class:`DRUNet <deepinv.models.DRUNet>` is a powerful Gaussian denoiser.
# We wrap it with :class:`AnscombeDenoiserWrapper <deepinv.models.AnscombeDenoiserWrapper>`
# to lift it to the Poisson-Gaussian domain. The GAT output has standard deviation
# approximately :math:`\gamma`, so the wrapper calls DRUNet at noise level :math:`\gamma`.

drunet = DRUNet(device=device)
anscombe_denoiser = AnscombeDenoiserWrapper(drunet)

# %%
# Set up the plain DRUNet baseline (no Anscombe transform)
# --------------------------------------------------------
#
# As a baseline we apply DRUNet **directly** to the noisy measurement, using a
# global noise-level heuristic derived from the Poisson-Gaussian variance.
# For Poisson-Gaussian noise, :math:`\mathrm{Var}(y_i) \approx \gamma \bar{x} + \sigma^2`,
# so a reasonable single-number estimate of the noise standard deviation is:
#
# .. math::
#
#     \sigma_{\mathrm{approx}} = \sqrt{\bar{y} \cdot \gamma + \sigma^2},
#
# where :math:`\bar{y}` is the mean intensity of the noisy observation (used as a proxy
# for :math:`\bar{x}`).  For pure Poisson noise (:math:`\sigma = 0`) this reduces to
# :math:`\sigma_{\mathrm{approx}} = \sqrt{\bar{y} \cdot \gamma}`.
# This heuristic ignores the spatial heteroscedasticity of the Poisson variance
# (brighter pixels are noisier), which the Anscombe transform explicitly corrects for.

# %%
# Denoise — pure Poisson scenario
# --------------------------------
#
# For pure Poisson noise we set :math:`\sigma \approx 0` and pass only
# :math:`\gamma` to both denoisers.

with torch.no_grad():
    # Anscombe + DRUNet
    x_hat_anscombe_poisson = anscombe_denoiser(y_poisson, gain=gain, sigma=1e-6)

    # Plain DRUNet with approximate σ = sqrt(ȳ · γ)
    sigma_approx_poisson = (y_poisson.mean() * gain) ** 0.5
    x_hat_drunet_poisson = drunet(y_poisson, sigma_approx_poisson)

# %%
# Denoise — mixed Poisson-Gaussian scenario
# ------------------------------------------

with torch.no_grad():
    # Anscombe + DRUNet
    x_hat_anscombe_pg = anscombe_denoiser(y_pg, gain=gain, sigma=sigma_pg)

    # Plain DRUNet with approximate σ = sqrt(ȳ · γ + σ²)
    sigma_approx_pg = (y_pg.mean() * gain + sigma_pg ** 2) ** 0.5
    x_hat_drunet_pg = drunet(y_pg, sigma_approx_pg)

# %%
# Compute PSNR metrics
# --------------------

psnr = dinv.metric.PSNR()

psnr_noisy_poisson    = psnr(y_poisson, x).item()
psnr_anscombe_poisson = psnr(x_hat_anscombe_poisson, x).item()
psnr_drunet_poisson   = psnr(x_hat_drunet_poisson, x).item()

psnr_noisy_pg    = psnr(y_pg, x).item()
psnr_anscombe_pg = psnr(x_hat_anscombe_pg, x).item()
psnr_drunet_pg   = psnr(x_hat_drunet_pg, x).item()

# %%
# Visualize results — pure Poisson noise
# ----------------------------------------

dinv.utils.plot(
    [x, y_poisson, x_hat_drunet_poisson, x_hat_anscombe_poisson],
    titles=[
        "Ground truth",
        f"Noisy\n({psnr_noisy_poisson:.2f} dB)",
        f"DRUNet\n({psnr_drunet_poisson:.2f} dB)",
        f"DRUNet+Ans.\n({psnr_anscombe_poisson:.2f} dB)",
    ],
    rescale_mode='clip'
)

# %%
# Visualize results — mixed Poisson-Gaussian noise
# -------------------------------------------------

dinv.utils.plot(
    [x, y_pg, x_hat_drunet_pg, x_hat_anscombe_pg],
    titles=[
        "Ground truth",
        f"Noisy\n({psnr_noisy_pg:.2f} dB)",
        f"DRUNet\n({psnr_drunet_pg:.2f} dB)",
        f"DRUNet+Ans.\n({psnr_anscombe_pg:.2f} dB)",
    ]
)

# %%
# Print PSNR summary
# ------------------

print(f"\nPure Poisson  (γ={gain}):")
print(f"  Noisy          : {psnr_noisy_poisson:.2f} dB")
print(f"  DRUNet (approx): {psnr_drunet_poisson:.2f} dB")
print(f"  Anscombe+DRUNet: {psnr_anscombe_poisson:.2f} dB")

print(f"\nMixed Poisson-Gaussian  (γ={gain}, σ={sigma_pg}):")
print(f"  Noisy          : {psnr_noisy_pg:.2f} dB")
print(f"  DRUNet (approx): {psnr_drunet_pg:.2f} dB")
print(f"  Anscombe+DRUNet: {psnr_anscombe_pg:.2f} dB")

# %%
# Conclusion
# ----------
#
# Both methods successfully suppress the Poisson-Gaussian noise.
#
# * :class:`AnscombeDenoiserWrapper <deepinv.models.AnscombeDenoiserWrapper>` is a
#   **zero-cost upgrade**: wrap any off-the-shelf Gaussian denoiser to handle
#   Poisson-Gaussian noise without re-training, by properly stabilizing the
#   heteroscedastic Poisson variance before denoising.
# * Applying DRUNet **directly** with the heuristic
#   :math:`\sigma_{\mathrm{approx}} = \sqrt{\bar{y}\cdot\gamma + \sigma^2}` treats
#   the noise as spatially uniform, ignoring the fact that Poisson variance grows
#   with signal intensity. This mismatch can lead to over-smoothing of bright regions
#   and under-smoothing of dark regions compared to the Anscombe approach.
#
# :References:
#
# .. footbibliography::


