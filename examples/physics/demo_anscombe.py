r"""
Poisson-Gaussian Denoising with the Generalized Anscombe Transform
==================================================================

This example demonstrates how to denoise images corrupted by
:class:`Poisson-Gaussian noise <deepinv.physics.PoissonGaussianNoise>` using the
:class:`Generalized Anscombe Transform (GAT) <deepinv.models.AnscombeDenoiser>`, which
converts any Gaussian denoiser into a Poisson-Gaussian denoiser, :footcite:p:`makitalo2012optimal`.

We compare three approaches on a butterfly image:

* **DRUNet baseline** -- the pretrained :class:`DRUNet <deepinv.models.DRUNet>` Gaussian
  denoiser applied directly to the noisy measurement, using a global noise-level heuristic.
* **DRUNet with spatial noise maps** -- the pretrained :class:`DRUNet <deepinv.models.DRUNet>`
  conditioned on spatial noise levels, which capture the variance of the Poisson-Gaussian noise at each pixel.
* **Anscombe + DRUNet** -- the same DRUNet wrapped with
  :class:`AnscombeDenoiser <deepinv.models.AnscombeDenoiser>`, which first
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

The **Generalized Anscombe Transform (GAT)** stabilizes the
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
              - \frac{\sigma^2}{\gamma^2}
              \right), \qquad u = \frac{z}{\gamma}.

The full pipeline of :class:`AnscombeDenoiser <deepinv.models.AnscombeDenoiser>`
reads:

.. math::

    \hat{x} = \mathrm{IGAT}\!\left(\denoisername\!\left(z,\; \gamma\right)\right), \qquad z = \mathrm{GAT}(y).

"""

# %%
import torch
import deepinv as dinv
from deepinv.models import AnscombeDenoiser, DRUNet, PatchCovarianceNoiseEstimator
from deepinv.utils import load_example

device = dinv.utils.get_device()
torch.manual_seed(0)

# %%
# Load a clean butterfly image
# ----------------------------

x = load_example("butterfly.png", device=device, grayscale=True)

# %%
# Case 1: Poisson noise
# ---------------------
#
# In the **pure Poisson** model, the observed measurement :math:`y` satisfies
#
# .. math::
#
#     y = \gamma \, z, \qquad z \sim \mathcal{P}\!\left(\tfrac{x}{\gamma}\right),
#
# where :math:`\mathcal{P}(\lambda)` denotes a Poisson random variable with rate
# :math:`\lambda`, :math:`x \geq 0` is the clean image, and :math:`\gamma > 0`
# is the photon gain.  The variance of :math:`y` is signal-dependent:
# :math:`\mathrm{Var}(y_i) = \gamma \, x_i`, making the noise **heteroscedastic**.

gain = 0.3  # photon gain gamma

physics_poisson = dinv.physics.Denoising(
    dinv.physics.PoissonNoise(gain=gain, clip_positive=True),
    device=device,
)

y_poisson = physics_poisson(x)

# %%
# Verifying variance stabilization with the Anscombe transform
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# A key property of the GAT is that, after applying it to a Poisson-noisy image,
# the resulting signal is approximately Gaussian with a **constant** standard
# deviation equal to the gain :math:`\gamma`.  We can verify this empirically by
# running a blind Gaussian noise-level estimator
# (:class:`PatchCovarianceNoiseEstimator <deepinv.models.PatchCovarianceNoiseEstimator>`)
# on the transformed image: the estimated :math:`\hat{\sigma}` should be close to
# :math:`\gamma`.

with torch.no_grad():
    z = dinv.models.anscombe.generalized_anscombe_transform(
        y_poisson, gain=gain, sigma=0.0
    )
    sigma_est = PatchCovarianceNoiseEstimator()(z)
    print(
        f"Estimated noise level after Anscombe transform: {sigma_est.item():.4f}  (expected ≈ gain = {gain:.4f})"
    )

dinv.utils.plot([y_poisson, z], ["Noisy", "GAT(Noisy)"], rescale_mode="clip")

# %%
# Set up the Anscombe-wrapped DRUNet denoiser
# -------------------------------------------
#
# :class:`DRUNet <deepinv.models.DRUNet>` is a powerful Gaussian denoiser.
# We wrap it with :class:`AnscombeDenoiser <deepinv.models.AnscombeDenoiser>`
# to lift it to the Poisson-Gaussian domain. The GAT output has standard deviation
# approximately :math:`\gamma`, so the wrapper calls DRUNet at noise level :math:`\gamma`.

drunet = DRUNet(in_channels=1, out_channels=1, device=device)
anscombe_denoiser = AnscombeDenoiser(drunet)

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
# Denoise -- pure Poisson scenario
# --------------------------------
#
# For pure Poisson noise we set :math:`\sigma \approx 0` and pass only
# :math:`\gamma` to both denoisers.
#
# We compare three DRUNet variants:
#
# * **Space-varying noise maps** -- each pixel gets its own
#   :math:`\sigma_i = \sqrt{y_i \cdot \gamma}`.
# * **Global average** -- the spatially averaged noise standard deviation
#   :math:`\bar{\sigma} = \mathrm{mean}(\sigma_i)` is applied uniformly.
# * **Anscombe + DRUNet** -- variance-stabilized via the GAT before denoising.

with torch.no_grad():
    # Anscombe + DRUNet
    x_hat_anscombe_poisson = anscombe_denoiser(y_poisson, gain=gain, sigma=1e-6)

    # Plain DRUNet with space-varying approximate noise maps
    sigma_approx_poisson = (y_poisson * gain).clamp(min=1e-6) ** 0.5
    x_hat_drunet_poisson = drunet(y_poisson, sigma_approx_poisson)

    # Plain DRUNet with global (spatially averaged) noise level
    sigma_global_poisson = sigma_approx_poisson.mean() * torch.ones_like(
        sigma_approx_poisson
    )
    x_hat_drunet_global_poisson = drunet(y_poisson, sigma_global_poisson)

# %%
# Case 2: Mixed Poisson-Gaussian noise
# -------------------------------------
#
# In many real imaging systems (e.g. sCMOS cameras, low-light photography) an
# additional Gaussian read-out noise :math:`\varepsilon \sim \mathcal{N}(0, \sigma^2 I)`
# is superimposed on the Poisson photon noise:
#
# .. math::
#
#     y = \gamma \, z + \varepsilon, \qquad z \sim \mathcal{P}\!\left(\tfrac{x}{\gamma}\right),\;
#     \varepsilon \sim \mathcal{N}(0, \sigma^2 I).
#
# The pixel-wise variance is now :math:`\mathrm{Var}(y_i) = \gamma \, x_i + \sigma^2`,
# combining signal-dependent Poisson variance with a constant Gaussian floor.
# The **Generalized** Anscombe Transform accounts for both terms and still
# stabilizes the variance to approximately :math:`\gamma^2`.

sigma_pg = 0.1  # Gaussian read-out noise sigma

physics_pg = dinv.physics.Denoising(
    dinv.physics.PoissonGaussianNoise(gain=gain, sigma=sigma_pg, clip_positive=True),
    device=device,
)
y_pg = physics_pg(x)


with torch.no_grad():
    # Anscombe + DRUNet
    x_hat_anscombe_pg = anscombe_denoiser(y_pg, gain=gain, sigma=sigma_pg)

    # Plain DRUNet with space-varying approximate noise maps
    sigma_approx_maps_pg = (y_pg * gain + sigma_pg**2).clamp(min=1e-6) ** 0.5
    x_hat_drunet_pg = drunet(y_pg, sigma_approx_maps_pg)

    # Plain DRUNet with global (spatially averaged) noise level
    sigma_global_pg = sigma_approx_maps_pg.mean() * torch.ones_like(
        sigma_approx_maps_pg
    )
    x_hat_drunet_global_pg = drunet(y_pg, sigma_global_pg)

# %%
# Compute PSNR metrics
# --------------------

psnr = dinv.metric.PSNR()

psnr_noisy_poisson = psnr(y_poisson, x).item()
psnr_anscombe_poisson = psnr(x_hat_anscombe_poisson, x).item()
psnr_drunet_poisson = psnr(x_hat_drunet_poisson, x).item()
psnr_drunet_global_poisson = psnr(x_hat_drunet_global_poisson, x).item()

psnr_noisy_pg = psnr(y_pg, x).item()
psnr_anscombe_pg = psnr(x_hat_anscombe_pg, x).item()
psnr_drunet_pg = psnr(x_hat_drunet_pg, x).item()
psnr_drunet_global_pg = psnr(x_hat_drunet_global_pg, x).item()

# %%
# Visualize results -- pure Poisson noise
# ----------------------------------------

dinv.utils.plot(
    [
        x,
        y_poisson,
        x_hat_drunet_global_poisson,
        x_hat_drunet_poisson,
        x_hat_anscombe_poisson,
    ],
    titles=[
        "Ground truth",
        f"Noisy\n({psnr_noisy_poisson:.2f} dB)",
        f"Global sigma\n({psnr_drunet_global_poisson:.2f} dB)",
        f"Noise maps\n({psnr_drunet_poisson:.2f} dB)",
        f"Anscombe\n({psnr_anscombe_poisson:.2f} dB)",
    ],
    rescale_mode="clip",
)

# %%
# Visualize results -- mixed Poisson-Gaussian noise
# -------------------------------------------------

dinv.utils.plot(
    [x, y_pg, x_hat_drunet_global_pg, x_hat_drunet_pg, x_hat_anscombe_pg],
    titles=[
        "Ground truth",
        f"Noisy\n({psnr_noisy_pg:.2f} dB)",
        f"Global sigma\n({psnr_drunet_global_pg:.2f} dB)",
        f"Noise maps\n({psnr_drunet_pg:.2f} dB)",
        f"Anscombe\n({psnr_anscombe_pg:.2f} dB)",
    ],
    rescale_mode="clip",
)

# %%
# Print PSNR summary
# ------------------

print(f"\nPure Poisson  (gamma={gain}):")
print(f"  Noisy               : {psnr_noisy_poisson:.2f} dB")
print(f"  DRUNet global sigma : {psnr_drunet_global_poisson:.2f} dB")
print(f"  DRUNet noise maps   : {psnr_drunet_poisson:.2f} dB")
print(f"  Anscombe            : {psnr_anscombe_poisson:.2f} dB")

print(f"\nMixed Poisson-Gaussian  (gamma={gain}, sigma={sigma_pg}):")
print(f"  Noisy               : {psnr_noisy_pg:.2f} dB")
print(f"  DRUNet global sigma : {psnr_drunet_global_pg:.2f} dB")
print(f"  DRUNet noise maps   : {psnr_drunet_pg:.2f} dB")
print(f"  Anscombe            : {psnr_anscombe_pg:.2f} dB")

# %%
# Conclusion
# ----------
#
# All three DRUNet variants successfully suppress the Poisson-Gaussian noise.
#
# * **DRUNet with global sigma** -- uses a single spatially-averaged noise level
#   :math:`\bar{\sigma} = \mathrm{mean}_i(\sqrt{y_i \cdot \gamma + \sigma^2})`.
#   This is the simplest baseline; it treats the noise as spatially uniform.
# * **DRUNet with noise maps** -- feeds a per-pixel noise-level map
#   :math:`\sigma_i = \sqrt{y_i \cdot \gamma + \sigma^2}` to DRUNet, providing
#   spatial heteroscedasticity information but still operating in the original domain.
# * :class:`AnscombeDenoiser <deepinv.models.AnscombeDenoiser>` is a
#   **zero-cost upgrade**: wrap any off-the-shelf Gaussian denoiser to handle
#   Poisson-Gaussian noise without re-training, by properly stabilizing the
#   heteroscedastic Poisson variance via the GAT before denoising.
#
# :References:
#
# .. footbibliography::
