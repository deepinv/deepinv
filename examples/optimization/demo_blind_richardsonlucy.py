r"""
Blind Richardson-Lucy deconvolution
===================================

This example introduces the Richardson-Lucy algorithm for deconvolution and
then moves to the blind setting, where both the sharp image and the blur kernel
are unknown.

We first consider the non-blind problem

.. math::

    y = A_h x,

where :math:`A_h` is the convolution operator with known kernel :math:`h`.
Richardson-Lucy :footcite:t:`richardsonBayesianBasedIterativeMethod1972,lucyIterativeTechniqueRectification1974`
is the Maximum-Likelihood Expectation-Maximization (MLEM) algorithm for Poisson
deconvolution. Starting from a nonnegative image :math:`x^{(0)}`, it iterates

.. math::

    x^{(k+1)}
    =
    \frac{x^{(k)}}{A_h^\top \mathbf{1}}
    \odot
    A_h^\top\left(\frac{y}{A_h x^{(k)}}\right),

where all products and divisions are pointwise.

When :math:`h` is unknown, blind Richardson-Lucy alternates MLEM updates for the
image :math:`x` and the kernel :math:`h`. This is a classical, simple baseline
for blind deconvolution. It is also very sensitive to noise, which motivates
regularization.


"""

# %%
import matplotlib.pyplot as plt
import torch

import deepinv as dinv

# %%
# Setup
# -----

torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
img_size = 128 if torch.cuda.is_available() else 64

psnr = dinv.metric.PSNR()
mae = dinv.metric.MAE()

gaussian_psf = dinv.physics.functional.gaussian_blur(
    psf_size=(17, 17),
    sigma=3.0,
    device=device,
)

# %%
# Non-blind Richardson-Lucy on a grayscale image
# ----------------------------------------------
#
# In the non-blind setting, the kernel is known and the only unknown is the
# image. In deepinv, this is exactly :class:`deepinv.optim.MLEM` applied to a
# blur physics. We first test it on a grayscale image.

x_gray = dinv.utils.load_example(
    "SheppLogan.png",
    img_size=img_size,
    grayscale=True,
    resize_mode="resize",
    device=device,
)

gain = 1 / 100
physics = dinv.physics.BlurFFT(
    img_size=(1, img_size, img_size),
    filter=gaussian_psf,
    device=device,
    noise_model=dinv.physics.PoissonNoise(
        gain=gain, normalize=True, clip_positive=True
    ),
)
y_gray = physics(x_gray)

data_fidelity = dinv.optim.PoissonLikelihood(gain=1)
mlem = dinv.optim.MLEM(
    data_fidelity=data_fidelity,
    prior=None,
    max_iter=40,
    early_stop=True,
    thres_conv=1e-6,
    crit_conv="residual",
    verbose=True,
)

x_rl, metrics = mlem(
    y_gray,
    physics,
    x_gt=x_gray,
    compute_metrics=True,
)

dinv.utils.plot(
    {
        "Ground truth": x_gray,
        "Blurred": y_gray,
        "Richardson-Lucy": x_rl.clamp(0, 1),
    },
    subtitles=[
        "Reference",
        f"PSNR: {psnr(x_gray, y_gray).item():.2f} dB",
        f"PSNR: {psnr(x_gray, x_rl).item():.2f} dB",
    ],
    figsize=(9, 3),
    rescale_mode="clip",
    vmin=0,
    vmax=1,
)

dinv.utils.plot_curves(metrics)

# %%
# Blind Richardson-Lucy on clean data
# -----------------------------------
#
# We now assume that both the clean image :math:`x` and the blur kernel
# :math:`h` are unknown. Following the library convention, :math:`y` denotes
# the measurement and :math:`A` denotes a forward operator. For fixed
# :math:`h`, :math:`A_h` maps an image to :math:`h * x`; for fixed
# :math:`x`, :math:`A_x` maps a kernel to :math:`x * h`. Thus
# :math:`y = A_h x = A_x h`, and blind Richardson-Lucy alternates
#
# .. math::
#
#    h^{(k+1)}
#    =
#    \Pi_{\Delta}\left[
#    \frac{h^{(k)}}{A_{x^{(k)}}^\top \mathbf{1}}
#    \odot
#    A_{x^{(k)}}^\top\left(\frac{y}{A_{x^{(k)}} h^{(k)}}\right)
#    \right],
#
# and
#
# .. math::
#
#    x^{(k+1)}
#    =
#    \frac{x^{(k)}}{A_{h^{(k+1)}}^\top \mathbf{1}}
#    \odot
#    A_{h^{(k+1)}}^\top\left(\frac{y}{A_{h^{(k+1)}} x^{(k)}}\right).
#
# The projection :math:`\Pi_{\Delta}` keeps the kernel nonnegative and
# normalized to unit sum. The :class:`deepinv.optim.BlindRL` class implements
# these alternating updates.
#
# For simplicity we first test this algorithm on a clean measurement, i.e. without noise. In this case, the algorithm performs fairly well, with some oscillation artefacts near the edges known as [Gibbs phenomenon](https://en.wikipedia.org/wiki/Gibbs_phenomenon).

physics_clean = dinv.physics.Blur(
    filter=gaussian_psf,
    padding="circular",
    device=device,
)
y_clean = physics_clean(x_gray)

blindrl = dinv.optim.BlindRL(
    max_iter=100,
    x_steps=1,
    k_steps=1,
    normalize_kernel=True,
    verbose=True,
)
(x_blindrl, k_blindrl), metrics_clean_blind = blindrl(
    y_clean,
    physics_clean,
    init=y_clean.clamp_min(1e-12),
    x_gt=x_gray,
    compute_metrics=True,
)
x_blindrl = x_blindrl.clamp(0, 1)

dinv.utils.plot(
    {
        "Ground truth": x_gray,
        "Measurement": y_clean,
        "Blind RL": x_blindrl,
        "True kernel": gaussian_psf,
        "Estimated kernel": k_blindrl,
    },
    subtitles=[
        "Reference",
        f"PSNR: {psnr(x_gray, y_clean).item():.2f} dB",
        f"PSNR: {psnr(x_gray, x_blindrl).item():.2f} dB",
        "Reference",
        f"MAE: {mae(gaussian_psf, k_blindrl).item():.4f}",
    ],
    figsize=(13, 3),
    rescale_mode="clip",
    vmin=0,
    vmax=1,
)
dinv.utils.plot_curves(metrics_clean_blind)

# %%
# Blind Richardson-Lucy on noisy data
# -----------------------------------
#
# Blind Richardson-Lucy is much less stable when the data are noisy. With no
# regularization, the multiplicative updates tend to amplify noise and the blur kernel tends to converge very fast to a Dirac, which is never the expected kernel, otherwise the image would already be sharp.
# Early stopping is a common heuristic as a form of implicit regularization, but it is not that efficient.

y_noisy = physics(x_gray)

physics_clean.update_parameters(filter=gaussian_psf)
blindrl = dinv.optim.BlindRL(
    max_iter=80,
    x_steps=1,
    k_steps=1,
    normalize_kernel=True,
    verbose=True,
)
(x_blindrl, k_blindrl), metrics = blindrl(
    y_noisy,
    physics_clean,
    init=y_noisy.clamp_min(1e-12),
    x_gt=x_gray,
    compute_metrics=True,
)
x_blindrl = x_blindrl.clamp(0, 1)

dinv.utils.plot(
    {
        "Ground truth": x_gray,
        "Measurement": y_noisy,
        "Blind RL, noisy": x_blindrl,
        "True kernel": gaussian_psf,
        "Estimated kernel": k_blindrl,
    },
    subtitles=[
        "Reference",
        f"PSNR: {psnr(x_gray, y_noisy).item():.2f} dB",
        f"PSNR: {psnr(x_gray, x_blindrl).item():.2f} dB",
        "Reference",
        f"MAE: {mae(gaussian_psf, k_blindrl).item():.4f}",
    ],
    figsize=(13, 3),
    rescale_mode="clip",
    vmin=0,
    vmax=1,
)
dinv.utils.plot_curves(metrics)

# %%
# One-Step-Late regularization
# ----------------------------
#
# A standard heuristic for regularized EM methods is One-Step-Late (OSL)
# regularization :footcite:t:`greenUseEmAlgorithm1990`. For an image
# regularizer :math:`R_x` and a kernel regularizer :math:`R_h`, the denominators
# are modified using the current subgradients:
#
# .. math::
#
#    x^{(k+1)}
#    =
#    \frac{x^{(k)}}{A_h^\top \mathbf{1}
#    + \lambda_x \partial R_x(x^{(k)})}
#    \odot
#    A_h^\top\left(\frac{y}{A_h x^{(k)}}\right),
#
# .. math::
#
#    h^{(k+1)}
#    =
#    \Pi_{\Delta}\left[
#    \frac{h^{(k)}}{A_x^\top \mathbf{1}
#    + \lambda_h \nabla R_h(h^{(k)})}
#    \odot
#    A_x^\top\left(\frac{y}{A_x h^{(k)}}\right)
#    \right].
#
# This enables the use of any prior implementing the :class:`deepinv.optim.prior.Prior` interface inside the :class:`deepinv.optim.BlindRL` class.
# Remember that this method is heuristic and in particular is not guaranteed to converge.
# Still it is fairly robust in most cases and computationally efficient compared to more sophisticated methods.
# The image and kernel regularizers are independent. Below, we use TV regularization on the image through the :class:`deepinv.optim.TVPrior` class.

tv_prior = dinv.optim.TVPrior(n_it_max=30)

blindrl = dinv.optim.BlindRL(
    x_prior=tv_prior,
    k_prior=None,
    lambda_reg_x=0.015,
    lambda_reg_k=0.0,
    max_iter=30,
    x_steps=1,
    k_steps=1,
    normalize_kernel=True,
    verbose=True,
)
(x_blindrl, k_blindrl), metrics = blindrl(
    y_noisy,
    physics,
    init=y_noisy.clamp_min(1e-12),
    x_gt=x_gray,
    compute_metrics=True,
)
x_blindrl = x_blindrl.clamp(0, 1)

dinv.utils.plot(
    {
        "Ground truth": x_gray,
        "Measurement": y_noisy,
        "Blind RL + TV": x_blindrl,
        "True kernel": gaussian_psf,
        "Estimated kernel": k_blindrl,
    },
    subtitles=[
        "Reference",
        f"PSNR: {psnr(x_gray, y_noisy).item():.2f} dB",
        f"PSNR: {psnr(x_gray, x_blindrl).item():.2f} dB",
        "Reference",
        f"MAE: {mae(gaussian_psf, k_blindrl).item():.4f}",
    ],
    figsize=(13, 3),
    rescale_mode="clip",
    vmin=0,
    vmax=1,
)
dinv.utils.plot_curves(metrics)

# %%
# Blind Richardson-Lucy on an RGB image
# -------------------------------------
#
# The same algorithm also works on RGB images. The blur kernel is shared across
# channels, and the kernel update aggregates information from all channels.
# Here we show a noiseless example.

x_rgb = dinv.utils.load_example(
    "butterfly.png",
    img_size=img_size,
    grayscale=False,
    resize_mode="resize",
    device=device,
)

physics_clean.update_parameters(filter=gaussian_psf)
y_rgb = physics_clean(x_rgb)

blindrl = dinv.optim.BlindRL(
    max_iter=80,
    x_steps=1,
    k_steps=1,
    normalize_kernel=True,
    verbose=True,
)
(x_blindrl, k_blindrl), metrics = blindrl(
    y_rgb,
    physics_clean,
    init=y_rgb.clamp_min(1e-12),
    x_gt=x_rgb,
    compute_metrics=True,
)
x_blindrl = x_blindrl.clamp(0, 1)

dinv.utils.plot(
    {
        "Ground truth": x_rgb,
        "Measurement": y_rgb,
        "Blind RL": x_blindrl,
        "True kernel": gaussian_psf,
        "Estimated kernel": k_blindrl,
    },
    subtitles=[
        "Reference",
        f"PSNR: {psnr(x_rgb, y_rgb).item():.2f} dB",
        f"PSNR: {psnr(x_rgb, x_blindrl).item():.2f} dB",
        "Reference",
        f"MAE: {mae(gaussian_psf, k_blindrl).item():.4f}",
    ],
    figsize=(13, 3),
    rescale_mode="clip",
    vmin=0,
    vmax=1,
)
dinv.utils.plot_curves(metrics)

# %%
# :References:
#
# .. footbibliography::
