r"""
Sparse aperture recovery in plane-wave ultrasound with zea
==========================================================

This example shows how to use the `zea <https://zea.readthedocs.io/>`_ ultrasound
toolbox with deepinv to solve an inverse problem in medical ultrasound imaging.

**Background: how ultrasound images are formed.**
An ultrasound probe is a strip of piezoelectric elements. In coherent plane-wave
compounding (CPWC), each transmit event fires all elements in phase to produce
an unfocused plane wave that illuminates the entire tissue at once. The reflected
echoes are recorded on every element as a function of time, giving a 2-D array of
signals called **channel data** (or RF data). Beamforming -- delay-and-sum across
elements and transmits -- focuses these echoes into a B-mode image.

**The inverse problem: sparse aperture.**
Using all 128 elements is not always possible. Failing elements in an aging probe,
a sparse array design to reduce cable count, or very fast acquisition modes (where
only a fraction of elements fire or record per transmit) all result in missing
receive channels. Formally, the measured IQ channel data is

.. math::

    y = M x^* + \varepsilon

where :math:`x^* \in \mathbb{R}^{N_\mathrm{tx} \times 2 \times N_z \times N_\mathrm{el}}`
holds the full-aperture IQ data for all transmits, :math:`M` is a binary
element-selection mask, and :math:`\varepsilon` is noise. Beamforming
:math:`y` directly produces grating lobes and degrades lateral resolution.

**Recovery with TV regularisation.**
IQ channel data are spatially coherent across the aperture: nearby elements receive
almost identical delayed copies of the same echoes. Total variation (TV) regularisation
exploits this by penalising lateral discontinuities in the IQ profiles:

.. math::

    \hat{x} = \underset{x}{\arg\min} \;
    \frac{1}{2}\|Mx - y\|_2^2 + \lambda\,\mathrm{TV}(x).

We solve this with :class:`deepinv.optim.PGD` (proximal gradient descent), using a
linear-interpolation warm start (via :func:`torch.nn.functional.interpolate`) to
initialise from a reasonable aperture estimate. The recovered IQ data is then passed
through zea's beamforming pipeline to obtain a restored B-mode image.

.. note::
    This example requires `zea <https://zea.readthedocs.io/>`_. Install with
    ``pip install zea``.
"""

# sphinx_gallery_thumbnail_number = 3
# sphinx_gallery_tags = ["Ultrasound", "Inpainting"]

# %%
# Setup
# -----

import os

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("MPLBACKEND", "Agg")  # non-interactive backend for standalone runs

import keras
import keras.ops as kops
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
import torch
import torch.nn.functional as F
import zea
from zea.ops import Beamform, Demodulate, Downsample, EnvelopeDetect, LogCompress, Normalize

import deepinv as dinv

device = dinv.utils.get_device()

# %%
# Load raw channel data
# ---------------------
# The zea carotid dataset contains raw RF data acquired with a 128-element
# linear probe. A single acquisition frame contains 149 transmit events;
# we select the 21 coherent plane-wave transmits, which span angles from
# -18 to +18 degrees.
#
# ``zea.File`` resolves the ``hf://`` URI and caches the file locally on first use.

file_path = "hf://zeahub/zea-carotid-2023/data/2_cross_bifur_right_0000_small.hdf5"

with zea.File(file_path, revision="v0.1.0") as f:
    params = f.load_parameters()
    plane_txs = params.find_transmits("plane")  # 21 plane-wave indices
    params.selected_transmits = plane_txs
    raw_data = f.data.raw_data[:1, plane_txs]  # (1, n_tx, n_ax, n_el, 1)

n_tx, n_ax, n_el = raw_data.shape[1], raw_data.shape[2], raw_data.shape[3]
tx_mid = n_tx // 2  # central (0-degree) transmit

print(
    f"Transmit events: {n_tx}  |  Depth samples: {n_ax}  |  Elements: {n_el}"
)

# extent_imshow triggers internal numpy calls; keras.device("cpu") keeps those on CPU
with keras.device("cpu"):
    ext_mm = kops.convert_to_numpy(params.extent_imshow * 1000).tolist()  # [x0,x1,z1,z0] mm

# Crop B-mode display to probe aperture and 40 mm depth.
bmode_xlim = (-20.0, 20.0)  # mm
bmode_ylim = (40.0, 0.0)  # mm; display 0-40 mm depth

# %%
# Raw RF channel data: seeing the arcs
# -------------------------------------
# For a single plane-wave transmit, the channel data form a 2-D array with
# time (depth) on the vertical axis and transducer element index on the
# horizontal axis.
#
# A point-like reflector at depth :math:`z` and lateral position :math:`x`
# produces a **hyperbolic arc** in this view because the round-trip travel
# time to element :math:`i` at position :math:`x_i` is
#
# .. math::
#
#    t(i) = \frac{z}{c} + \frac{\sqrt{(x - x_i)^2 + z^2}}{c},
#
# which is a hyperbola in :math:`i`. Beamforming aligns and sums these arcs
# to form a focused image.
#
# The RF signal oscillates at the 7.8 MHz carrier frequency; the sampling
# frequency is 31.25 MHz (about 4 samples per cycle). We display both the
# raw oscillating signal and the log-compressed envelope, which reveals tissue
# structure across 40 mm of depth at high dynamic range.

rf_mid = raw_data[0, tx_mid, :, :, 0]  # (n_ax, n_el) -- central transmit (numpy)

# Log-compressed envelope for the full depth
env = np.abs(hilbert(rf_mid, axis=0))
log_env = 20 * np.log10(env / env.max() + 1e-10)
log_env = np.clip(log_env, -60, 0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

clip = np.percentile(np.abs(rf_mid), 99)
axes[0].imshow(
    rf_mid,
    cmap="seismic",
    aspect="auto",
    vmin=-clip,
    vmax=clip,
    interpolation="none",
)
axes[0].set_title(f"Raw RF signal (central transmit, 0 deg)")
axes[0].set_xlabel("Element index")
axes[0].set_ylabel("Time sample")

im = axes[1].imshow(
    log_env,
    cmap="gray",
    aspect="auto",
    vmin=-60,
    vmax=0,
    interpolation="none",
)
axes[1].set_title("Log-compressed RF envelope (60 dB dynamic range)")
axes[1].set_xlabel("Element index")
axes[1].set_ylabel("Time sample")
plt.colorbar(im, ax=axes[1], label="dB")

plt.tight_layout()
plt.show()

# %%
# Beamforming pipeline: from raw RF to B-mode
# --------------------------------------------
# The zea pipeline processes raw RF data through five stages:
#
# 1. **Demodulate** -- mix with the carrier and low-pass filter to obtain
#    complex baseband (IQ) samples (one real and one imaginary channel).
# 2. **Downsample(4)** -- reduce the sample rate 4x; the baseband signal
#    is band-limited to 1/4 of the RF bandwidth.
# 3. **Beamform (delay-and-sum)** -- for each image pixel, apply per-element
#    time delays computed from the probe geometry and the transmit angle, then
#    sum coherently. Repeating this for all 21 transmit angles and averaging
#    is called coherent plane-wave compounding (CPWC).
# 4. **EnvelopeDetect** -- compute the analytic signal magnitude.
# 5. **Normalize / LogCompress** -- map to a 60 dB display range.
#
# We split the pipeline at the IQ stage (after step 2) so that deepinv can
# operate on IQ channel data before beamforming.
#
# .. note::
#     ``prepare_parameters`` calls NumPy internally in zea v0.1.x; we run it
#     on a CPU pipeline and move the resulting delay tables to ``device``.

pipe_inputs = zea.Pipeline(
    operations=[
        Demodulate(),
        Downsample(factor=4),
        Beamform(beamformer="delay_and_sum", enable_pfield=False),
        EnvelopeDetect(),
        Normalize(),
        LogCompress(),
    ],
    jit_options=None,
    device="cpu",
).prepare_parameters(params)
pipe_inputs = {
    k: v.to(device) if isinstance(v, torch.Tensor) else v
    for k, v in pipe_inputs.items()
}
# prepare_parameters stores the RF sampling rate; the split post-pipeline receives
# IQ data (4x downsampled), so we override to the IQ rate for correct delay indexing.
pipe_inputs["sampling_frequency"] = pipe_inputs["demodulation_frequency"]

# Separate pipe_inputs for single-transmit beamforming (index 0 = first/most-negative angle).
# prepare_parameters checks n_tx consistency, so we need a dedicated 1-TX parameter dict.
params.selected_transmits = [plane_txs[0]]
pipe_inputs_1tx = zea.Pipeline(
    operations=[
        Demodulate(),
        Downsample(factor=4),
        Beamform(beamformer="delay_and_sum", enable_pfield=False),
        EnvelopeDetect(),
        Normalize(),
        LogCompress(),
    ],
    jit_options=None,
    device="cpu",
).prepare_parameters(params)
pipe_inputs_1tx = {
    k: v.to(device) if isinstance(v, torch.Tensor) else v
    for k, v in pipe_inputs_1tx.items()
}
pipe_inputs_1tx["sampling_frequency"] = pipe_inputs_1tx["demodulation_frequency"]
params.selected_transmits = plane_txs  # restore

pre_pipeline = zea.Pipeline(
    operations=[Demodulate(), Downsample(factor=4)],
    jit_options=None,
    device=str(device),
)
post_pipeline = zea.Pipeline(
    operations=[
        Beamform(beamformer="delay_and_sum", enable_pfield=False),
        EnvelopeDetect(),
        Normalize(),
        LogCompress(),
    ],
    jit_options=None,
    device=str(device),
)

# RF -> IQ: (1, n_tx, n_ax, n_el, 1) -> (1, n_tx, n_ax4, n_el, 2)
x_raw = torch.tensor(raw_data[0], dtype=torch.float32).to(device)
x_raw = x_raw / x_raw.abs().max()

with torch.no_grad():
    iq_zea = pre_pipeline(data=x_raw.unsqueeze(0), **pipe_inputs)["data"]

n_ax4 = iq_zea.shape[2]
# Permute to deepinv channel-first convention: (n_tx, 2, n_ax4, n_el)
x_iq = iq_zea[0].permute(0, 3, 1, 2)

print(f"IQ tensor shape (n_tx, I/Q, depth, elements): {tuple(x_iq.shape)}")


def iq_to_bmode(t_iq, pargs=None):
    """(n_tx, 2, n_ax4, n_el) -> log-compressed B-mode (H, W) as numpy array."""
    if pargs is None:
        pargs = pipe_inputs
    zea_iq = t_iq.permute(0, 2, 3, 1).unsqueeze(0)  # (1, n_tx, n_ax4, n_el, 2)
    with torch.no_grad():
        out = post_pipeline(data=zea_iq, **pargs)
    return kops.convert_to_numpy(out["data"][0])


# CPWC: coherently compound all 21 transmits (the reference full-aperture image)
bmode_ref = iq_to_bmode(x_iq)
# Single off-axis transmit (index 0 = first/most-negative angle); uses the
# dedicated 1-TX pipe_inputs so delay table shapes are consistent.
bmode_1tx = iq_to_bmode(x_iq[0:1], pargs=pipe_inputs_1tx)

# %%
# From one angle to coherent compounding
# ----------------------------------------
# Delay-and-sum on a **single** plane-wave transmit focuses the echoes from that
# one illumination angle but leaves behind directional side-lobes and spatial
# ambiguities: reflectors at different lateral positions that share the same
# travel time for that particular angle are not fully separated.
#
# **Coherent plane-wave compounding (CPWC)** fires :math:`N_\mathrm{tx}` plane
# waves at different angles :math:`\{\theta_i\}` and sums the individual
# delay-and-sum images coherently:
#
# .. math::
#
#    \hat{u}(r) = \sum_{i=1}^{N_\mathrm{tx}} \mathrm{DAS}_{\theta_i}(r)
#
# Each angle brings independent angular diversity; side-lobes of one transmit
# partly cancel the main lobes of another. The result is a dramatic improvement
# in lateral resolution and contrast.

fig, axes = plt.subplots(1, 2, figsize=(10, 7))
for ax, bmode, title in zip(
    axes,
    [bmode_1tx, bmode_ref],
    [
        "Single-transmit DAS (1 angle)",
        f"CPWC ({n_tx} angles, full aperture)",
    ],
    strict=False,
):
    ax.imshow(bmode, cmap="gray", aspect="auto", origin="upper", extent=ext_mm)
    ax.set_xlim(*bmode_xlim)
    ax.set_ylim(*bmode_ylim)
    ax.set_title(title)
    ax.set_xlabel("Lateral (mm)")
    ax.set_ylabel("Depth (mm)")
plt.tight_layout()
plt.show()

# %%
# Sparse aperture acquisition: forward model
# -------------------------------------------
# Sparse aperture acquisition arises in several practical scenarios:
#
# - **Defective probes**: individual elements fail over time, leaving gaps
#   in the array.
# - **Sparse array design**: fewer cables and amplifiers are needed if only
#   every k-th element is read out, reducing system cost and data throughput.
# - **Fast acquisition**: some ultra-high frame rate protocols fire and receive
#   on only a subset of elements per transmit to increase temporal resolution.
#
# We simulate 4x element subsampling: keep every 4th element (32 of 128),
# zeroing the remaining 96 receive channels. The :class:`deepinv.physics.Inpainting`
# operator applies this binary mask column-wise to the IQ data.

STRIDE = 4  # 32 of 128 elements active (25 % aperture)

mask = torch.zeros(2, n_ax4, n_el, device=device)
mask[:, :, ::STRIDE] = 1.0  # same mask for the I and Q channels

physics = dinv.physics.Inpainting(img_size=(2, n_ax4, n_el), mask=mask, device=device)
y_iq = physics.A(x_iq)  # (n_tx, 2, n_ax4, n_el) -- missing columns zeroed

n_obs = int(mask[0, 0].sum().item())
print(f"Active elements: {n_obs} / {n_el}  ({100 * n_obs // n_el} %)")

bmode_sub = iq_to_bmode(y_iq)

fig, axes = plt.subplots(1, 2, figsize=(10, 7))
for ax, bmode, title in zip(
    axes,
    [bmode_ref, bmode_sub],
    [
        f"Full aperture ({n_el} elements)",
        f"Sparse aperture ({n_obs}/{n_el} elements)",
    ],
    strict=False,
):
    ax.imshow(bmode, cmap="gray", aspect="auto", origin="upper", extent=ext_mm)
    ax.set_xlim(*bmode_xlim)
    ax.set_ylim(*bmode_ylim)
    ax.set_title(title)
    ax.set_xlabel("Lateral (mm)")
    ax.set_ylabel("Depth (mm)")
plt.tight_layout()
plt.show()

# %%
# Recovery with a linear warm start and TV regularisation
# --------------------------------------------------------
# IQ channel data are spatially coherent across the receive aperture: nearby
# elements record closely related delayed copies of the same wavefront.
# TV regularisation exploits this by penalising large lateral variations in the
# IQ profiles across the element axis.
#
# We use :class:`deepinv.optim.PGD` (proximal gradient descent) with a
# :class:`deepinv.optim.TVPrior`. The observed elements give the data-fidelity
# term; we penalise TV on the estimated full-aperture IQ data.
#
# A **linear-interpolation warm start** -- computed with
# :func:`torch.nn.functional.interpolate` -- provides a much better initial
# estimate than zero-fill and substantially accelerates convergence. The
# observed elements (every ``STRIDE``-th column) are extracted, upsampled back
# to 128 using linear interpolation, and passed as the starting point.

# Warm start: reshape to (batch, 1, n_obs) for F.interpolate, then back.
B = n_tx * 2 * n_ax4
x_init = F.interpolate(
    y_iq[:, :, :, ::STRIDE].reshape(B, 1, n_obs),
    size=n_el,
    mode="linear",
    align_corners=True,
).reshape(n_tx, 2, n_ax4, n_el)

model = dinv.optim.PGD(
    data_fidelity=dinv.optim.data_fidelity.L2(),
    prior=dinv.optim.TVPrior(),
    lambda_reg=5e-3,
    max_iter=300,
    early_stop=True,
    thres_conv=1e-5,
    verbose=True,
    custom_init=lambda y, _: x_init,
)

with torch.no_grad():
    x_hat_iq = model(y_iq, physics)

bmode_lin = iq_to_bmode(x_init)
bmode_rec = iq_to_bmode(x_hat_iq)

fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
for ax, bmode, title in zip(
    axes,
    [bmode_ref, bmode_sub, bmode_lin, bmode_rec],
    [
        f"Full aperture\n({n_el} elements)",
        f"Sparse (zero-fill)\n({n_obs}/{n_el} elements)",
        "Linear interp\n(warm start)",
        "PGD + TV\n(recovered)",
    ],
    strict=False,
):
    ax.imshow(bmode, cmap="gray", aspect="auto", origin="upper", extent=ext_mm)
    ax.set_xlim(*bmode_xlim)
    ax.set_ylim(*bmode_ylim)
    ax.set_title(title)
    ax.set_xlabel("Lateral (mm)")
axes[0].set_ylabel("Depth (mm)")
plt.tight_layout()
plt.show()

# %%
# Reconstruction quality
# -----------------------

with torch.no_grad():
    mse = ((x_iq - x_hat_iq) ** 2).mean().item()
    psnr = -10.0 * torch.log10(torch.tensor(mse)).item()

print(f"PSNR (IQ domain, recovered vs. reference): {psnr:.2f} dB")

# %%
# What's next?
# ------------
#
# IQ-domain TV provides a simple and interpretable baseline that reduces grating
# lobe artefacts. The framework generalises to more powerful methods:
#
# - Replace TV with a learned prior using :ref:`PnP denoisers <denoisers>` or
#   deep unrolling (:func:`deepinv.unfolded.unfolded_builder`), which can
#   model the oscillatory plane-wave structure of IQ data that TV cannot.
# - Define the loss in the **B-mode domain**: the zea beamformer is fully
#   differentiable, so TV can be applied to the envelope-detected image instead
#   of the raw IQ data -- where piecewise-constant tissue structure is a
#   natural prior (see :ref:`[Zhang2026] <references>`).
# - Use a multi-domain loss combining IQ-domain data fidelity with B-mode
#   domain perceptual quality, as in :ref:`[Xiao2022] <references>`.
# - Try transmit subsampling: use fewer plane-wave angles instead of fewer
#   elements, which trades off temporal vs. spatial resolution.
# - Extend to 3-D matrix array probes, where both lateral dimensions of the
#   aperture can be jointly subsampled.

# %%
# .. _references:
#
# References
# ----------
#
# .. [Stevens2025] Stevens, T.S.W., van Nierop, W.L., Luijten, B., van de Schaft, V.,
#    Nolan, O., Federici, B., van Harten, L.D., Penninga, S.W., Schueler, N.I.P. &
#    van Sloun, R.J.G. (2025). *zea: A Toolbox for Cognitive Ultrasound Imaging.*
#    arXiv:2512.01433.
#    `https://arxiv.org/abs/2512.01433 <https://arxiv.org/abs/2512.01433>`_
#
# .. [Montaldo2009] Montaldo, G., Tanter, M., Bercoff, J., Benech, N. & Fink, M. (2009).
#    Coherent plane-wave compounding for very high frame rate ultrasonography and
#    transient elastography. *IEEE Trans. Ultrason. Ferroelectr. Freq. Control*,
#    56(3):489--506.
#    DOI: `10.1109/TUFFC.2009.1067 <https://doi.org/10.1109/TUFFC.2009.1067>`_
#
# .. [Zhang2026] Zhang, L., You, Y., Tandri, A., Feng, Y., Song, H., Kang, J. & Lin, Y.
#    (2026). *Physics-Guided Multi-Objective Deep Learning for Ultrasound RF Data
#    Interpolation in Resource-Constrained Imaging.* arXiv:2609.28775.
#    `https://arxiv.org/abs/2609.28775 <https://arxiv.org/abs/2609.28775>`_
#
# .. [Xiao2022] Xiao, D., Pitman, W.M.K., Yiu, B.Y.S., Chee, A.J.Y. & Yu, A.C.H. (2022).
#    Minimizing image quality loss after channel count reduction for plane wave ultrasound
#    via deep learning inference. *IEEE Trans. Ultrason. Ferroelectr. Freq. Control*,
#    69(10):2849--2861.
#    DOI: `10.1109/TUFFC.2022.3192854 <https://doi.org/10.1109/TUFFC.2022.3192854>`_
