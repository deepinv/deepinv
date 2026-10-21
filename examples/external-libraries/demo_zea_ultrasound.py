r"""
Regularized beamforming of plane-wave ultrasound with zea
=========================================================

This example shows how to use the `zea <https://zea.readthedocs.io/>`_ ultrasound
toolbox :ref:`[Stevens2026] <references>` with deepinv to form ultrasound images by
solving an inverse problem, instead of with the standard delay-and-sum beamformer.

zea implements the full ultrasound processing chain, from raw channel data to
the displayed image (demodulation, beamforming, envelope detection, log
compression, ...), as a pipeline of composable operations. It is built on
Keras 3, so the same pipeline runs natively in PyTorch (as here), JAX or
TensorFlow, and every step is differentiable. This makes it easy to plug zea
operators into deepinv, as physics operators in an inverse problem. zea also
provides public ultrasound datasets on the Hugging Face Hub, which we use below.

**Background: how ultrasound images are formed.**
An ultrasound probe is an array of (often) piezoelectric elements that both transmit
sound pulses and record the returning echoes. There are multiple ways of acquiring
echo data that involve different transmit schemes, such as focused imaging or
divering waves. In this example we focus on plane-wave imaging. In plane-wave imaging,
each transmit event fires all elements at once (optionally with a small steering
delay) to produce an unfocused plane wave that illuminates the entire field of
view in one shot. The reflected echoes are recorded on every element as a
function of time, giving a 2-D array of signals called **channel data** (see
`this blog post <https://www.vincentvandeschaft.net/notes/channel-data>`_ for an
intuitive explanation).
**Beamforming** turns these channel data into an image. The standard method,
delay-and-sum (DAS), computes for every pixel the travel time of the echo to
each element, picks the corresponding sample from each channel and sums them.
A single plane wave gives a fast but low-quality image; coherent plane-wave
compounding (CPWC) :ref:`[Montaldo2009] <references>` sums the DAS images of many
steered plane waves to improve resolution and contrast, at the cost of frame rate.

**Beamforming as an inverse problem.**
Let :math:`x` be the (complex) reflectivity of the tissue on a fixed pixel grid
(often written :math:`s` or :math:`\gamma` in the ultrasound literature),
i.e. the amplitude of a scatterer at every pixel. Neglecting the shape of the
pulse, each scatterer contributes its amplitude to the channel data at the
travel time from the transmitted wave to the pixel and back to each element.
This is a linear model

.. math::

    y = A x + \varepsilon,

and it turns out that DAS applies exactly the adjoint operator,
:math:`\mathrm{DAS} = A^{\top}` :ref:`[Besson2018] <references>`: DAS "smears"
each channel sample back over all pixels that could have produced it. DAS is
thus a back-projection, not an inversion, which is what causes its side lobes
and clutter. Instead, we estimate :math:`x` from a **single** plane wave by
solving the regularized problem

.. math::

    \hat{x} = \underset{x}{\arg\min} \;
    \frac{1}{2}\|A x - y\|_2^2 + \lambda\,\|\Psi x\|_1,

where :math:`\Psi` is a wavelet transform. Ultrasound images are sparse in
wavelets, and the problem is heavily ill-posed (many more pixels than useful
measurements), so the prior is essential :ref:`[Besson2018, Ozkan2018] <references>`.

**zea + deepinv.** zea provides a differentiable DAS beamformer. We use it as
the adjoint :math:`A^{\top}`, and obtain the forward operator :math:`A` with
automatic differentiation via :func:`deepinv.physics.adjoint_function`. The
problem is then solved with :class:`deepinv.optim.FISTA` and a
:class:`deepinv.optim.WaveletPrior`.

.. note::
    This example requires `zea <https://zea.readthedocs.io/>`_. Install with
    ``pip install zea``.
"""

# %%
# Setup
# -----

# sphinx_gallery_thumbnail_number = 5
# sphinx_gallery_tags = ["Ultrasound", "Beamforming"]

import os

# non-interactive matplotlib backend for standalone runs
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("KERAS_BACKEND", "torch")

import matplotlib.pyplot as plt
import numpy as np
import torch
import zea
from matplotlib.ticker import MultipleLocator
from scipy import ndimage
from zea.ops import (
    Beamform,
    Demodulate,
    Downsample,
    EnvelopeDetect,
    LogCompress,
    Normalize,
)
from zea.ops.keras_ops import Transpose

import deepinv as dinv

device = dinv.utils.get_device()

# %%
# Load raw channel data
# ---------------------
# We use a recording of the **carotid artery**, the main blood vessel in the neck
# that supplies the brain. It lies only a few centimetres below the skin, so it
# is routinely imaged with ultrasound, for example to look for plaque or
# narrowing of the vessel. In a B-mode image the vessel lumen (filled with blood)
# appears dark, surrounded by the brighter vessel wall and tissue.
#
# The `zea carotid dataset <https://huggingface.co/datasets/zeahub/zea-carotid-2023>`_
# contains raw RF data acquired with a 128-element
# linear probe. We use a longitudinal view near the carotid bifurcation, where
# the artery splits in two: the vessel runs horizontally through the image, with
# its dark lumen at roughly 20-27 mm depth. A single acquisition frame contains
# 149 transmit events; we select the 21 plane-wave transmits, which span angles
# from -18 to +18 degrees.
#
# ``zea.File`` resolves the ``hf://`` URI and caches the file locally on first use.

file_path = "hf://zeahub/zea-carotid-2023/data/9_long_bifur_R_0000.hdf5"

with zea.File(file_path, revision="v0.1.5") as f:
    params = f.load_parameters()
    plane_txs = params.find_transmits("plane")  # 21 plane-wave indices
    params.selected_transmits = plane_txs
    raw_data = f.data.raw_data[:1, plane_txs]  # (1, n_tx, n_ax, n_el, 1)

# Pixel grid spanning the probe aperture and 2-40 mm depth, with 2 pixels per
# wavelength (about 0.1 mm). Every pixel is an unknown of the inverse problem.
params.xlims = (-20e-3, 20e-3)
params.zlims = (2e-3, 40e-3)
params.pixels_per_wavelength = 2

n_tx, n_ax, n_el = raw_data.shape[1], raw_data.shape[2], raw_data.shape[3]
print(f"Transmit events: {n_tx}  |  Time samples: {n_ax}  |  Elements: {n_el}")
print(f"Pixel grid: {params.grid_size_z} x {params.grid_size_x}")

ext_mm = params.extent_imshow * 1000  # image extent for plotting, in mm

# %%
# Raw RF channel data
# -------------------
# For a single plane-wave transmit, the channel data form a 2-D array with
# time (depth) on the vertical axis and transducer element index on the
# horizontal axis. A point-like scatterer at :math:`(x_0, z_0)` produces a
# **hyperbolic arc** in this view, because the travel time back to element
# :math:`i` at position :math:`x_i` is :math:`\sqrt{(x_0 - x_i)^2 + z_0^2}/c`.
# Beamforming aligns and sums these arcs to form a focused image.

rf_0deg = raw_data[0, 0, :, :, 0]  # (n_ax, n_el), the 0-degree plane wave

clip = np.percentile(np.abs(rf_0deg), 99)
plt.figure(figsize=(6, 5))
plt.imshow(
    rf_0deg, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip, interpolation="none"
)
plt.title("Raw RF channel data (0 deg plane wave)")
plt.xlabel("Element index")
plt.ylabel("Time sample")
plt.tight_layout()
plt.show()

# %%
# Delay-and-sum with zea
# ----------------------
# We build three small zea pipelines:
#
# 1. **Demodulate + Downsample(4)**: convert the RF data to complex baseband
#    (IQ) data, stored as two real channels, at a 4x lower sample rate.
# 2. **Beamform**: delay-and-sum, from IQ channel data to an IQ image on the
#    pixel grid. This is the operator :math:`A^{\top}`. zea beamforms the grid in
#    patches; a large ``patch_size`` makes it much faster on GPU. A final
#    ``Transpose`` puts the I/Q channels first, as deepinv expects.
# 3. **B-mode**: from IQ image to its envelope (the magnitude of the complex
#    signal), normalized to its maximum and log-compressed to dB. For display,
#    :func:`zea.display.to_8bit` maps a 60 dB range to grey levels.

pre_pipeline = zea.Pipeline(
    [Demodulate(), Downsample(factor=4)], jit_options=None, device=str(device)
)
das_pipeline = zea.Pipeline(
    [
        Beamform(beamformer="delay_and_sum", enable_pfield=False, patch_size=2**15),
        Transpose(axes=(2, 0, 1)),  # (n_z, n_x, 2) -> (2, n_z, n_x), per batch
    ],
    jit_options=None,
    device=str(device),
)
image_pipeline = zea.Pipeline(
    [
        Transpose(axes=(1, 2, 0)),  # back to (n_z, n_x, 2)
        EnvelopeDetect(),
        Normalize(input_range=(0, None)),
        LogCompress(clip=False),
    ],
    jit_options=None,
    device=str(device),
)


# zea converts the acquisition parameters (probe geometry, transmit delays, pixel
# grid, ...) of the selected transmits into tensors on ``device``.
def prepare_parameters(params):
    return {
        k: v
        for pipeline in [pre_pipeline, das_pipeline, image_pipeline]
        for k, v in pipeline.prepare_parameters(params).items()
    }


def rf_to_iq(rf, inputs):
    """(n_tx, n_ax, n_el, 1) RF -> (1, n_tx, n_ax4, n_el, 2) IQ, plus parameters
    updated for the IQ data (e.g. the sampling frequency is divided by 4)."""
    with torch.no_grad():
        out = pre_pipeline(data=rf.unsqueeze(0), **inputs)
    return out.pop("data"), {**inputs, **out}


def to_image(x, inputs):
    """(1, 2, n_z, n_x) IQ image -> B-mode image (n_z, n_x), numpy array in dB."""
    with torch.no_grad():
        return image_pipeline(data=x, return_numpy=True, **inputs)["data"][0]


rf_raw = torch.tensor(raw_data[0], dtype=torch.float32, device=device)
rf_raw = rf_raw / rf_raw.abs().max()

# A single plane wave (index 0, the 0-degree transmit): our measurements.
params.selected_transmits = [plane_txs[0]]
y, inputs = rf_to_iq(rf_raw[:1], prepare_parameters(params))


def das(y):
    """DAS beamforming of the single plane wave: channel data -> IQ image."""
    return das_pipeline(data=y, **inputs)["data"]


with torch.no_grad():
    x_das = das(y)
image_das = to_image(x_das, inputs)

print(f"Measurements y (1, n_tx, time, elements, I/Q): {tuple(y.shape)}")
print(f"Image x (1, I/Q, depth, lateral): {tuple(x_das.shape)}")

plt.figure(figsize=(5, 5))
plt.imshow(
    zea.display.to_8bit(image_das, dynamic_range=(-60, 0), pillow=False),
    cmap="gray",
    vmin=0,
    vmax=255,
    aspect="auto",
    extent=ext_mm,
)
plt.title("DAS, 1 plane wave")
plt.xlabel("Lateral (mm)")
plt.ylabel("Depth (mm)")
plt.tight_layout()
plt.show()

# %%
# Quantifying ultrasound image quality
# ------------------------------------
# As reference we use the best image this acquisition can give: DAS on all 21
# plane waves, combined by coherent compounding (CPWC, see the next section).
#
# **Qualitative comparison.** Different image formation methods change the
# overall brightness and dynamic range of an image, which biases a visual
# comparison. We therefore show every image **histogram-matched** to the CPWC
# reference with :func:`zea.display.histogram_match`
# :ref:`[Bottenus2021] <references>`: a monotone mapping of grey levels gives each
# image the grey-level distribution of the reference, so that only the image
# structure differs.
#
# **Quantitative comparison.** We measure how well the dark vessel lumen stands
# out from the surrounding tissue with the **generalized contrast-to-noise
# ratio** (gCNR) :ref:`[RodriguezMolares2020] <references>` of
# :mod:`zea.metrics`: one minus the overlap of the grey-level histograms of a
# lumen region and a tissue region. A gCNR of 0 means the lumen is
# indistinguishable from tissue, 1 means perfectly separable.
#
# The lumen region is segmented automatically from the CPWC image: we smooth it
# over about 1 mm to average out the speckle, keep the darkest 30 % of the
# pixels, and take the connected region in the middle of the vessel. The tissue
# region is a box above the vessel.

params.selected_transmits = plane_txs  # all 21 plane waves
y_all, inputs_all = rf_to_iq(rf_raw, prepare_parameters(params))
with torch.no_grad():
    x_cpwc = das_pipeline(data=y_all, **inputs_all)["data"]
image_cpwc = to_image(x_cpwc, inputs_all)

x_mm = np.linspace(ext_mm[0], ext_mm[1], image_cpwc.shape[1])
z_mm = np.linspace(ext_mm[3], ext_mm[2], image_cpwc.shape[0])
pixel_mm = z_mm[1] - z_mm[0]

smooth = ndimage.gaussian_filter(np.clip(image_cpwc, -60, 0), 1.0 / pixel_mm)
dark = ndimage.binary_opening(
    smooth < np.percentile(smooth, 30), iterations=round(0.5 / pixel_mm)
)
labels, _ = ndimage.label(dark)
lumen = labels == labels[np.abs(z_mm - 23).argmin(), np.abs(x_mm - 5).argmin()]
lumen = ndimage.binary_fill_holes(
    ndimage.binary_closing(lumen, iterations=round(1.0 / pixel_mm))
)

TISSUE = (-15.0, 15.0, 8.0, 14.0)  # x0, x1, z0, z1 in mm
tissue = (
    (x_mm[None] >= TISSUE[0])
    & (x_mm[None] <= TISSUE[1])
    & (z_mm[:, None] >= TISSUE[2])
    & (z_mm[:, None] <= TISSUE[3])
)


def to_display(image):
    """Image in dB -> histogram-matched to the CPWC image, as 8-bit (60 dB range)."""
    matched = zea.display.histogram_match(image, image_cpwc)
    return zea.display.to_8bit(matched, dynamic_range=(-60, 0), pillow=False)


def gcnr(image):
    """gCNR between the lumen and the tissue region."""
    return float(zea.metrics.gcnr(image[lumen], image[tissue]))


plt.figure(figsize=(5, 5))
plt.imshow(
    to_display(image_cpwc), cmap="gray", vmin=0, vmax=255, aspect="auto", extent=ext_mm
)
plt.contour(x_mm, z_mm, lumen, [0.5], colors="tab:red", linewidths=2)
x0, x1, z0, z1 = TISSUE
plt.gca().add_patch(
    plt.Rectangle((x0, z0), x1 - x0, z1 - z0, fill=False, ec="tab:cyan", lw=2)
)
plt.title("Regions for the gCNR: lumen (red), tissue (cyan)")
plt.xlabel("Lateral (mm)")
plt.ylabel("Depth (mm)")
plt.tight_layout()
plt.show()


def plot_images(images, titles, zoom=None):
    """Side-by-side images, histogram-matched to the CPWC reference."""
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 5), sharey=True)
    for ax, image, title in zip(axes, images, titles, strict=True):
        ax.imshow(
            to_display(image),
            cmap="gray",
            vmin=0,
            vmax=255,
            aspect="auto",
            extent=ext_mm,
        )
        if zoom is None:
            ax.set_title(f"{title}\ngCNR = {gcnr(image):.2f}")
        else:
            ax.set_xlim(*zoom[0])
            ax.set_ylim(*zoom[1])
            ax.set_title(title)
        tick = 5 if zoom is not None else 10  # mm
        ax.xaxis.set_major_locator(MultipleLocator(tick))
        ax.yaxis.set_major_locator(MultipleLocator(tick))
        ax.set_xlabel("Lateral (mm)")
    axes[0].set_ylabel("Depth (mm)")
    plt.tight_layout()
    plt.show()


# %%
# From one angle to coherent compounding
# --------------------------------------
# DAS on a single plane wave leaves side lobes and clutter: the dark lumen of
# the vessel is filled with a grey haze and the speckle is laterally blurred.
# **Coherent plane-wave compounding** (CPWC) fires several plane waves at
# different angles and sums their DAS images. Each angle brings new
# information, and the side lobes of different angles partly cancel out, so we
# expect a clearly better image: less clutter in the lumen, sharper vessel walls
# and a higher gCNR. This is indeed what we see. The price is frame rate: 21
# plane waves need 21 transmit events per image. The goal of regularized
# beamforming is to get closer to the right image from the single plane wave on
# the left.

plot_images([image_das, image_cpwc], ["DAS, 1 plane wave", f"DAS, {n_tx} plane waves"])

# %%
# The forward operator
# --------------------
# DAS is linear in the channel data, so it is the adjoint :math:`A^{\top}` of a
# linear forward operator :math:`A` that maps a reflectivity image to channel
# data. Since the zea beamformer is differentiable, we do not need to implement
# :math:`A` by hand: :func:`deepinv.physics.adjoint_function` computes it as the
# adjoint of DAS with automatic differentiation (a vector-Jacobian product).
# Together they define a :class:`deepinv.physics.LinearPhysics`.

A = dinv.physics.adjoint_function(das, y.shape, device=device)
physics = dinv.physics.LinearPhysics(A=A, A_adjoint=das)

# Check that A and DAS are adjoint: <A x, y> = <x, DAS y> for random x and y.
x_rand = torch.randn_like(x_das)
y_rand = torch.randn_like(y)
with torch.no_grad():
    lhs = (physics.A(x_rand) * y_rand).sum().item()
    rhs = (x_rand * physics.A_adjoint(y_rand)).sum().item()
print(f"Adjoint test: <A x, y> = {lhs:.4f}, <x, DAS y> = {rhs:.4f}")

# Lipschitz constant of the data-fidelity gradient, for the step size.
with torch.no_grad():
    norm = physics.compute_norm(torch.randn_like(x_das), tol=1e-3, verbose=False)
print(f"||A^T A|| = {norm.item():.1f}")

# %%
# Regularized beamforming with FISTA
# ----------------------------------
# We solve the problem with :class:`deepinv.optim.FISTA` (accelerated proximal
# gradient descent): each iteration takes a gradient step on
# :math:`\frac{1}{2}\|Ax - y\|_2^2`, which costs one forward and one adjoint
# beamforming, and then applies the proximal operator of the prior.
#
# As prior we use the :math:`\ell_1` norm of the wavelet coefficients of the I
# and Q images. Following :ref:`[Besson2018] <references>`, :math:`\Psi`
# concatenates the Daubechies wavelets db1 to db8 ("sparsity averaging"), which
# :class:`deepinv.optim.WaveletPrior` supports by passing a list of wavelets.
# The regularization weight is set relative to the scale of the DAS image.
#
# To see why the prior is needed, we also solve the unregularized least-squares
# problem (:math:`\lambda = 0`).


def regularized_beamforming(lambda_rel, max_iter=100):
    prior = dinv.optim.WaveletPrior(
        level=3, wv=[f"db{i}" for i in range(1, 9)], device=device
    )
    model = dinv.optim.FISTA(
        data_fidelity=dinv.optim.data_fidelity.L2(),
        prior=prior,
        lambda_reg=lambda_rel * x_das.abs().max().item(),
        stepsize=1.0 / norm.item(),
        max_iter=max_iter,
        custom_init=lambda y, physics: (torch.zeros_like(x_das),) * 2,
    )
    with torch.no_grad():
        return model(y, physics)


x_ls = regularized_beamforming(lambda_rel=0.0)
x_hat = regularized_beamforming(lambda_rel=0.03)

image_ls = to_image(x_ls, inputs)
image_hat = to_image(x_hat, inputs)

# %%
# Results
# -------
# All images are histogram-matched to the 21-angle CPWC reference.
#
# - **Least squares** inverts the operator without any prior. It sharpens the
#   speckle somewhat, but the lumen is as cluttered as with DAS: the clutter is
#   consistent with the measurements, so data fidelity alone cannot remove it.
# - **Wavelet** :math:`\ell_1` keeps only the image content that is both
#   consistent with the data and sparse in wavelets. The clutter in the lumen
#   is strongly reduced, the vessel walls stand out, and the lumen contrast
#   of a single plane wave exceeds that of 21-angle compounding.

images = [image_das, image_ls, image_hat, image_cpwc]
titles = [
    "DAS\n1 plane wave",
    "Least squares\n1 plane wave",
    r"Wavelet $\ell_1$ (FISTA)" "\n1 plane wave",
    f"DAS (CPWC)\n{n_tx} plane waves",
]
plot_images(images, titles)

# %%
# Zoom on the vessel (10-30 mm depth):

plot_images(images, titles, zoom=((-15.0, 15.0), (30.0, 10.0)))

# %%
# Distortion metrics
# ------------------
# Finally, we compare each single-plane-wave image with the 21-angle CPWC image
# using the full-reference metrics PSNR and SSIM of :mod:`deepinv.loss.metric`,
# on the histogram-matched B-mode images scaled to :math:`[0, 1]`. These scores
# stay modest, because a different acquisition produces a different speckle
# pattern even for the same tissue, but the wavelet image is the closest to the
# compounded one. The table also repeats the gCNR of the figures.


def to_tensor(image):
    """Displayed (histogram-matched) image -> (1, 1, n_z, n_x) tensor in [0, 1]."""
    b = torch.from_numpy(to_display(image)).float() / 255
    return b[None, None].to(device)


psnr, ssim = dinv.metric.PSNR(), dinv.metric.SSIM()
b_ref = to_tensor(image_cpwc)
names = ["DAS, 1 PW", "Least squares, 1 PW", "Wavelet, 1 PW", f"CPWC, {n_tx} PW"]
for name, image in zip(names, images, strict=True):
    line = f"{name:20s} gCNR {gcnr(image):.2f}"
    if image is not image_cpwc:
        b = to_tensor(image)
        line += (
            f" | PSNR {psnr(b, b_ref).item():5.2f} dB"
            f" | SSIM {ssim(b, b_ref).item():.3f}"
        )
    print(line)

# %%
# What's next?
# ------------
#
# A wavelet sparsity prior already brings single-plane-wave images close to
# 21-angle compounding, at the cost of 100 iterations of the zea beamformer.
# Some directions to go further:
#
# - Use several plane waves: stacking the per-angle operators with
#   :class:`deepinv.physics.StackedLinearPhysics` gives a joint reconstruction
#   from a few angles.
# - Replace the wavelet prior with a learned denoiser in a plug-and-play
#   algorithm (:class:`deepinv.optim.PnP`), or unroll FISTA and train it end to
#   end (:func:`deepinv.unfolded.unfolded_builder`).
# - Refine the forward model, for example with the pulse shape or the element
#   directivity :ref:`[Besson2018] <references>`, or add priors on the envelope
#   and the spectrum of the image :ref:`[Ozkan2018] <references>`.
# - Go off the grid: instead of fixing the scatterer positions on a pixel grid,
#   optimize their positions and amplitudes together with the parameters of the
#   forward model itself, such as the waveform, by stochastic optimization
#   :ref:`[vandeSchaft2025] <references>`.

# %%
# .. _references:
#
# References
# ----------
#
# .. [Stevens2026] Stevens, T.S.W., van Nierop, W.L., Luijten, B., van de Schaft, V.,
#    Nolan, O.I., Federici, B., van Harten, L.D., Penninga, S.W., Schueler, N.I.P. &
#    van Sloun, R.J.G. (2026). zea: A Toolbox for Cognitive Ultrasound Imaging.
#    *Journal of Open Source Software*, 11(121):9881.
#    DOI: `10.21105/joss.09881 <https://doi.org/10.21105/joss.09881>`_
#
# .. [Besson2018] Besson, A., Perdios, D., Martinez, F., Chen, Z., Carrillo, R.E.,
#    Arditi, M., Wiaux, Y. & Thiran, J.-P. (2018). Ultrafast ultrasound imaging as an
#    inverse problem: matrix-free sparse image reconstruction.
#    *IEEE Trans. Ultrason. Ferroelectr. Freq. Control*, 65(3):339--355.
#    DOI: `10.1109/TUFFC.2017.2768583 <https://doi.org/10.1109/TUFFC.2017.2768583>`_
#
# .. [Ozkan2018] Ozkan, E., Vishnevsky, V. & Goksel, O. (2018). Inverse problem of
#    ultrasound beamforming with sparsity constraints and regularization.
#    *IEEE Trans. Ultrason. Ferroelectr. Freq. Control*, 65(3):356--365.
#    DOI: `10.1109/TUFFC.2017.2757880 <https://doi.org/10.1109/TUFFC.2017.2757880>`_
#
# .. [vandeSchaft2025] van de Schaft, V., Nolan, O. & van Sloun, R.J.G. (2025).
#    Off-grid ultrasound imaging by stochastic optimization.
#    *IEEE Trans. Ultrason. Ferroelectr. Freq. Control*, 72(9):1245--1255.
#    DOI: `10.1109/TUFFC.2025.3586377 <https://doi.org/10.1109/TUFFC.2025.3586377>`_
#
# .. [Montaldo2009] Montaldo, G., Tanter, M., Bercoff, J., Benech, N. & Fink, M. (2009).
#    Coherent plane-wave compounding for very high frame rate ultrasonography and
#    transient elastography. *IEEE Trans. Ultrason. Ferroelectr. Freq. Control*,
#    56(3):489--506.
#    DOI: `10.1109/TUFFC.2009.1067 <https://doi.org/10.1109/TUFFC.2009.1067>`_
#
# .. [Bottenus2021] Bottenus, N., Byram, B.C. & Hyun, D. (2021). Histogram matching
#    for visual ultrasound image comparison.
#    *IEEE Trans. Ultrason. Ferroelectr. Freq. Control*, 68(5):1487--1495.
#    DOI: `10.1109/TUFFC.2020.3035965 <https://doi.org/10.1109/TUFFC.2020.3035965>`_
#
# .. [RodriguezMolares2020] Rodriguez-Molares, A., Rindal, O.M.H., D'hooge, J.,
#    Måsøy, S.-E., Austeng, A., Lediju Bell, M.A. & Torp, H. (2020). The generalized
#    contrast-to-noise ratio: a formal definition for lesion detectability.
#    *IEEE Trans. Ultrason. Ferroelectr. Freq. Control*, 67(4):745--759.
#    DOI: `10.1109/TUFFC.2019.2956855 <https://doi.org/10.1109/TUFFC.2019.2956855>`_
