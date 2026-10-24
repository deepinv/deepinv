r"""
Ultrasound regularized beamforming with zea toolbox
=========================================================

This example shows how to use the `zea <https://zea.readthedocs.io/>`_ ultrasound
toolbox :footcite:p:`stevens2026zea` with deepinv to form ultrasound images by
solving an inverse problem, instead of with the standard delay-and-sum beamformer.

zea implements the full ultrasound processing chain, from raw channel data to
the displayed image, as a pipeline of composable and differentiable operations.
zea operations run natively in PyTorch via Keras 3 so can be used easily in deepinv.
zea also provides public ultrasound datasets on `HuggingFace <https://huggingface.co/zeahub>`__,
which we use below.

**Background.** An ultrasound probe is an array of elements that both transmit
sound pulses and record the returning echoes. In plane-wave imaging, each
transmit fires all elements at once (optionally with a small steering delay),
producing an unfocused plane wave that illuminates the whole field of view in
one shot. Every element records the echoes over time, giving a 2-D array of
signals called the **channel data**
(see `this blog post <https://www.vincentvandeschaft.net/notes/channel-data>`_
for an intuitive explanation).

**Beamforming** turns the channel data into an image. The standard method,
delay-and-sum (DAS), computes for every pixel the travel time of the echo to
each element, and sums the corresponding samples. A single plane wave gives a
fast but low-quality image. Coherent plane-wave compounding (CPWC)
:footcite:p:`montaldo2009coherent` sums the DAS images of many steered plane
waves to improve resolution and contrast, at the cost of frame rate.

**Beamforming as an inverse problem.** The channel data :math:`y` are linear in
the tissue reflectivity :math:`x` on a pixel grid, :math:`y = A x + \varepsilon`,
and DAS is exactly the adjoint :math:`A^{\top}` :footcite:p:`besson2018ultrafast`.
DAS is thus a back-projection rather than an inversion. Instead, we reconstruct
the image from a **single** plane wave by solving

.. math::

    \hat{x} = \underset{x}{\arg\min} \;
    \frac{1}{2}\|A x - y\|_2^2 + \lambda\,\|\Psi x\|_1,

where :math:`\Psi` is a wavelet transform :footcite:p:`besson2018ultrafast,ozkan2018inverse`.
We use the zea DAS beamformer as :math:`A^{\top}`, obtain :math:`A` by automatic
differentiation, and solve the problem with :class:`deepinv.optim.FISTA`.

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
# We use a recording of the carotid artery from the
# `zea carotid dataset <https://huggingface.co/datasets/zeahub/zea-carotid-2023>`_,
# acquired with a linear probe. The vessel runs horizontally through the image,
# with its dark lumen surrounded by brighter tissue. We keep the plane-wave
# transmits of one frame and define the pixel grid on which we reconstruct.

file_path = "hf://zeahub/zea-carotid-2023/data/9_long_bifur_R_0000.hdf5"

with zea.File(file_path, revision="v0.1.5") as f:
    params = f.load_parameters()
    plane_txs = params.find_transmits("plane")
    params.selected_transmits = plane_txs
    raw_data = f.data.raw_data[:1, plane_txs]  # (1, n_tx, n_ax, n_el, 1)

params.xlims = (-20e-3, 20e-3)
params.zlims = (2e-3, 40e-3)
params.pixels_per_wavelength = 2

n_tx, n_ax, n_el = raw_data.shape[1], raw_data.shape[2], raw_data.shape[3]
print(f"Transmit events: {n_tx}  |  Time samples: {n_ax}  |  Elements: {n_el}")
print(f"Pixel grid: {params.grid_size_z} x {params.grid_size_x}")

ext_mm = params.extent_imshow * 1000
DYNAMIC_RANGE = (-50, 0)  # dB

# %%
# Raw RF channel data
# -------------------
# For one plane wave, the channel data are a 2-D array of time against element.
# Each scatterer appears as a hyperbola, which beamforming aligns and sums.

rf_0deg = raw_data[0, 0, :, :, 0]

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
# We build three zea pipelines:
#
# 1. **Preprocessing**: demodulate the RF data to complex (IQ) data, stored as
#    two real channels, and downsample.
# 2. **Beamforming**: DAS, from IQ channel data to an IQ image. This is
#    :math:`A^{\top}`.
# 3. **B-mode**: envelope detection and log compression, for display.

pre_pipeline = zea.Pipeline(
    [Demodulate(), Downsample(factor=4)], jit_options=None, device=str(device)
)
das_pipeline = zea.Pipeline(
    [
        # a large patch size makes beamforming much faster on GPU
        Beamform(beamformer="delay_and_sum", enable_pfield=False, patch_size=2**15),
        Transpose(axes=(2, 0, 1)),  # I/Q channels first, as in deepinv
    ],
    jit_options=None,
    device=str(device),
)
image_pipeline = zea.Pipeline(
    [
        Transpose(axes=(1, 2, 0)),
        EnvelopeDetect(),
        Normalize(input_range=(0, None)),
        LogCompress(clip=False),
    ],
    jit_options=None,
    device=str(device),
)


def prepare_parameters(params):
    return {
        k: v
        for pipeline in [pre_pipeline, das_pipeline, image_pipeline]
        for k, v in pipeline.prepare_parameters(params).items()
    }


def rf_to_iq(rf, inputs):
    """RF channel data -> IQ channel data, and the updated parameters."""
    with torch.no_grad():
        out = pre_pipeline(data=rf.unsqueeze(0), **inputs)
    return out.pop("data"), {**inputs, **out}


def to_image(x, inputs):
    """IQ image -> B-mode image in dB."""
    with torch.no_grad():
        return image_pipeline(data=x, return_numpy=True, **inputs)["data"][0]


rf_raw = torch.tensor(raw_data[0], dtype=torch.float32, device=device)
rf_raw = rf_raw / rf_raw.abs().max()

# %%
# Our measurements: the 0 degree plane wave only. This is called single plane-wave imaging and allows for ultrafast ultrasound.
params.selected_transmits = [plane_txs[0]]
y, inputs = rf_to_iq(rf_raw[:1], prepare_parameters(params))


def das(y):
    """DAS beamforming of the single plane wave."""
    return das_pipeline(data=y, **inputs)["data"]


with torch.no_grad():
    x_das = das(y)
image_das = to_image(x_das, inputs)

print(f"Measurements y (1, n_tx, time, elements, I/Q): {tuple(y.shape)}")
print(f"Image x (1, I/Q, depth, lateral): {tuple(x_das.shape)}")

plt.figure(figsize=(5, 5))
plt.imshow(
    zea.display.to_8bit(image_das, dynamic_range=DYNAMIC_RANGE, pillow=False),
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
# As reference we use DAS on all plane waves combined (CPWC).
# To compare methods fairly, we
# **histogram-match** every image to this reference :footcite:p:`bottenus2021histogram`,
# so that only the image structure differs.
# We measure how well the dark lumen stands out from the tissue with the
# generalized contrast-to-noise ratio (gCNR) :footcite:p:`rodriguezmolares2020generalized`,
# which ranges from 0 (indistinguishable) to 1 (perfectly separable).
# The lumen is segmented automatically from the reference image.

params.selected_transmits = plane_txs
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

TISSUE = (-15.0, 15.0, 8.0, 14.0)  # x0, x1, z0, z1 (mm)
tissue = (
    (x_mm[None] >= TISSUE[0])
    & (x_mm[None] <= TISSUE[1])
    & (z_mm[:, None] >= TISSUE[2])
    & (z_mm[:, None] <= TISSUE[3])
)


def to_display(image):
    """Histogram-match an image to the CPWC reference and convert to 8-bit."""
    matched = zea.display.histogram_match(image, image_cpwc)
    return zea.display.to_8bit(matched, dynamic_range=DYNAMIC_RANGE, pillow=False)


def gcnr(image):
    """gCNR between the lumen and tissue regions."""
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


def plot_images(images, titles, zoom=None, arrows=None):
    """Plot images side by side. ``arrows`` maps a subplot index to (tail, head) pairs in mm."""
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
        tick = 5 if zoom is not None else 10
        ax.xaxis.set_major_locator(MultipleLocator(tick))
        ax.yaxis.set_major_locator(MultipleLocator(tick))
        ax.set_xlabel("Lateral (mm)")
    for i, pairs in (arrows or {}).items():
        for tail, head in pairs:
            axes[i].annotate(
                "",
                xy=head,
                xytext=tail,
                arrowprops=dict(color="red", width=2, headwidth=8),
            )
    axes[0].set_ylabel("Depth (mm)")
    plt.tight_layout()
    plt.show()


# %%
# From one angle to coherent compounding
# --------------------------------------
# A single plane wave leaves clutter in the dark lumen. Compounding all plane
# waves removes much of it, but needs one transmit per angle and thus lowers the
# frame rate. Our goal is to approach the compounded image from a single plane
# wave.

plot_images([image_das, image_cpwc], ["DAS, 1 plane wave", f"DAS, {n_tx} plane waves"])

# %%
# The forward operator
# --------------------
# Since the zea beamformer is differentiable, we do not implement :math:`A` by
# hand: :func:`deepinv.physics.adjoint_function` computes it as the adjoint of
# DAS with automatic differentiation.

A = dinv.physics.adjoint_function(das, y.shape, device=device)
physics = dinv.physics.LinearPhysics(A=A, A_adjoint=das)

# Check that A and DAS are adjoint.
x_rand = torch.randn_like(x_das)
y_rand = torch.randn_like(y)
with torch.no_grad():
    lhs = (physics.A(x_rand) * y_rand).sum().item()
    rhs = (x_rand * physics.A_adjoint(y_rand)).sum().item()
print(f"Adjoint test: <A x, y> = {lhs:.4f}, <x, DAS y> = {rhs:.4f}")

# Operator norm, for the step size.
with torch.no_grad():
    norm = physics.compute_norm(torch.randn_like(x_das), tol=1e-3, verbose=False)
print(f"||A^T A|| = {norm.item():.1f}")

# %%
# Regularized beamforming with FISTA
# ----------------------------------
# Each FISTA iteration costs one forward and one adjoint beamforming.
# Following :footcite:t:`besson2018ultrafast`, the prior :math:`\Psi` concatenates
# several Daubechies wavelets, which :class:`deepinv.optim.WaveletPrior` supports
# by passing a list of wavelets. For comparison, we also solve the unregularized
# least-squares problem (:math:`\lambda = 0`).


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
# Least squares alone cannot remove the clutter in the lumen, since it is
# consistent with the measurements. The wavelet prior removes most of it, and
# reaches a higher lumen contrast than compounding all plane waves.

images = [image_das, image_ls, image_hat, image_cpwc]
titles = [
    "DAS\n1 plane wave",
    "Least squares\n1 plane wave",
    r"Wavelet $\ell_1$ (FISTA)" "\n1 plane wave",
    f"DAS (CPWC)\n{n_tx} plane waves",
]
plot_images(
    images,
    titles,
    arrows={0: [((-4.0, 20.0), (-8.2, 24.9))], 1: [((-7.0, 24.0), (2.0, 24.0))]},
)

# %%
# Zoom on the vessel:

plot_images(images, titles, zoom=((-15.0, 15.0), (30.0, 10.0)))

# %%
# Distortion metrics
# ------------------
# We also compare each single-plane-wave image to the compounded reference with
# :class:`PSNR <deepinv.loss.metric.PSNR>` and :class:`SSIM <deepinv.loss.metric.SSIM>`.
# Scores stay modest because a different acquisition produces different speckle,
# but the wavelet reconstruction is the closest.


def to_tensor(image):
    """Displayed image as a tensor in [0, 1]."""
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
# Possible extensions
# -------------------
#
# - Replace the wavelet prior with a learned denoiser
#   (:class:`deepinv.optim.PnP`), or unroll and train the algorithm
#   (:func:`deepinv.unfolded.unfolded_builder`).
# - Refine the forward model :footcite:p:`besson2018ultrafast,ozkan2018inverse`,
#   or optimize scatterer positions off the grid :footcite:p:`vandeschaft2025offgrid`.
# - Try other data, such as the `Open-H-RF <https://open-h.github.io/Open-H-RF/>`_
#   dataset, stored in the same zea format.

# %%
# :References:
#
# .. footbibliography::
