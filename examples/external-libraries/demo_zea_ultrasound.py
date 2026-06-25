"""
Ultrasound reconstruction from raw RF data with zea and deepinv
=====================================================================

This example demonstrates recovering raw RF channel data :math:`x` from a
beamformed ultrasound image :math:`y`, using the
`zea <https://zea.readthedocs.io/>`_ ultrasound toolbox as the forward
operator inside `deepinv <https://deepinv.github.io/>`_.

.. note::

    Work in progress!  This example is currently being developed and
    is not be fully functional yet.

**Setup — KERAS_BACKEND must be set before importing keras / zea:**

.. code-block:: bash

    KERAS_BACKEND=torch python demo_zea_ultrasound.py

or in Python **before any other imports**:

.. code-block:: python

    import os
    os.environ["KERAS_BACKEND"] = "torch"

**Inverse problem formulation:**

The DAS beamforming pipeline defines a linear forward operator
:math:`A : x \\mapsto y` where

- :math:`x` — raw RF channel data, shape ``(batch, n_tx, n_ax, n_el, 1)``
- :math:`y` — beamformed IQ image (before envelope), shape ``(batch, iq_channels, grid_z, grid_x)``

We solve

.. math::

    \\hat{x} = \\underset{x}{\\arg\\min} \\;
    \\frac{1}{2}\\|Ax - y\\|_2^2 + \\lambda \\|x\\|_1

with PGD (proximal gradient descent / ISTA).  The L1 prior promotes sparse
tissue reflectors in the raw RF data.

**Two-pipeline design:**

1. **Physics pipeline** (``Demodulate → Downsample → Beamform``):
   the linear, differentiable forward operator used in the inverse loop.
   This is implemented using the zea toolbox and wrapped with
    :class:`~deepinv.physics.LinearPhysics`.
2. **Visualisation pipeline** (``EnvelopeDetect → Normalize → LogCompress``):
   applied only for display; not part of the optimisation.

**Re-beamforming with a different beamformer:**

Because we recover the raw RF data :math:`\\hat{x}`, we can re-beamform
with *any* algorithm.  Here we apply Delay-Multiply-and-Sum (DMAS), on the
recovered raw channel data :math:`\\hat{x}`. DMAS improves side-lobe suppression
compared to DAS. The DMAS pipeline is applied only
for display, not in the inverse problem.
"""

# %%
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import matplotlib.pyplot as plt
import numpy as np
import torch
import zea
from zea.ops import (
    Beamform,
    Companding,
    Demodulate,
    Downsample,
    EnvelopeDetect,
    LogCompress,
    Normalize,
)

import deepinv as dinv
from deepinv.physics.ultrasound import UltrasoundBeamformingWithZea

zea.visualize.set_mpl_style()

device = dinv.utils.get_device()

# %%
# Load data and scan parameters
# ------------------------------
# Scan parameters (grid, delays, probe geometry, etc.) are loaded from the
# acquisition file and config.  The pipeline *operations* are built
# explicitly below, rather than from ``Pipeline.from_config``.

file_path = "hf://zeahub/zea-carotid-2023/data/2_cross_bifur_right_0000_small.hdf5"

config = zea.Config.from_path(
    "hf://zeahub/zea-carotid-2023/config.yaml", revision="v0.1.0"
)
config.parameters.selected_transmits = "plane"  # select plane-wave transmits only

with zea.File(file_path, revision="v0.1.0") as f:
    parameters = f.load_parameters()
    parameters.update(**config.parameters)
    raw_data = f.data.raw_data[:1, parameters.selected_transmits]
    # raw_data shape: (1, n_tx, n_ax, n_el, 1)

# %%
# Build the physics (beamforming) pipeline — DAS
# ------------------------------------------------
# The forward operator :math:`A` chains three explicit operations:
#
#   ``Demodulate  →  Downsample(4×)  →  Beamform(DAS)``

das_pipeline = zea.Pipeline(
    operations=[
        Demodulate(),
        Downsample(factor=4),
        Beamform(
            beamformer="delay_and_sum",
            enable_pfield=False,
            num_patches=300,
        ),
    ],
    device=str(device),
    jit_options=None,
)

# %%
# Build the visualisation pipeline
# ----------------------------------
# This non-linear chain converts beamformed IQ to a log-compressed B-mode
# image.  It is applied only for display — it is never part of the inverse
# problem.
#
#   ``EnvelopeDetect  →  Normalize  →  LogCompress``

viz_pipeline = zea.Pipeline(
    operations=[
        EnvelopeDetect(),
        Normalize(),
        LogCompress(),
    ],
    device=str(device),
    jit_options=None,
)

# %%
# Set up the physics operator
# ---------------------------
# Wrap the DAS pipeline as a deepinv :class:`~deepinv.physics.LinearPhysics`.
# ``img_size`` is the shape of :math:`x` *without* the batch dimension.
# The adjoint :math:`A^\top` is computed automatically via deepinv's
# built-in ``torch.func.vjp`` mechanism.

img_size = tuple(raw_data[0].shape)  # (n_tx, n_ax, n_el, 1)

physics = UltrasoundBeamformingWithZea(
    pipeline=das_pipeline,
    parameters=parameters,
    img_size=img_size,
    device=str(device),
)

# %%
# Forward pass: raw RF data → beamformed IQ image
# ------------------------------------------------
# Raw ADC data has amplitudes up to ~16 384 (int16 range).  Normalising to
# ``[-1, 1]`` keeps the L1 regularisation weight scale-independent.

x = torch.tensor(raw_data[0], dtype=torch.float32).unsqueeze(0).to(device)
x = x / x.abs().max()  # normalise to [-1, 1]; does not affect BMode output
# x shape: (1, n_tx, n_ax, n_el, 1)

with torch.no_grad():
    y = physics.A(x)  # (1, iq_channels, grid_z, grid_x)

# %%
# Reference B-mode from the original RF
# ----------------------------------------
viz_inputs = viz_pipeline.prepare_parameters(parameters)

# viz_pipeline expects channel-last: (batch, grid_z, grid_x, n_ch)
with torch.no_grad():
    bmode_das_ref = viz_pipeline(data=y.permute(0, 2, 3, 1), **viz_inputs)["data"]
    # bmode_das_ref shape: (1, grid_z, grid_x)

# %%
# Solve the inverse problem: recover raw RF from the beamformed image
# -------------------------------------------------------------------
# Minimise  ``(1/2)||A(x) - y||^2 + λ||x||_1``  using PGD (ISTA).
#
# The L1 prior promotes sparsity in the raw RF domain, corresponding to
# discrete, localised tissue reflectors.  Backtracking line-search adapts
# the step size automatically.

data_fidelity = dinv.optim.data_fidelity.L2()
prior = dinv.optim.prior.L1Prior()

model = dinv.optim.PGD(
    data_fidelity=data_fidelity,
    prior=prior,
    lambda_reg=1e-3,
    stepsize=1.0,
    backtracking=True,
    max_iter=50,
    early_stop=True,
    thres_conv=1e-4,
)

with torch.no_grad():
    x_hat = model(y, physics)  # (1, n_tx, n_ax, n_el, 1)

# %%
# Re-beamform the recovered RF with DAS
# -----------------------------------------
# Apply :math:`A` to :math:`\\hat{x}` to produce the DAS B-mode from the
# recovered data.  This should match the reference closely.

with torch.no_grad():
    y_hat_das = physics.A(x_hat)
    bmode_das_recon = viz_pipeline(data=y_hat_das.permute(0, 2, 3, 1), **viz_inputs)[
        "data"
    ]

# %%
# Re-beamform the recovered RF with DMAS
# -----------------------------------------
# Delay-Multiply-and-Sum (DMAS) improves side-lobe suppression compared to
# standard DAS at the cost of extra computation.  Because we have recovered
# the raw RF :math:`\\hat{x}`, we can apply *any* beamformer — including
# ones not used in the inverse problem.

dmas_pipeline = zea.Pipeline(
    operations=[
        Demodulate(),
        Downsample(factor=4),
        Beamform(
            beamformer="delay_multiply_and_sum", enable_pfield=False, num_patches=300
        ),
    ],
    device=str(device),
    jit_options=None,
)

dmas_inputs = dmas_pipeline.prepare_parameters(parameters)

with torch.no_grad():
    # dmas_pipeline output is channel-last (zea convention): no permute needed
    y_hat_dmas = dmas_pipeline(data=x_hat, **dmas_inputs)["data"]
    bmode_dmas_recon = viz_pipeline(data=y_hat_dmas, **viz_inputs)["data"]

# %%
# Visualise: B-mode images and RF channel data
# ---------------------------------------------
# μ-law companding compresses the dynamic range while preserving sign, making
# both weak and strong reflectors visible simultaneously.

_companding = Companding()


def compand_rf(arr):
    """Normalise to [-1, 1] then apply zea μ-law companding."""
    arr_norm = arr / (float(np.abs(arr).max()) + 1e-8)
    return (
        _companding.call(data=torch.tensor(arr_norm, dtype=torch.float32))["data"]
        .cpu()
        .numpy()
    )


tx_mid = x.shape[1] // 2  # centre plane-wave transmit
# Sub-sample axially by 4 (matches Downsample(4) in the pipeline) for display
rf_orig = x[0, tx_mid, ::4, :, 0].cpu().numpy()  # (n_ax//4, n_el)
rf_recon = x_hat[0, tx_mid, ::4, :, 0].cpu().numpy()

# Spatial extent for B-mode imshow axes (mm)
ext_mm = (parameters.extent_imshow * 1000).tolist()  # [x_min, x_max, z_max, z_min]

# ── Figure layout ───────────────────────────────────
fig = plt.figure(figsize=(14, 12))

gs_top = fig.add_gridspec(
    1, 3, left=0.05, right=0.98, top=0.96, bottom=0.52, wspace=0.25
)
gs_bot = fig.add_gridspec(
    1, 2, left=0.08, right=0.95, top=0.44, bottom=0.06, wspace=0.35
)

ax_bmode = [fig.add_subplot(gs_top[0, i]) for i in range(3)]
ax_rf_orig = fig.add_subplot(gs_bot[0, 0])
ax_rf_recon = fig.add_subplot(gs_bot[0, 1])

# ── B-mode panels ──────────────────────────────────
for ax, bmode, title in zip(
    ax_bmode,
    [bmode_das_ref, bmode_das_recon, bmode_dmas_recon],
    ["DAS — original RF", "DAS — recovered RF", "DMAS — recovered RF"],
    strict=False,
):
    ax.imshow(
        bmode[0].cpu().numpy(),
        cmap="gray",
        aspect="auto",
        origin="upper",
        extent=ext_mm,
    )
    ax.set_title(title)
    ax.set_xlabel("Lateral (mm)")
    ax.set_ylabel("Depth (mm)")

# ── RF channel images ──────────────────────────────
for ax, rf, title in zip(
    [ax_rf_orig, ax_rf_recon],
    [rf_orig, rf_recon],
    [f"RF — original (tx {tx_mid})", f"RF — recovered (tx {tx_mid})"],
    strict=False,
):
    ax.imshow(
        compand_rf(rf),
        aspect="auto",
        cmap="viridis",
        vmin=-1,
        vmax=1,
        origin="upper",
        interpolation="none",
    )
    ax.set_title(title)
    ax.set_xlabel("Element")
    ax.set_ylabel("Depth (sample ÷ 4)")

plt.savefig("demo_zea_ultrasound.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Reconstruction quality: NMSE and LPIPS
# -----------------------------------------

from deepinv.loss.metric import LPIPS, NMSE

nmse = NMSE()


# LPIPS requires 3-channel images in [0, 1]
def bmode_for_lpips(bmode):
    """(1, H, W) → (1, 3, H, W) min-max normalised."""
    b = bmode.unsqueeze(1).float()  # (1, 1, H, W)
    b = b - b.min()
    b = b / b.max().clamp(min=1e-8)
    return b.expand(-1, 3, -1, -1).contiguous()


lpips = LPIPS(device=device, check_input_range=False)

with torch.no_grad():
    nmse_val = nmse(x_hat, x).mean().item()
    lpips_val = (
        lpips(
            bmode_for_lpips(bmode_das_recon),
            bmode_for_lpips(bmode_das_ref),
        )
        .mean()
        .item()
    )

print(f"NMSE  (raw RF domain)  : {nmse_val:.4f}  (0 = perfect)")
print(f"LPIPS (B-mode, DAS)    : {lpips_val:.4f}  (0 = identical)")
