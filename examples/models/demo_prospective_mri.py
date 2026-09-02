r"""
Reconstruct prospectively-undersampled raw multi-coil MRI with RAM
==================================================================

This example reconstructs real prospectively (compressed-sensing) undersampled 3D multi-coil
brain k-space from Yu et al., stored as raw ISMRMRD, with the foundation model
:class:`deepinv.models.RAM`. There is no ground truth: the data was acquired undersampled.

We load the raw k-space, take a 2D slice, recover the sampling mask and estimate coil maps to
build a :class:`deepinv.physics.MultiCoilMRI` operator, then reconstruct with RAM.

This example requires ISMRMRD. Install it with ``pip install ismrmrd``.
"""

# %%
import torch
import deepinv as dinv
from ram_experiments.datasets.io import load_ismrmrd_raw
from ram_experiments.physics.mri import phase_correct_maps

device = dinv.utils.get_device()

# %%
# Load the raw k-space
# --------------------
#
# :func:`ram_experiments.datasets.io.load_ismrmrd_raw` grids the acquired lines onto a Cartesian
# k-space ``(1, 2, N, D, H, W)`` (N coils). We inverse-FFT the fully-sampled readout/slice dimension
# ``D`` (``ifft_slice_dim=True``) and take the middle slice to obtain a 2D multi-coil k-space.

y = load_ismrmrd_raw("/Volumes/E/ram-experiments/data/yu_melba/t2_space_fs_sag_cs7_iso.h5", ifft_slice_dim=True)
y = y[..., y.shape[-1] // 2, :, :].to(device)  # middle slice -> (1, 2, N, H, W)

# %%
# Build the physics
# -----------------
#
# We recover the prospective sampling mask from the k-space zeros, estimate ESPIRiT coil maps,
# and normalize the k-space by the 99th percentile of its RSS reconstruction. RAM uses the physics
# coil maps, so we phase-correct them against the zero-filled image with
# :func:`ram_experiments.physics.mri.phase_correct_maps`.

mask = (y != 0).any(1).any(1, keepdim=True).float()  # (1, 1, H, W)
coil_maps = dinv.physics.MultiCoilMRI.estimate_coil_maps(y, calib_size=24, espirit_crop=0.99)  # (1, N, H, W) complex

M = dinv.utils.MRIMixin()
y = y / torch.quantile(M.rss(M.kspace_to_im(y)), 0.99)
physics = dinv.physics.MultiCoilMRI(mask=mask, coil_maps=coil_maps, device=device)
with torch.no_grad():
    physics.update(coil_maps=phase_correct_maps(physics.coil_maps, physics.A_adjoint(y)))

# %%
# Reconstruct with RAM
# --------------------

model = dinv.models.RAM(device=device, pretrained=True)
with torch.inference_mode():
    x_ram = model(y, physics)

dinv.utils.plot([physics.A_dagger(y), x_ram], titles=["zero-filled", "RAM"])
