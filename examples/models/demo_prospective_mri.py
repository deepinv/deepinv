r"""
Reconstruct prospectively-undersampled raw multicoil MRI
========================================================

This example reconstructs real prospectively undersampled multicoil brain k-space from Yu et al TODO CITE.

This demonstrates the performance of reconstruction models (:class:`deepinv.models.RAM`) in a deployment scenario rather than a typical simulated (retrospective) scenario.

The data is stored in the raw ISMRMRD format.

.. note::
    This example requires `ismrmrd` to load the data. Install it with `pip install ismrmrd`.
"""

# %%
import torch
import deepinv as dinv

device = dinv.utils.get_device()

# %%
# Load the raw k-space
# --------------------
#
# The data is stored in the ISMRMRD format, as 3D multicoil kspace of shape `(1, 2, N, D, H, W)`.
# We inverse-FFT the fully-sampled readout/slice dimension `D` and take the middle slice to obtain a 2D multicoil kspace
# for demonstration purposes.

dinv.datasets.download_archive(
    dinv.utils.get_image_url("t2_space_fs_sag_cs7_iso.h5"),
    dinv.utils.get_cache_home() / "mridata" / "prospective_t2.h5",
)

y = dinv.io.load_ismrmrd_raw(
    dinv.utils.get_cache_home() / "mridata" / "prospective_t2.h5", ifft_slice_dim=True
)
y = y[..., y.shape[-1] // 2, :, :].to(device)

# %%
# Build the physics
# -----------------
#
# We recover the prospective sampling mask from the kspace zeros
# We also estimate coil maps using ESPIRiT.
# We estimate manually the noise level as `sigma=0.02`. Decreasing it increases the noise in the reconstruction, whereas
# increasing it increases the smoothness in the reconstruction.

mask = (y != 0).any(1).any(1, keepdim=True).float()  # (1, 1, H, W)
coil_maps = dinv.physics.MultiCoilMRI.estimate_coil_maps(
    y, calib_size=24, espirit_crop=0.99
)  # (1, N, H, W) complex

physics = dinv.physics.MultiCoilMRI(
    mask=mask,
    coil_maps=coil_maps,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=0.02),
)

# %%
# Reconstruct with RAM
# --------------------

model = dinv.models.RAM(device=device, pretrained=True)

with torch.no_grad():
    x_zf = physics.A_adjoint(y)
    physics.phase_correct_maps(x_zf)
    x_ram = model(y / x_zf.max(), physics) * x_zf.max()
    x_sense = physics.A_dagger(y)

dinv.utils.plot([x_zf, x_sense, x_ram], titles=["Zero-filled", "SENSE", "RAM"])
