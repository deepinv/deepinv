r"""
Reconstruct accelerated non-Cartesian breast MRI acquisition data
=================================================================

This example reconstructs non-Cartesian multicoil kspace data from the FastMRI breast dataset TODO CITE, for mammography.

The data was acquired using radial sampling, with 288 spokes. We simulate 4x acceleration (i.e. reconstruct from 72 spokes)
and model the non-uniform FFT physics with `MRI-NUFFT <https://github.com/mind-inria/mri-nufft>`_.

.. note::
    This example requires the `mri-nufft` library to model the physics.
    Install with `pip install mrinufft[finufft]` (CPU or MPS) or `pip install mrinufft[cufinufft]` (GPU).

    You can choose between the various backends, see mri-nufft docs for more details. We suggest using `backend='cufinufft'` for cuda devices,
    or `backend='finufft'` for CPU. For MPS, use `backend='mps'`, which uses `finufft` but bypasses a torch multithreading problem.
"""

# %%
import torch
import deepinv as dinv

device = dinv.utils.get_device()

try:
    import mrinufft
except ImportError as e:
    raise ImportError("mri-nufft is required for NonCartesianMRI. Install with `pip install mrinufft[finufft]` (CPU or MPS) or `pip install mrinufft[cufinufft]` (GPU).") from e

if torch.device(device).type == "cuda":
    backend = "cufinufft"
elif torch.device(device).type == "mps":
    backend = "mps"
else:
    backend = "finufft"


# %%
# Load the non-Cartesian data
# ---------------------------
#
# The data is stored in the ISMRMRD format, as 3D multicoil kspace of shape `(1, 2, N, D, H, W)`.
# We inverse-FFT the fully-sampled readout/slice dimension `D` and take the middle slice to obtain a 2D multicoil kspace
# for demonstration purposes.

# dinv.utils.download_example("fastMRI_breast_001_1_slice_96.pt", dinv.utils.get_data_home() / "fastMRI_breast")

y = torch.load(dinv.utils.get_data_home() / "fastMRI_breast" / "fastMRI_breast_001_1_slice_96.pt").to(device) # 2SYN

print(y.shape) # 2, num shots, num samples per shot, num coils

y = y.reshape(2, y.shape[1] * y.shape[2], y.shape[3]).swapaxes(-2, -1).unsqueeze(0) # 1, 2, N, S*Y

print(y.shape) # multicoil breast data should be 1, 2, N, S*Y

# %%
# Fully-sampled reconstruction
# ----------------------------
# We compute the root-sum-squares adjoint reconstruction with all 288 angles, along with
# density compensation, which is standard for non-Cartesian MRI data.
#
# The FastMRI data was acquired with golden-angle radial sampling, with 640 samples per shot.
# We use the standard reconstruction size of 320*320.

physics_fs = dinv.physics.NonCartesianMRI(
    img_size=(320, 320),
    num_shots=288,
    num_samples_per_shot=640,
    coil_maps=y.shape[2], # dummy coil maps, not used for RSS
    trajectory="radial",
    tilt='golden',
    in_out=True,
    density_mode="compensate",
    backend=backend,
    device=device,
)

with torch.no_grad():
    x = physics_fs.A_adjoint(y, rss=True) # 1, H, W

# %%
# Accelerated data

# %%
# Build the physics
# -----------------
#
# We recover the prospective sampling mask from the kspace zeros
# We also estimate coil maps using ESPIRiT.
# We estimate manually the noise level as `sigma=0.02`. Decreasing it increases the noise in the reconstruction, whereas
# increasing it increases the smoothness in the reconstruction.


undersampling_factor = 4

y = y[..., :y.shape[-1] // undersampling_factor] # undersampling by taking first few (golden angle) shots

physics = dinv.physics.NonCartesianMRI(
    img_size=(320, 320),
    num_shots=288 // undersampling_factor,
    num_samples_per_shot=640,
    coil_maps=y.shape[2], # dummy
    trajectory="radial",
    tilt='golden',
    in_out=True,
    density_mode=None,
    backend=backend,
    device=device,
    normalize=True,
    noise_model=dinv.physics.GaussianNoise(0.001),
)

# Estimate coil maps
coil_maps = physics.estimate_coil_maps(y, method='espirit', decim=4)
physics.update(coil_maps=coil_maps)

# NOTE target always on wrong scale for metrics since it uses density compensation which doesn't preserve norm

# %%
# Reconstruct with RAM
# --------------------

with torch.no_grad():
    x_cg = physics.A_dagger(y)

dinv.utils.plot([x, x_cg], titles=["Fully-sampled", "SENSE"])
