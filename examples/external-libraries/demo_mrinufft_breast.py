r"""
Reconstruct accelerated non-Cartesian breast MRI acquisition data
=================================================================

This example reconstructs raw non-Cartesian multicoil kspace data from the FastMRI breast dataset :footcite:p:`solomonFastMRI2025`, for mammography.

The data was acquired using radial sampling, with 288 spokes. We compare image reconstruction
using all spokes vs. with 72 spokes (i.e. 4x acceleration).
We model the 2D non-uniform FFT physics with :class:`deepinv.physics.NonCartesianMRI`, which uses `MRI-NUFFT <https://github.com/mind-inria/mri-nufft>`_.

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

import importlib

if importlib.util.find_spec("mrinufft") is None:
    raise ImportError(
        "mri-nufft is required for NonCartesianMRI. Install with `pip install mrinufft[finufft]` (CPU or MPS) or `pip install mrinufft[cufinufft]` (GPU)."
    )

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
# The data is originally provided as `h5` files, where each file consists of raw kspace for one patient volume, of shape:
#
# - `C` = channels = 2,
# - `S` = number of spokes = 288,
# - `Y` = number of samples per spoke = 640,
# - `N` = num coils = 16,
# - `P` = partitions = 83 (this will become part of slices = 192)
#
# The data was acquired with a Cartesian fully-sampled partition (slice) readout, and is stored in the frequency domain.
# Therefore, preprocessing steps are needed to zero-fill the partition axis, then iFFT the partition axis, then take a 2D slice.
#
# For the demo, we perform these steps offline and provide on HuggingFace a sample slice available to download.
# To reproduce this slice preprocessing, you can run the following code::
#
#     import h5py
#     with h5py.File("/path/to/fastMRI_breast_001_1.h5", "r") as f:
#         y = torch.from_numpy(f["kspace"][:, :, :]).float() # C S Y N P

#     y = torch.complex(y[0], y[1]).permute(3, 2, 0, 1) # P N S Y
#     P, N, num_shots, num_samples = y.shape

#     shift = 192 // 2 - 31

#     W = dinv.utils.MRIMixin.ifft(torch.eye(102, dtype=y.dtype), dim=(0,)) # Z,Z centered ifft basis
#     y = torch.einsum("p,pnsy->nsy", W[96, shift : shift + P], y) # Take middle slice -> N S Y
#     y = dinv.utils.MRIMixin.to_torch_complex(y).permute(1, 2, 3, 0) # 2SYN

dinv.utils.download_example(
    "fastMRI_breast_001_1_slice_96.pt", dinv.utils.get_data_home() / "fastMRI_breast"
)

y = torch.load(
    dinv.utils.get_data_home() / "fastMRI_breast" / "fastMRI_breast_001_1_slice_96.pt"
).to(
    device
)  # 2SYN

print("Provided slice shape (2, num shots, num samples per shot, num coils):", y.shape)

# %%
# The final kspace should be of shape `(1,2,N,S)` to be used with :class:`deepinv.physics.NonCartesianMRI`.

y = (
    y.reshape(2, y.shape[1] * y.shape[2], y.shape[3]).swapaxes(-2, -1).unsqueeze(0)
)  # 1, 2, N, S*Y

print("Ready slice shape (1, 2, N, S*Y):", y.shape)

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
    coil_maps=y.shape[2],  # dummy coil maps, not used for RSS
    trajectory="radial",
    tilt="golden",
    in_out=True,
    density_mode="compensate",
    backend=backend,
    device=device,
)

with torch.no_grad():
    x = physics_fs.A_adjoint(y, rss=True)  # 1, H, W

# %%
# Reconstruct accelerated data
# ----------------------------
#
# We undersample the radial data by taking the first few (golden angle) shots.
# We construct non-Cartesian MRI without density compensation. We empirically normalise the physics for
# solvers that require this.
# Then, we estimate the coil sensitivity maps using ESPIRiT (on a lower dimensional image).
#

undersampling_factor = 4

y = y[..., : y.shape[-1] // undersampling_factor]

physics = dinv.physics.NonCartesianMRI(
    img_size=(320, 320),
    num_shots=288 // undersampling_factor,
    num_samples_per_shot=640,
    coil_maps=y.shape[2],  # dummy
    trajectory="radial",
    tilt="golden",
    in_out=True,
    density_mode=None,
    backend=backend,
    device=device,
    normalize=True,
    noise_model=dinv.physics.GaussianNoise(0.001),
)

coil_maps = physics.estimate_coil_maps(y, method="espirit", decim=4)
physics.update(coil_maps=coil_maps)

# %%
# Reconstruct with conjugate-gradient
# -----------------------------------
# We reconstruct the data with the conjugate-gradient algorithm, which gives a least-squares solution.
# Notice that streak artifacts are present, which are expected for CG on undersampled data.
#
# .. note::
#     The target x is on wrong scale for metrics since it uses density compensation which doesn't preserve norm

with torch.no_grad():
    x_cg = physics.A_dagger(y)

dinv.utils.plot(
    [x, x_cg],
    titles=[
        "Fully-sampled RSS",
        "Conjugate-gradient 4x acc",
    ],
)

# %%
# What's next?
# ------------
#
# We didn't show any model-based on deep learning reconstruction methods in this example to keep the example lightweight.
# You can try out other types of reconstruction algorithms listed in the :ref:`user guide <reconstructors>`.
