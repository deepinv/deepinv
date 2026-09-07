r"""
Reconstruct prospectively-undersampled raw multicoil MRI
========================================================

This example reconstructs real prospectively undersampled multicoil brain k-space from :footcite:t:`yu2022validation`.

This demonstrates the performance of reconstruction algorithms in a deployment scenario rather than a typical simulated (retrospective) scenario.
Here, it is impossible to compute full-reference metrics, since fully-sampled ground-truth does not exist.

We compare two types of reconstruction:

* Deep learning, using a pretrained general model :class:`deepinv.models.RAM` from :footcite:t:`terris2025reconstruct`
* Compressed sensing, using FISTA with a wavelet prior :footcite:p:`chambolle2015convergence`

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
# We also estimate coil maps using ESPIRiT :footcite:p:`uecker2013espirit`.
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
# Baseline reconstructions
# ------------------------
# As with any MRI problem, the baselines can be considered to be the zero-filled reconstruction, and the least-squares conjugate-gradient SENSE :footcite:p:`pruessmann1999sense`.

with torch.no_grad():
    x_zf = physics.A_adjoint(y)
    x_sense = physics.A_dagger(y)

# %%
# Deep learning reconstruction with RAM
# -------------------------------------
# Reconstruct Anything Model :footcite:p:`terris2025reconstruct` was not trained on multicoil MRI physics, nor on axial knee MRI slices. Here, we therefore
# test its generalisability.
#
# .. note::
#     ESPIRiT estimates coil maps with arbitrary phase per pixel, because the phases are unconstrained, leading to low spatial correlation.
#     Even though RAM is not trained on multicoil MRI, it performs better when the phase maps are also smooth. We use
#     `physics.phase_correct_maps` to constrain the phases to a smooth map, improving performance.
#
# .. tip::
#     The `sigma` of the physics noise model controls the denoising strength. Here, we show a few options.
#

model = dinv.models.RAM(device=device, pretrained=True)

with torch.no_grad():
    coil_maps = physics.phase_correct_maps(x_zf)
    physics.update(coil_maps=coil_maps)

    # Although RAM is scale-equivariant, we bring y into friendlier scale (currently it's very small)
    x_ram = model(y / x_zf.max(), physics) * x_zf.max()

    physics.update(sigma=0.04)
    x_ram_high = model(y / x_zf.max(), physics) * x_zf.max()

    physics.update(sigma=0.005)
    x_ram_low = model(y / x_zf.max(), physics) * x_zf.max()

dinv.utils.plot(
    [x_ram_low, x_ram, x_ram_high],
    titles=["RAM sigma=0.005", "RAM sigma=0.02", "RAM sigma=0.04"],
)

# %%
# Compressed sensing reconstruction
# ---------------------------------
# We compare to compressed sensing reconstruction using the FISTA algorithm with a wavelet prior and L2 data fidelity.
#
# .. tip::
#     The `lambda_reg` of the regularisation controls the regularisation strength. Here, we choose 2e-5.
#     We choose to use max 50 iterations for the demo. In practice, increase this to run to convergence.


prior = dinv.optim.WaveletPrior(
    level=3,
    wv=["db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8"],
    p=1,
    device="cpu",
    clamp_min=0,
)

model = dinv.optim.FISTA(
    prior=prior,
    data_fidelity=dinv.optim.L2(),
    stepsize=0.1,
    lambda_reg=2e-5,
    early_stop=True,
    max_iter=50,
    verbose=True,
    custom_init=lambda y, physics: (physics.A_dagger(y), physics.A_dagger(y)),
    show_progress_bar=True,
)

x_fista = model(y, physics)

# %%
# Plot the final comparison between methods:

dinv.utils.plot(
    [x_zf, x_sense, x_ram, x_fista],
    titles=["Zero-filled", "SENSE", "RAM", "FISTA+wavelets"],
)
