r"""
Time-of-flight PET in 2D
========================

Time-of-flight (ToF) PET records where along each line of response an event is
likely to have occurred. We simulate one acquisition with three ToF bins, then
reconstruct it twice with OSEM: once retaining the bin information and once
after summing the bins. Both reconstructions use the same detected events.

.. note::

    This example requires parallelproj (available in the full pixi environment
    or from conda-forge).
"""

# %%
import deepinv as dinv
import parallelproj
import torch
from array_api_compat import torch as torch_compat

from deepinv.physics import PET
from deepinv.utils.phantoms import generate_pet_phantom

# %%
# Scanner and phantom
# -------------------
# Three 140 mm bins cover the diameter of this 384 mm field of view. The ToF
# timing spread is represented by a 30 mm standard deviation along the LOR.
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
img_size = (128, 128)
scanner = parallelproj.pet_scanners.DemoPETScannerGeometry(
    torch_compat,
    dev=device,
    num_rings=1,
    num_sides=32,
    num_lor_endpoints_per_side=16,
)
tof_info = parallelproj.tof.TOFParameters(
    num_tofbins=3, tofbin_width=140.0, sigma_tof=30.0, num_sigmas=3.0
)

# Keep the operator and counts unnormalized so summing ToF bins gives
# measurements in the same units as the non-ToF projector.
physics_tof = PET(
    img_size=img_size,
    voxel_size=(3, 3),
    scanner=scanner,
    device=device,
    gain=0.05,
    normalize=False,
    normalize_counts=True,
    tof_info=tof_info,
)
physics_non_tof = PET(
    img_size=img_size,
    voxel_size=(3, 3),
    scanner=scanner,
    device=device,
    gain=0.05,
    normalize=False,
    normalize_counts=True,
)

x, attenuation, labels = generate_pet_phantom(
    img_size, device=device, return_labels=True
)
physics_tof.update(attenuation=attenuation)
# The same attenuation factor applies to every ToF bin of a line of response.
physics_non_tof.update(attenuation=physics_tof.attenuation.squeeze(-1))
dinv.utils.plot([x, attenuation], ["Emission phantom", "Attenuation map"])

# %%
# One acquisition, viewed in three ToF bins
# -----------------------------------------
# The last axis contains the ToF bins. Their sum discards localization
# information but preserves all measured counts for the non-ToF reconstruction.
with torch.no_grad():
    y_tof = physics_tof(x)
    y_non_tof = y_tof.sum(dim=-1)

print(f"ToF sinogram shape: {tuple(y_tof.shape)}")
dinv.utils.plot(
    [y_non_tof, *(y_tof[..., i] for i in range(tof_info.num_tofbins))],
    ["All events", "Early bin", "Central bin", "Late bin"],
    figsize=(12, 3),
)

# %%
# OSEM with and without ToF
# -------------------------
# Both runs start from the same image, use eight angular subsets and make the
# same number of passes over the acquisition. Only the measurement model differs.
num_subsets = 8
num_epochs = 5
osem = dinv.optim.OSEM(num_subsets=num_subsets, max_iter=num_epochs)

with torch.no_grad():
    x_non_tof = osem(y_non_tof, physics_non_tof, init=torch.ones_like(x))
    x_tof = osem(y_tof, physics_tof, init=torch.ones_like(x))

nrmse = dinv.metric.NRMSE()
nrmse_non_tof = 100 * nrmse(x_non_tof, x).item()
nrmse_tof = 100 * nrmse(x_tof, x).item()
print(f"Non-ToF OSEM: NRMSE={nrmse_non_tof:.2f}%")
print(f"ToF OSEM:     NRMSE={nrmse_tof:.2f}%")

# Plastic (label 1) is the reference activity. The outside background
# (label 0) has zero activity and is not used for contrast recovery.
plastic = labels == 1


def contrast_recovery(reconstruction, region):
    measured = reconstruction[region].mean() / reconstruction[plastic].mean()
    reference = x[region].mean() / x[plastic].mean()
    return ((measured - 1) / (reference - 1)).item()


print("Region contrast recovery relative to plastic (1.0 is ideal):")
print(f"{'Insert':<22} {'Non-ToF':>8} {'ToF':>8}")
for name, value in [("Lung", 2), ("Hot spheres", 3), ("Cold spheres", 4)]:
    region = labels == value
    print(
        f"{name:<22} "
        f"{contrast_recovery(x_non_tof, region):8.2f} "
        f"{contrast_recovery(x_tof, region):8.2f}"
    )

dinv.utils.plot(
    [x, x_non_tof, x_tof],
    ["Ground truth", "OSEM without ToF", "OSEM with ToF"],
    subtitles=["Reference", f"NRMSE: {nrmse_non_tof:.2f}%", f"NRMSE: {nrmse_tof:.2f}%"],
    rescale_mode="clip",
    vmin=0,
    vmax=x.max().item(),
    figsize=(9, 3),
)

# %%
# The same comparison in a small 3D volume is shown in
# :ref:`the 3D ToF PET example <sphx_glr_auto_examples_physics_demo_pet3dToF.py>`.
