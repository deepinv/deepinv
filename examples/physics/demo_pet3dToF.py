r"""
Time-of-flight PET in 3D
========================

As in the :ref:`2D ToF PET example <sphx_glr_auto_examples_physics_demo_pet2dToF.py>`,
we simulate one acquisition and compare OSEM reconstructions with and without
time-of-flight (ToF) information.

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
# A compact 3D scanner
# --------------------
# Ten rings cover the 48 mm axial field of view. The 320 detectors per ring
# resolve the small inserts, while five ToF bins localize events along each ray.
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
img_size = (16, 96, 96)
mid_slice = img_size[0] // 2
scanner = parallelproj.pet_scanners.DemoPETScannerGeometry(
    torch_compat,
    dev=device,
    num_rings=10,
    num_sides=20,
    num_lor_endpoints_per_side=16,
)
tof_info = parallelproj.tof.TOFParameters(
    num_tofbins=5, tofbin_width=80.0, sigma_tof=20.0, num_sigmas=3.0
)

physics_tof = PET(
    img_size=img_size,
    voxel_size=(3, 3, 3),
    scanner=scanner,
    device=device,
    gain=0.01,
    normalize=False,
    normalize_counts=True,
    tof_info=tof_info,
)
physics_non_tof = PET(
    img_size=img_size,
    voxel_size=(3, 3, 3),
    scanner=scanner,
    device=device,
    gain=0.01,
    normalize=False,
    normalize_counts=True,
)

x, attenuation, labels = generate_pet_phantom(
    img_size, device=device, return_labels=True
)
physics_tof.update(attenuation=attenuation)
physics_non_tof.update(attenuation=physics_tof.attenuation.squeeze(-1))
dinv.utils.plot(
    [x[:, :, mid_slice], attenuation[:, :, mid_slice]],
    ["Emission phantom", "Attenuation map"],
)

# %%
# Five ToF sinograms from the same acquisition
# --------------------------------------------
# Display one axial plane of each bin; the full volume is used below.
with torch.no_grad():
    y_tof = physics_tof(x)
    y_non_tof = y_tof.sum(dim=-1)
mid_plane = y_tof.shape[-2] // 2

print(f"ToF sinogram shape: {tuple(y_tof.shape)}")
dinv.utils.plot(
    [
        y_non_tof[..., mid_plane],
        *(y_tof[..., mid_plane, i] for i in range(tof_info.num_tofbins)),
    ],
    ["All events", *(f"ToF bin {i + 1}" for i in range(tof_info.num_tofbins))],
    figsize=(16, 3),
)

# %%
# Compare OSEM reconstructions
# ----------------------------
# Both runs use the same events, initialization, subsets and number of epochs.
num_subsets = 8
num_epochs = 8
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
    [x[:, :, mid_slice], x_non_tof[:, :, mid_slice], x_tof[:, :, mid_slice]],
    ["Ground truth", "OSEM without ToF", "OSEM with ToF"],
    subtitles=[
        "Reference",
        f"NRMSE: {nrmse_non_tof:.2f}%",
        f"NRMSE: {nrmse_tof:.2f}%",
    ],
    rescale_mode="clip",
    vmin=0,
    vmax=x.max().item(),
    figsize=(10, 4),
)

# %%
