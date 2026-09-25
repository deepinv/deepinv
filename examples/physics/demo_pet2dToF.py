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
import matplotlib.pyplot as plt

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
    gain=1.0,
    normalize=False,
    normalize_counts=True,
    tof_info=tof_info,
)
physics_non_tof = PET(
    img_size=img_size,
    voxel_size=(3, 3),
    scanner=scanner,
    device=device,
    gain=1.0,
    normalize=False,
    normalize_counts=True,
)

x, attenuation, labels = generate_pet_phantom(
    img_size, device=device, return_labels=True
)
physics_tof.update(attenuation=attenuation)
# The same attenuation factor applies to every ToF bin of a line of response.
physics_non_tof.update(attenuation=physics_tof.attenuation.squeeze(-1))
dinv.utils.plot(
    [x, attenuation, labels],
    ["Emission phantom", "Attenuation map", "Insert labels"],
)

# %%
# One acquisition, viewed in three ToF bins
# -----------------------------------------
# The last axis contains the ToF bins. Their sum discards localization
# information but preserves all measured counts for the non-ToF reconstruction.
# We set the Poisson gain for approximately five million prompt counts, including a
# uniform background with 20% as many expected events as the true signal.
with torch.no_grad():
    expected_signal = physics_tof.A(x)
    target_prompt_counts = 5e6
    background_to_signal_ratio = 0.2
    background_tof = torch.full_like(
        expected_signal, background_to_signal_ratio * expected_signal.mean()
    )
    gain = (expected_signal.sum() + background_tof.sum()).item() / target_prompt_counts
    physics_tof.noise_model.update_parameters(gain=gain)
    physics_non_tof.noise_model.update_parameters(gain=gain)
    physics_tof.update(background=background_tof)
    physics_non_tof.update(background=background_tof.sum(dim=-1))
    del expected_signal

    y_tof = physics_tof(x)
    y_non_tof = y_tof.sum(dim=-1)

realized_prompt_counts = round((y_tof / gain).sum().item())
print(
    f"Expected prompt counts: {target_prompt_counts:,.0f}; realized: {realized_prompt_counts:,}"
)
print(f"ToF sinogram shape: {tuple(y_tof.shape)}")
dinv.utils.plot(
    [y_non_tof, *(y_tof[..., i] for i in range(tof_info.num_tofbins))],
    ["All events", "Early bin", "Central bin", "Late bin"],
    figsize=(12, 3),
)

# %%
# OSEM with and without ToF
# -------------------------
# Both runs start from the same image and use eight angular subsets. We follow
# their NRMSE over five epochs, then display the lowest-error iterate of each.
# This stopping rule uses the known phantom and is only available in simulation.
num_subsets = 8
num_epochs = 5
nrmse = dinv.metric.NRMSE()


def reconstruct_with_early_stopping(y, physics):
    best = {"nrmse": float("inf"), "epoch": 0, "image": None}

    def reconstruction_nrmse(metric_history, x_prev, x_cur):
        error = 100 * nrmse(x_cur.unsqueeze(0), x).item()
        if error < best["nrmse"]:
            best.update(
                nrmse=error,
                epoch=len(metric_history[0]) + 1,
                image=x_cur.unsqueeze(0).detach().clone(),
            )
        return error

    osem = dinv.optim.OSEM(
        num_subsets=num_subsets,
        max_iter=num_epochs,
        custom_metrics={"nrmse": reconstruction_nrmse},
    )
    _, metrics = osem(y, physics, init=torch.ones_like(x), compute_metrics=True)
    return best["image"], metrics, best["epoch"]


with torch.no_grad():
    x_non_tof, metrics_non_tof, epoch_non_tof = reconstruct_with_early_stopping(
        y_non_tof, physics_non_tof
    )
    x_tof, metrics_tof, epoch_tof = reconstruct_with_early_stopping(y_tof, physics_tof)

nrmse_non_tof = 100 * nrmse(x_non_tof, x).item()
nrmse_tof = 100 * nrmse(x_tof, x).item()
print(f"Non-ToF OSEM (epoch {epoch_non_tof}): NRMSE={nrmse_non_tof:.2f}%")
print(f"ToF OSEM (epoch {epoch_tof}):     NRMSE={nrmse_tof:.2f}%")

recovery_coefficient = dinv.metric.RecoveryCoefficient()
hot_spheres = labels == 3
rc_non_tof = recovery_coefficient(x_non_tof, x, mask=hot_spheres).item()
rc_tof = recovery_coefficient(x_tof, x, mask=hot_spheres).item()
print(f"Hot-sphere recovery coefficient (non-ToF): {rc_non_tof:.2f}")
print(f"Hot-sphere recovery coefficient (ToF):     {rc_tof:.2f}")

dinv.utils.plot(
    [x, x_non_tof, x_tof],
    [
        "Ground truth",
        f"OSEM without ToF ({epoch_non_tof} epochs)",
        f"OSEM with ToF ({epoch_tof} epochs)",
    ],
    subtitles=["Reference", f"NRMSE: {nrmse_non_tof:.2f}%", f"NRMSE: {nrmse_tof:.2f}%"],
    rescale_mode="clip",
    vmin=0,
    vmax=x.max().item(),
    figsize=(9, 3),
)

# %%
# NRMSE along the OSEM epochs
# ---------------------------
# Each epoch processes every subset once. The dashed lines mark the displayed
# early-stopped reconstructions.
fig, axis = plt.subplots(figsize=(7, 4))
epochs = range(1, num_epochs + 1)
axis.plot(epochs, metrics_non_tof["nrmse"][0], "o-", label="Without ToF")
axis.plot(epochs, metrics_tof["nrmse"][0], "o-", label="With ToF")
axis.axvline(epoch_non_tof, color="tab:blue", linestyle="--", linewidth=1)
axis.axvline(epoch_tof, color="tab:orange", linestyle="--", linewidth=1)
axis.set_xlabel("OSEM epoch")
axis.set_ylabel("NRMSE (%)")
axis.set_xticks(list(epochs))
axis.legend()
fig.tight_layout()

# %%
# The same comparison in a small 3D volume is shown in
# :ref:`the 3D ToF PET example <sphx_glr_auto_examples_physics_demo_pet3dToF.py>`.
