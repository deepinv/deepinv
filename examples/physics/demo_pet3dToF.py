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
import matplotlib.pyplot as plt

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
    gain=1.0,
    normalize=False,
    normalize_counts=True,
    tof_info=tof_info,
)
physics_non_tof = PET(
    img_size=img_size,
    voxel_size=(3, 3, 3),
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
physics_non_tof.update(attenuation=physics_tof.attenuation.squeeze(-1))
dinv.utils.plot(
    [x[:, :, mid_slice], attenuation[:, :, mid_slice], labels[:, :, mid_slice]],
    ["Emission phantom", "Attenuation map", "Insert labels"],
)

# %%
# Five ToF sinograms from the same acquisition
# --------------------------------------------
# Display one axial plane of each bin; the full volume is used below.
# We set the Poisson gain for approximately 100 million prompt counts, including a
# uniform background with 20% as many expected events as the true signal.
with torch.no_grad():
    expected_signal = physics_tof.A(x)
    target_prompt_counts = 1e8
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
mid_plane = y_tof.shape[-2] // 2

realized_prompt_counts = round((y_tof / gain).sum().item())
print(
    f"Expected prompt counts: {target_prompt_counts:,.0f}; realized: {realized_prompt_counts:,}"
)
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
# Both runs use the same events, initialization and subsets. We follow their
# NRMSE over eight epochs, then display the lowest-error iterate of each. This
# stopping rule uses the known phantom and is only available in simulation.
num_subsets = 8
num_epochs = 8
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
    [x[:, :, mid_slice], x_non_tof[:, :, mid_slice], x_tof[:, :, mid_slice]],
    [
        "Ground truth",
        f"OSEM without ToF ({epoch_non_tof} epochs)",
        f"OSEM with ToF ({epoch_tof} epochs)",
    ],
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
# NRMSE along the OSEM epochs
# ---------------------------
# Each epoch processes every subset once. The dashed lines mark the displayed
# early-stopped reconstructions.
fig, axis = plt.subplots(figsize=(7, 4))
epochs = range(1, num_epochs + 1)
axis.plot(epochs, metrics_non_tof["nrmse"][0], label="Without ToF")
axis.plot(epochs, metrics_tof["nrmse"][0], label="With ToF")
axis.axvline(epoch_non_tof, color="tab:blue", linestyle="--", linewidth=1)
axis.axvline(epoch_tof, color="tab:orange", linestyle="--", linewidth=1)
axis.set_xlabel("OSEM epoch")
axis.set_ylabel("NRMSE (%)")
axis.set_xticks(list(epochs))
axis.legend()
fig.tight_layout()

# %%
