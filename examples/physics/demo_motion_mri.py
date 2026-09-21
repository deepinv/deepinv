r"""
MRI with motion corruption and correction
=========================================

In Cartesian MRI, kspace lines are sampled sequentially. Therefore, if there is motion during acquisition, the resulting reconstructions will be motion-corrupted.

We can model this motion corruption by modelling sequential acquisition with :class:`deepinv.physics.SequentialMultiCoilMRI`, by using a `transform` that
applies a transformation per time-step. If we ignore the motion during reconstruction, this gives motion-corrupted/blind reconstructions. However, if we
are able to model the motion transformation parameters during reconstruction, we can **compensate**/**correct** for the motion.

We show three motion transforms: a rigid rotation (:class:`deepinv.transform.Rotate`) using a linear trajectory,
a rigid rotation + translation (:class:`deepinv.transform.projective.Euclidean`) using a random Brownian trajectory,
and a non-rigid deformation (:class:`deepinv.transform.CPABDiffeomorphism`) using a periodic trajectory.

.. note::

    This this example we simulate acquisition by simulating coil maps, mask and motion trajectory.
    In practice these are estimated from the data, i.e. the mask is known from the
    acquisition, the coil maps from :func:`deepinv.physics.MultiCoilMRI.estimate_coil_maps`, and the
    motion trajectory from motion estimation (not covered here).
"""

import matplotlib.pyplot as plt
import torch
import deepinv as dinv

device = "cpu"  # dinv.utils.get_device()
torch.manual_seed(0)
rng = torch.Generator(device=device).manual_seed(1)

import matplotlib as mpl

mpl.rcParams["animation.html"] = "jshtml"

# %%
# Load a brain image
# ------------------
#
# We use a sample ground truth image from FastMRI and simulate a fully-sampled mask and simulate coil maps.
# The mask is sequential: e.g. time-step contains non-overlapping kspace samples.

dataset = dinv.datasets.SimpleFastMRISliceDataset(
    "data", anatomy="brain", download=True, use_dict_output=True
)
x = dataset[0]["x"].unsqueeze(0).to(device)  # (1, 2, H, W)

n_frames = 32

mask = dinv.physics.generator.SequentialMaskGenerator(
    (2, n_frames, *x.shape[-2:]), device=device
).step()["mask"]

blind = dinv.physics.SequentialMultiCoilMRI(
    img_size=x.shape[1:], mask=mask, coil_maps=4, device=device
)

coil_maps = blind.coil_maps  # simulated

# %%
# Simulate linear rotation motion-corruption
# ------------------------------------------
#
# We rotate the object by ``theta`` degrees at each time-step, simulating a linear trajectory.

theta = torch.linspace(0, 15, n_frames, device=device).unsqueeze(0)

plt.figure()
plt.plot(theta.squeeze().cpu())
plt.ylabel("theta (degrees)")
plt.xlabel("time-step")
plt.show()

# %%
# Construct motion physics by assuming knowledge of the transform params:

physics = dinv.physics.SequentialMultiCoilMRI(
    mask=mask,
    coil_maps=coil_maps,
    transform=dinv.transform.Rotate(
        interpolation_mode="bilinear", index_params_into_batch=True
    ),
    transform_params={"theta": theta},
)

# %%
# Simulate measurements, then show blind reconstruction vs. motion-corrected reconstruction.
# The motion-blind adjoint smears the anatomy whereas the motion-compensated adjoint rotates each frame back before summing over time.

y = physics(x)
x_blind = blind.A_adjoint(y)
x_aware = physics.A_adjoint(y)

metric = dinv.metric.PSNR(complex_abs=True)

dinv.utils.plot(
    {
        "Ground truth": x,
        f"Motion ignored": x_blind,
        f"Motion corrected": x_aware,
    },
    subtitles=[
        "",
        f"{metric(x_blind, x).item():.1f} dB",
        f"{metric(x_aware, x, x).item():.1f} dB",
    ],
    figsize=(9, 3),
)

# %%
# Simulate Euclidean motion with a Brownian trajectory
# ----------------------------------------------------
#
# We now combine the rotation with random translations giving Euclidean transformations. We model a random Brownian trajectory for the parameters (rotation angle and shifts)

params = {
    "theta_z": dinv.physics.generator.BrownianGenerator(
        n_frames=n_frames, sigma=6.0, bound=15.0, device=device, rng=rng
    ).step()["pos"],
    "shift_x": dinv.physics.generator.BrownianGenerator(
        n_frames=n_frames, sigma=4.0, bound=10.0, device=device, rng=rng
    ).step()["pos"],
    "shift_y": dinv.physics.generator.BrownianGenerator(
        n_frames=n_frames, sigma=4.0, bound=10.0, device=device, rng=rng
    ).step()["pos"],
}

plt.figure()
for label, values in params.items():
    plt.plot(values[0].cpu(), label=label)
plt.legend()
plt.xlabel("time-step")
plt.show()

# %%
# Construct motion physics like before:

physics = dinv.physics.SequentialMultiCoilMRI(
    mask=mask,
    coil_maps=coil_maps,
    transform=dinv.transform.projective.Euclidean(
        index_params_into_batch=True, padding="zeros"
    ),
    transform_params=params,
)

# %%
# The moving object (left) and the cumulative k-space coverage (right) over the acquisition:

anim = dinv.utils.plot_videos(
    [
        physics.apply_motion(physics.repeat(x, physics.mask)),
        mask.cumsum(dim=2).clamp(max=1),
    ],
    titles=["Moving object", "k-space coverage"],
    return_anim=True,
)
anim

# %%
# Simulate measurements, then show blind reconstruction vs. motion-corrected reconstruction:

y = physics(x)
x_blind = blind.A_adjoint(y)
x_aware = physics.A_adjoint(y)

dinv.utils.plot(
    {
        "Ground truth": x,
        f"Motion ignored": x_blind,
        f"Motion corrected": x_aware,
    },
    subtitles=[
        "",
        f"{metric(x_blind, x).item():.1f} dB",
        f"{metric(x_aware, x, x).item():.1f} dB",
    ],
    figsize=(9, 3),
)

# %%
# Non-rigid motion on a cardiac scan
# ----------------------------------
#
# The heartbeat deforms the anatomy non-rigidly and periodically. We simulate this with a
# diffeomorphism whose parameters follow a fixed random direction scaled by a sinusoidal trajectory.
# The ground truth is a T2-weighted, fully-sampled cardiac scan (iFFT + virtual coil combination).

x = dinv.utils.demo.load_example("demo_cmrxrecon2025_T2w_vcc.pt").to(device)

mask = dinv.physics.generator.SequentialMaskGenerator(
    (2, n_frames, *x.shape[-2:]), device=device
).step()["mask"]

blind = dinv.physics.SequentialMultiCoilMRI(
    img_size=x.shape[1:], mask=mask, coil_maps=4, device=device
)
coil_maps = blind.coil_maps

transform = dinv.transform.CPABDiffeomorphism(index_params_into_batch=True, n_trans=1)

# %%
# Simulate periodic motion of the steerable diffeomorphism params.
# Initialise with random diffeomorphism.

direction = torch.randn(1, 1, transform.cpab.sample_transformation(1).shape[-1])
phase = torch.linspace(0, 2 * torch.pi * 2, n_frames, device=device)  # two heartbeats
param = direction * (0.4 * phase.sin()).reshape(1, -1, 1)  # (1, T, d)

plt.figure()
for i in range(param.shape[-1]):
    plt.plot(param[0, :, i].cpu(), label=f"Diffeo param {i}")
plt.legend()
plt.xlabel("time-step")
plt.show()

# %%
# Construct motion physics like before:

physics = dinv.physics.SequentialMultiCoilMRI(
    mask=mask,
    coil_maps=coil_maps,
    transform=transform,
    transform_params={"diffeo": param},
)

# %%
# The moving object (left) and the cumulative k-space coverage (right) over the acquisition:

anim = dinv.utils.plot_videos(
    [
        physics.apply_motion(physics.repeat(x, physics.mask)),
        mask.cumsum(dim=2).clamp(max=1),
    ],
    titles=["Moving object", "k-space coverage"],
    return_anim=True,
    rescale_mode="clip",
    vmax=x.max() / 2,
    figsize=(7, 7),
)
anim


# %%
# Simulate measurements, then show blind reconstruction vs. motion-corrected reconstruction:

y = physics(x)
x_blind = blind.A_adjoint(y)
x_aware = physics.A_adjoint(y)

dinv.utils.plot(
    {
        "Ground truth": x,
        f"Motion ignored": x_blind,
        f"Motion corrected": x_aware,
    },
    subtitles=[
        "",
        f"{metric(x_blind, x).item():.1f} dB",
        f"{metric(x_aware, x, x).item():.1f} dB",
    ],
    figsize=(11, 7),
    rescale_mode="clip",
    vmax=x.max() / 2,
)
