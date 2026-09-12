r"""
Simulating rigid motion during an MRI acquisition
=================================================

This example simulates residual patient motion while Cartesian k-space lines
are acquired sequentially. The idea is to simulate residual patient (rigid) motion at each
acquisition time step.

We use a raw multi-coil brain scan from fastMRI, estimate its coil sensitivity
maps, animate the acquisition, and compare adjoint reconstructions that ignore
or account for the known motion trajectory.
"""

# %%
# Load a multi-coil fastMRI brain scan
# ------------------------------------
#
# DeepInv provides a small raw fastMRI volume for examples. We select its
# middle slice and estimate ESPIRiT coil sensitivity maps from 15
# auto-calibration lines with :class:`deepinv.datasets.MRISliceTransform`.
#
# .. important::
#
#    By using this dataset, you confirm that you have agreed to and signed the
#    `fastMRI data use agreement <https://fastmri.med.nyu.edu/>`_.

from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import torch
from torch import Tensor
from torch.utils.data import DataLoader

import deepinv as dinv
from deepinv.physics import MultiCoilMRI

# Render ``FuncAnimation`` objects as self-contained HTML in Jupyter and in the
# Sphinx-Gallery build. Native Python execution still uses Matplotlib's active
# graphical backend.
mpl.rcParams["animation.html"] = "jshtml"
# The default 20 MB limit truncates this 213-frame animation before the final
# k-space line. Keep every frame while retaining the interactive JSHTML player.
mpl.rcParams["animation.embed_limit"] = 100.0

# %%
# Load multi-coil MRI data
# ------------------------
# First, let's load some data. We will use a sample from the fastMRI dataset, and will define the
# associated "static" physics, i.e. the physics acquisition model that does not take into
# account patient motion.
# As our data is multi-coil, we use :class:`deepinv.physics.MultiCoilMRI`.

device = "cpu"

dinv.datasets.download_archive(
    dinv.utils.get_image_url("demo_fastmri_brain_multicoil.h5"),
    dinv.utils.get_cache_home() / "brain" / "fastmri.h5",
)

dataset = dinv.datasets.FastMRISliceDataset(
    dinv.utils.get_cache_home() / "brain",
    slice_index="middle",
    transform=dinv.datasets.MRISliceTransform(
        estimate_coil_maps=True,
        acs=15,  # Num. low frequency, fix to 15
    ),
    use_dict_output=True,
)

batch = next(iter(DataLoader(dataset)))
target, y, params = (
    batch["x"].to(device),
    batch["y"].to(device),
    batch["params"],
)
reconstruction_size = target.shape[-2:]

static_physics = MultiCoilMRI(
    img_size=y.shape[-2:],
    mask=torch.ones(y.shape[-2:]),
    coil_maps=torch.ones(y.shape[-3:], dtype=torch.complex64, device=device),
    device=device,
    three_d=False,
)

x = static_physics.A_adjoint(y)  # (B, 2, W, H)
static_physics.update(**params)
coil_maps = static_physics.coil_maps

# %%
# We can now plot the data at hand.

y_rss = dinv.utils.MRIMixin().rss(y)  # (B, 1, W, H)
x_rss = dinv.utils.MRIMixin().rss(x, multicoil=False)  # (B, 1, W, H)

images = [x_rss, torch.log10(1e1 * y_rss + 1e-6)]
titles = ["RSS", "Fully-sampled k-space data"]

dinv.utils.plot(images, titles, figsize=(10, 10))

# %%
# Create a sequential Cartesian acquisition
# ------------------------------------------
#
# The next steps consists in simulating a sequential mask acquisition. The fully sampled
# Cartesian mask is split into a temporal mask containing one
# phase-encoding line per time step. Its temporal union covers the complete
# k-space.

acceleration = 1
spatial_generator = dinv.physics.generator.EquispacedMaskGenerator(
    img_size=x.shape[-3:],
    acceleration=acceleration,
    device=device,
)
mask_generator = dinv.physics.generator.SequentialMaskGenerator(spatial_generator)
sequential_mask = mask_generator.step(batch_size=1, seed=0)["mask"]

# From this sequential mask, we can deduce the full-time coverage
static_mask = sequential_mask.sum(2)

# Next, check some shapes
print("Sequential mask shape (B,C,T,H,W):", sequential_mask.shape)
print(
    "One sampled line per time step:",
    bool((sequential_mask.sum((-2, -1)) == y.shape[-2]).all()),
)
print("Full k-space coverage:", bool(sequential_mask.amax(dim=2).bool().all()))

# %%
# Generate and apply residual rigid motion
# ----------------------------------------
#
# Now that we have a sequential mask, we can go back to the source image and generate rigid motion that will be
# applied to it.
#
# The :class:`deepinv.physics.generator.RigidMotionGenerator` samples one pose per
# timestep. Its current stochastic model is a bounded reflected Brownian
# process, which accounts for microscopic motion (sub-pixel motion, random rotations) happening during the acquisition.
#
# With these parameters at hand, we can apply it to our source image with
# :class:`deepinv.physics.TimeVaryingMotion`. Under the hood, this class applies the
# time-dependent parameters frame-by-frame.

sample_duration = 0.04  # seconds per k-space line
motion_generator = dinv.physics.generator.RigidMotionGenerator(
    n_frames=sequential_mask.shape[2],
    dt=sample_duration,
    rotation_sigma=0.4,
    translation_sigma=0.75,
    rotation_max=1.0,
    translation_max=3.0,
    device=device,
)
motion_params = motion_generator.step(batch_size=x.shape[0], seed=0)

motion = dinv.physics.TimeVaryingMotion(
    dinv.transform.Rotate(interpolation_mode="bilinear")
    * dinv.transform.FourierShift(),
    device=device,
)
physics = dinv.physics.SequentialMultiCoilMRI(
    mask=sequential_mask,
    coil_maps=coil_maps,
    motion=motion,
    motion_params=motion_params,
    device=device,
)

# The plotting helper is executed but omitted from the rendered gallery page,
# keeping the example focused on the dynamic MRI API.
# sphinx_gallery_start_ignore


def animate_mri_sampling(
    mask: Tensor,
    dynamic_image: Tensor | None = None,
    theta: Tensor | None = None,
    x_shift: Tensor | None = None,
    y_shift: Tensor | None = None,
    sampling_times: float | Sequence[float] | Tensor | None = None,
    batch_index: int = 0,
    channel_index: int = 0,
    interval: int = 250,
    frame_stride: int = 1,
    save_path: str | Path | None = None,
    repeat: bool = True,
    show: bool = True,
):
    """Animate a dynamic Cartesian sampling mask in the Fourier domain.

    The first two panels show the samples acquired during the current time
    frame and all samples acquired up to that frame. If ``dynamic_image`` is
    provided, a third panel shows the corresponding motion-transformed image.
    If ``theta`` is provided, an additional panel reveals the rotation
    trajectory up to the current time frame.

    :param torch.Tensor mask: sampling mask with shape ``(B, C, T, H, W)``,
        ``(C, T, H, W)``, or ``(T, H, W)``.
    :param torch.Tensor dynamic_image: optional motion-transformed image with
        shape ``(B,C,T,H,W)``. Two-channel complex data is displayed by
        magnitude.
    :param torch.Tensor theta: optional rotation trajectory in degrees with
        shape ``(B,T)`` or ``(T,)``.
    :param torch.Tensor x_shift: optional horizontal translation trajectory in
        pixels with shape ``(B,T)`` or ``(T,)``.
    :param torch.Tensor y_shift: optional vertical translation trajectory in
        pixels with shape ``(B,T)`` or ``(T,)``.
    :param sampling_times: acquisition times in seconds. A scalar is interpreted
        as the duration of one frame, while a sequence or tensor gives the time
        of every frame. If ``None``, frame indices are displayed.
    :param int batch_index: batch element to display.
    :param int channel_index: channel (or parallel coil mask) to display.
    :param int interval: delay between displayed frames in milliseconds.
    :param int frame_stride: display every ``frame_stride`` acquisition frames.
        The final acquisition frame is always included.
    :param save_path: optional output path. Matplotlib infers the writer from
        the extension (for example, ``.gif`` or ``.mp4``).
    :param bool repeat: whether the animation repeats.
    :param bool show: display the native Matplotlib window when using an
        interactive Python backend.
    :return: figure and :class:`matplotlib.animation.FuncAnimation`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if not isinstance(mask, Tensor):
        raise TypeError(f"mask must be a torch.Tensor, got {type(mask).__name__}.")

    if mask.ndim == 5:
        if not 0 <= batch_index < mask.shape[0]:
            raise IndexError(
                f"batch_index={batch_index} is invalid for batch size {mask.shape[0]}."
            )
        if not 0 <= channel_index < mask.shape[1]:
            raise IndexError(
                f"channel_index={channel_index} is invalid for {mask.shape[1]} channels."
            )
        frames = mask[batch_index, channel_index]
    elif mask.ndim == 4:
        if not 0 <= channel_index < mask.shape[0]:
            raise IndexError(
                f"channel_index={channel_index} is invalid for {mask.shape[0]} channels."
            )
        frames = mask[channel_index]
    elif mask.ndim == 3:
        frames = mask
    else:
        raise ValueError(
            "mask must have shape (B,C,T,H,W), (C,T,H,W), or (T,H,W), "
            f"got {tuple(mask.shape)}."
        )

    if frames.ndim != 3:
        raise ValueError(
            "The selected mask must have shape (T,H,W), but got "
            f"{tuple(frames.shape)}. For a dynamic BaseMaskGenerator, pass "
            "img_size=(C,T,H,W), not (T,H,W)."
        )
    if frames.shape[0] == 0:
        raise ValueError("mask must contain at least one time frame.")
    frames = frames.detach().abs().to(device="cpu", dtype=torch.float32)
    source_n_frames = frames.shape[0]
    if not isinstance(frame_stride, int) or frame_stride < 1:
        raise ValueError("frame_stride must be a positive integer.")
    frame_indices = torch.arange(0, source_n_frames, frame_stride)
    if frame_indices[-1] != source_n_frames - 1:
        frame_indices = torch.cat(
            (frame_indices, frame_indices.new_tensor([source_n_frames - 1]))
        )
    cumulative = frames.cumsum(dim=0).clamp_max(1)[frame_indices]
    frames = frames[frame_indices]
    n_frames = frame_indices.numel()

    image_frames = None
    if dynamic_image is not None:
        if dynamic_image.ndim != 5:
            raise ValueError(
                "dynamic_image must have shape (B,C,T,H,W), but got "
                f"{tuple(dynamic_image.shape)}."
            )
        if not 0 <= batch_index < dynamic_image.shape[0]:
            raise IndexError(
                f"batch_index={batch_index} is invalid for dynamic image batch "
                f"size {dynamic_image.shape[0]}."
            )
        if dynamic_image.shape[2] != source_n_frames:
            raise ValueError(
                f"Mask has {source_n_frames} frames but dynamic_image has "
                f"{dynamic_image.shape[2]}."
            )
        image_frames = dynamic_image[batch_index].detach()
        if image_frames.shape[0] == 2:
            image_frames = image_frames.square().sum(dim=0).sqrt()
        elif image_frames.shape[0] == 1:
            image_frames = image_frames[0]
        else:
            raise ValueError(
                "dynamic_image must have one magnitude channel or two "
                "real/imaginary channels."
            )
        image_frames = image_frames.to(device="cpu", dtype=torch.float32)[frame_indices]

    # Matplotlib consumes NumPy arrays. Convert the complete sequences once
    # instead of converting one Torch tensor during every animation callback.
    frames = frames.numpy()
    cumulative = cumulative.numpy()
    if image_frames is not None:
        image_frames = image_frames.numpy()

    def select_trajectory(values: Tensor | None, name: str):
        if values is None:
            return None
        if values.ndim == 2:
            if not 0 <= batch_index < values.shape[0]:
                raise IndexError(
                    f"batch_index={batch_index} is invalid for {name} batch "
                    f"size {values.shape[0]}."
                )
            values = values[batch_index]
        elif values.ndim != 1:
            raise ValueError(
                f"{name} must have shape (B,T) or (T,), got {tuple(values.shape)}."
            )
        if values.numel() != source_n_frames:
            raise ValueError(
                f"Mask has {source_n_frames} frames but {name} has "
                f"{values.numel()}."
            )
        return (
            values.detach().to(device="cpu", dtype=torch.float32)[frame_indices].numpy()
        )

    motion_trajectories = {
        r"$\theta$ (degrees)": select_trajectory(theta, "theta"),
        r"$\Delta W$ (pixels)": select_trajectory(x_shift, "x_shift"),
        r"$\Delta H$ (pixels)": select_trajectory(y_shift, "y_shift"),
    }
    motion_trajectories = {
        name: values
        for name, values in motion_trajectories.items()
        if values is not None
    }

    if sampling_times is None:
        times = None
    elif isinstance(sampling_times, (int, float)):
        if sampling_times <= 0:
            raise ValueError("A scalar sampling_times must be strictly positive.")
        times = torch.arange(source_n_frames, dtype=torch.float64) * sampling_times
    else:
        times = torch.as_tensor(sampling_times, dtype=torch.float64).flatten()
        if times.numel() != source_n_frames:
            raise ValueError(
                f"Expected {source_n_frames} sampling times, got {times.numel()}."
            )
        if source_n_frames > 1 and torch.any(times[1:] < times[:-1]):
            raise ValueError("sampling_times must be nondecreasing.")
    if times is not None:
        times = times[frame_indices]

    n_panels = 2 + int(image_frames is not None) + int(bool(motion_trajectories))
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4))
    image_kwargs = dict(cmap="gray", vmin=0, vmax=1, origin="lower", aspect="auto")
    current_image = axes[0].imshow(frames[0], **image_kwargs)
    cumulative_image = axes[1].imshow(cumulative[0], **image_kwargs)
    axes[0].set_title("Current acquisition")
    axes[1].set_title("Cumulative coverage")
    for axis in axes[:2]:
        axis.set_xlabel(r"$k_x$")
        axis.set_ylabel(r"$k_y$")
    motion_image = None
    if image_frames is not None:
        motion_image = axes[2].imshow(
            image_frames[0],
            cmap="gray",
            vmin=image_frames.min().item(),
            vmax=image_frames.max().item(),
            origin="lower",
        )
        axes[2].set_title("Motion-transformed image")
        axes[2].set_axis_off()

    motion_lines = []
    motion_markers = []
    if motion_trajectories:
        motion_axis = axes[2 + int(image_frames is not None)]
        time_axis = torch.arange(n_frames).numpy() if times is None else times.numpy()
        colors = ("tab:blue", "tab:orange", "tab:green")
        for (label, values), color in zip(
            motion_trajectories.items(), colors, strict=True
        ):
            (line,) = motion_axis.plot([], [], color=color, linewidth=1.5, label=label)
            (marker,) = motion_axis.plot([], [], "o", color=color, markersize=4)
            motion_lines.append((line, values))
            motion_markers.append((marker, values))
        motion_axis.axhline(0, color="black", linewidth=0.7, alpha=0.4)
        motion_axis.set_xlim(time_axis[0], time_axis[-1] if n_frames > 1 else 1)
        motion_limit = max(
            1.0,
            max(float(abs(values).max()) for values in motion_trajectories.values())
            * 1.05,
        )
        motion_axis.set_ylim(-motion_limit, motion_limit)
        motion_axis.set_xlabel("Time (s)" if times is not None else "Sampling frame")
        motion_axis.set_ylabel("Rotation / translation")
        motion_axis.set_title("Brownian rigid motion")
        motion_axis.legend(loc="upper right")
    # Compute the layout once. Re-running constrained layout while animating
    # large images makes every frame substantially more expensive.
    fig.tight_layout()

    def update(frame_index: int):
        current_image.set_data(frames[frame_index])
        cumulative_image.set_data(cumulative[frame_index])
        artists = [current_image, cumulative_image]
        if motion_image is not None:
            motion_image.set_data(image_frames[frame_index])
            artists.append(motion_image)
        for (line, values), (marker, _) in zip(
            motion_lines, motion_markers, strict=True
        ):
            line.set_data(time_axis[: frame_index + 1], values[: frame_index + 1])
            marker.set_data([time_axis[frame_index]], [values[frame_index]])
            artists.extend((line, marker))
        return tuple(artists)

    update(0)
    # FigureCanvasMac advertises blitting support, but its native TimerMac can
    # crash during blitted animations (including with an invalid callbacks
    # state). Use the robust full-redraw path only for that backend.
    use_blit = fig.canvas.supports_blit and "macosx" not in plt.get_backend().lower()
    animation = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval * frame_stride,
        repeat=repeat,
        blit=use_blit,
        cache_frame_data=False,
    )
    if save_path is not None:
        animation.save(Path(save_path))

    if show:
        if "inline" in plt.get_backend().lower():
            from IPython.display import display

            display(animation)
        else:
            plt.show()

    return fig, animation


# sphinx_gallery_end_ignore

# %%
# Animate the acquisition
# -----------------------
#
# We display every acquisition frame so that the cumulative mask reaches full
# k-space coverage in both native Matplotlib and the inline animation.

# TODO: use true physics instead of cooking something like that
x_dynamic = physics.repeat(x, sequential_mask)
x_motion = motion(x_dynamic, motion_params=motion_params)
x_motion = static_physics.crop(x_motion, shape=reconstruction_size)

figure, animation = animate_mri_sampling(
    sequential_mask,
    dynamic_image=x_motion,
    theta=motion_params["theta"],
    x_shift=motion_params["x_shift"],
    y_shift=motion_params["y_shift"],
    sampling_times=0.04,  # 40 ms per sampled line
    interval=30,
    frame_stride=1,  # plot every timestep - increase to reduce plot size
)


# %%
# Compare motion-blind and motion-aware adjoints
# ------------------------------------------------
#
# Measurements from the moving object are reconstructed in two ways.
# ``blind=True`` ignores the trajectory, whereas the default adjoint applies
# the inverse frame-wise motion before summing over time. For reference, we
# also simulate the same undersampling pattern without motion.
#
# These are adjoint (zero-filled) images rather than solutions of a full
# reconstruction algorithm. Nevertheless, the comparison isolates the motion
# artefact and shows the effect of incorporating the known trajectory.

reference_physics = dinv.physics.SequentialMultiCoilMRI(
    mask=sequential_mask,
    coil_maps=coil_maps,
    device=device,
)
x_reference = reference_physics.A_adjoint(reference_physics(x))

y = physics(x)
x_motion_blind = physics.A_adjoint(y, blind=True)
x_motion_aware = physics.A_adjoint(y)


def magnitude_and_crop(z):
    """Crop an adjoint reconstruction and convert it to magnitude."""
    z = static_physics.crop(z, shape=reconstruction_size)
    return torch.linalg.vector_norm(z, dim=1, keepdim=True)


x_reference_magnitude = magnitude_and_crop(x_reference)
x_motion_blind_magnitude = magnitude_and_crop(x_motion_blind)
x_motion_aware_magnitude = magnitude_and_crop(x_motion_aware)

psnr = dinv.metric.PSNR(max_pixel=None)
blind_psnr = psnr(x_motion_blind_magnitude, x_reference_magnitude).item()
aware_psnr = psnr(x_motion_aware_magnitude, x_reference_magnitude).item()

print(f"Motion-blind PSNR: {blind_psnr:.2f} dB")
print(f"Motion-aware PSNR: {aware_psnr:.2f} dB")

dinv.utils.plot(
    {
        "No motion": x_reference_magnitude,
        f"Motion ignored (PSNR: {blind_psnr:.2f} dB)": x_motion_blind_magnitude,
        f"Known motion (PSNR: {aware_psnr:.2f} dB)": x_motion_aware_magnitude,
        "Uncorrected difference": magnitude_and_crop(x_motion_blind - x_reference),
    },
    rescale_mode="min_max",
    figsize=(12, 3),
    plot_inset=True,
    extract_loc=(0.55, 0.15),
    extract_size=0.2,
    inset_loc=(0.0, 0.6),
    inset_size=0.4,
)
