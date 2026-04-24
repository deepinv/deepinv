r"""
Distributed Image Tiling Strategies
======================================

This example shows two tiling strategies for distributed processing of large images.
The distributed framework enables processing large images by automatically dividing them into tiles,
processing tiles independently or in parallel, and reconstructing the final result.

This example covers:

- :func:`deepinv.distributed.strategies.BasicStrategy`: Simple non-overlapping tiling
- :func:`deepinv.distributed.strategies.OverlapTilingStrategy`: Overlapping tiles with padding for artifact-free reconstruction

"""

# %%

import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
import numpy as np
import torch

from deepinv.distributed.strategies import BasicStrategy, OverlapTilingStrategy
from deepinv.utils.demo import load_example
from deepinv.utils.plotting import plot

# %%
# Configuration
# ---------------
#
# Define parameters for image size, patch size, and overlap settings.

device = torch.device("cpu")
img_size = 512
patch_size = 256

# %%
# Load image
# ---------------
#
# Load an example image and display the original.

clean_image = load_example(
    "butterfly.png",
    grayscale=False,
    device=device,
    img_size=img_size,
    resize_mode="resize",
)

plot(clean_image, titles="Original Image")

# %%
# Instantiate strategies
# ----------------------------
#
# Create both patching strategies to compare their behavior.
# BasicStrategy creates non-overlapping patches.

basic_strategy = BasicStrategy(
    img_size=clean_image.shape,
    tiling_dims=(-2, -1),
    num_splits=(img_size // patch_size, img_size // patch_size),
)

print(f"BasicStrategy: {basic_strategy.get_num_patches()} patches")


# %%
# ----------------------------
#
# OverlapTilingStrategy creates overlapping patches with padding to provide context at boundaries, reducing artifacts.

stride = None  # Defaults to patch_size when None
overlap = 64  # Receptive field padding for OverlapTilingStrategy
pad_mode = "reflect"  # Padding mode for global padding (e.g., 'reflect', 'constant')

overlap_strategy = OverlapTilingStrategy(
    img_size=clean_image.shape,
    tiling_dims=(-2, -1),
    patch_size=patch_size,
    overlap=overlap,
    stride=stride,
    pad_mode=pad_mode,
)

print(f"OverlapTilingStrategy: {overlap_strategy.get_num_patches()} patches")

# %%
# BasicStrategy
# --------------------------------

# %%
# BasicStrategy divides the image into a regular, non-overlapping grid of patches. This is simple and memory-efficient,
# but has a key disadvantage: patches at boundaries have no context from neighboring pixels. When a model
# processes a patch, it can not "see" information beyond the patch edges, which can lead to
# artifacts at patch boundaries (e.g., visible seams when reassembling patches) and reduced quality at edges.

# %%
# .. raw:: html
#
#    <details>
#    <summary style="cursor: pointer; color: #0066cc; font-weight: bold;">Click to expand: plot_patches() helper function</summary>
#
# We first define a helper function to plot patches. This function overlays a patch grid on an image
# and optionally visualizes the padding.


def plot_patches(ax, image, strategy, color, show_padding=False):
    """
    Overlay patch grid on image with optional padding visualization.

    :param ax: Axes object to draw on
    :param image: Input image tensor (C, H, W)
    :param strategy: Patching strategy (BasicStrategy or OverlapTilingStrategy)
    :param str color: Color for patch outlines
    :param bool show_padding: If True, display global padding region (OverlapTilingStrategy only)
    """
    # White halo effect for better visibility on patterned backgrounds
    halo = [pe.Stroke(linewidth=4, foreground="white"), pe.Normal()]
    img_np = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    if show_padding:
        # Extract global padding from metadata (left, right, top, bottom)
        pads = strategy._metadata.get("global_padding", (0, 0, 0, 0))
        # Pad the image to match the full padded extent for visualization
        img_display = np.pad(
            img_np,
            ((pads[2], pads[3]), (pads[0], pads[1]), (0, 0)),
            mode=getattr(strategy, "pad_mode", "reflect"),
        )
        h, w = img_np.shape[:2]
        # Draw rectangles for: original image boundary and full padded image boundary
        ax.add_patch(
            Rectangle(
                (pads[0], pads[2]),
                w,
                h,
                linewidth=2.5,
                edgecolor="red",
                facecolor="none",
                linestyle="--",
                path_effects=halo,
            )
        )
        ax.add_patch(
            Rectangle(
                (0, 0),
                *img_display.shape[:2][::-1],
                linewidth=2.5,
                edgecolor="black",
                facecolor="none",
                linestyle=":",
                path_effects=[
                    pe.Stroke(linewidth=3.5, foreground="white"),
                    pe.Normal(),
                ],
            )
        )
        ax.legend(
            handles=[
                Patch(
                    facecolor="none",
                    edgecolor="red",
                    linestyle="--",
                    linewidth=2,
                    label="Original image",
                ),
                Patch(
                    facecolor="none",
                    edgecolor="black",
                    linestyle=":",
                    linewidth=2,
                    label="Image with global padding",
                ),
            ],
            loc="upper right",
            fontsize=10,
            framealpha=0.9,
        )
    else:
        img_display = img_np

    ax.imshow(np.clip(img_display, 0, 1))
    # Get patch slices: _patch_slices for BasicStrategy, _global_slices for OverlapTilingStrategy
    slices = (
        strategy._patch_slices
        if isinstance(strategy, BasicStrategy)
        else strategy._global_slices
    )
    # Draw patches with highlighting on patch 0 (first/reference patch)
    for idx, slc in enumerate(
        slices if isinstance(slices, list) else [slices[i] for i in range(len(slices))]
    ):
        h_slice, w_slice = (
            (slc[-2], slc[-1])
            if isinstance(slices, list)
            else (strategy._patch_slices[idx][-2], strategy._patch_slices[idx][-1])
        )
        is_highlight = idx == 0  # Highlight first patch for reference
        edge = (
            color
            if is_highlight
            else ("salmon" if isinstance(strategy, BasicStrategy) else "skyblue")
        )
        # Draw rectangle for each patch with thicker outline and halo for highlighted patch
        ax.add_patch(
            Rectangle(
                (w_slice.start, h_slice.start),
                w_slice.stop - w_slice.start,
                h_slice.stop - h_slice.start,
                linewidth=2.5 if is_highlight else 1.2,
                edgecolor=edge,
                facecolor="none",
                path_effects=halo if is_highlight else None,
            )
        )
        ax.text(
            w_slice.start + 6,
            h_slice.start + 16,
            f"patch {idx}",
            color="black",
            fontsize=12 if is_highlight else 11,
            weight="bold" if is_highlight else "normal",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=edge,
                linewidth=0.8,
            ),
        )


# %%
# .. raw:: html
#
#    </details>

# %%
# Display all patches from BasicStrategy with non-overlapping grid overlay.
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plot_patches(ax, clean_image, basic_strategy, color="red", show_padding=False)
ax.set_title("BasicStrategy: Non-Overlapping Patches", fontsize=14, fontweight="bold")
ax.axis("off")
plt.show()


# %%
# OverlapTilingStrategy
# --------------------------------
#


# %%
# OverlapTilingStrategy addresses the boundary artifact problem by adding overlapping patches with padding
# (the "receptive field"). Instead of non-overlapping tiles, this strategy:
#
# - **Adds global padding**: Reflects or extends the image at boundaries, creating a padded version. This gives each
#   patch context from pixels beyond the original image boundaries.
# - **Creates overlapping patches**: Each patch is extracted from the padded image and includes both the original
#   content and the padding context. Patches overlap in the padded space (by 'overlap' pixels), providing context
#   that helps models produce smoother, artifact-free results.
#
# This method is more computationally intensive (larger patches, overlaps mean redundant processing) but produces
# higher-quality results, especially at image boundaries.


# %%
# This visualization shows all patches overlaid on the original image (red dashed line) and the padded
# extent (black dotted line). Patch 0 is highlighted with a thicker border. The legend explains the
# padding boundaries.
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plot_patches(ax, clean_image, overlap_strategy, color="blue", show_padding=True)
ax.set_title(
    "OverlapTilingStrategy: Overlapping Patches with Padding",
    fontsize=12,
    fontweight="bold",
)
ax.axis("off")
plt.show()

# %%
# .. raw:: html
#
#    <details>
#    <summary style="cursor: pointer; color: #0066cc; font-weight: bold;">Click to expand: plot_single_patch() helper function</summary>
#
# For clearer visualization, we display one patch from each strategy side by side. To do this,
# we first define a helper function that extracts and plots a single patch from a given patching strategy.


def plot_single_patch(ax, image, strategy, patch_idx, color):
    """
    Extract and display a single patch from a patching strategy.

    :param ax: Axes object to draw on
    :param image: Input image tensor (C, H, W)
    :param strategy: Patching strategy (BasicStrategy or OverlapTilingStrategy)
    :param int patch_idx: Index of the patch to extract and display
    :param str color: Color for the patch border
    :returns: Tuple of (height, width) of the displayed patch
    """
    # Extract patch using strategy's get_local_patches method (returns list of (idx, patch) tuples)
    idx, patch = strategy.get_local_patches(image, [patch_idx])[0]
    # Convert patch tensor to numpy array for visualization
    patch_np = patch.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    ax.imshow(np.clip(patch_np, 0, 1))
    h, w = patch_np.shape[:2]

    # Draw colored border with white halo for visibility
    ax.add_patch(
        Rectangle(
            (0, 0),
            w,
            h,
            linewidth=3,
            edgecolor=color,
            facecolor="none",
            path_effects=[pe.Stroke(linewidth=4.5, foreground="white"), pe.Normal()],
        )
    )
    ax.text(
        8,
        20,
        f"patch {idx}",
        color="black",
        fontsize=12,
        weight="bold",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="white", edgecolor=color, linewidth=1.2
        ),
    )
    ax.axis("off")  # Hide axes
    return h, w


# %%
# .. raw:: html
#
#    </details>


# %%
# Comparison
# --------------------------------
#

# %%
# Extract patch 0 from both strategies to compare their handling of padding and overlap.
patch_idx = 0
# Retrieve window height from OverlapTilingStrategy metadata to show size difference
overlap_h = overlap_strategy._metadata["window_shape"][
    0
]  # Height of overlap patch (includes receptive field)
fig = plt.figure(figsize=(6, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[patch_size, overlap_h], wspace=0.2)

# Left subplot: BasicStrategy patch (non-overlapping, base size)
ax0 = fig.add_subplot(gs[0])
plot_single_patch(ax0, clean_image, basic_strategy, patch_idx, "red")
ax0.set_title(
    f"BasicStrategy\nPatch {patch_idx} ({patch_size}×{patch_size})",
    fontsize=12,
    fontweight="bold",
    pad=10,
)

# Right subplot: OverlapTilingStrategy patch (includes receptive field/overlap padding)
ax1 = fig.add_subplot(gs[1])
plot_single_patch(ax1, clean_image, overlap_strategy, patch_idx, "blue")
ax1.set_title(
    f"OverlapTilingStrategy\nPatch {patch_idx} ({overlap_h}×{overlap_h})",
    fontsize=12,
    fontweight="bold",
    pad=10,
)
# plt.tight_layout()
plt.show()

# %%
# Summary
# -------
#
# Display comparison summary showing patch counts, stride settings, and overlap parameters.

print("\n" + "=" * 60)
print(f"SUMMARY\nImage: {img_size}×{img_size}")
print(
    f"\nBasicStrategy: {basic_strategy.get_num_patches()} patches, stride={patch_size}, Patch: {(patch_size,patch_size)}"
)
print(
    f"OverlapTilingStrategy: {overlap_strategy.get_num_patches()} patches, stride={patch_size}, overlap={overlap}, Patch: {overlap_strategy._metadata['window_shape']}"
)
print("=" * 60)
