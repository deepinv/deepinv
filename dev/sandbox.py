# %% [markdown]
# ## Tile strategies

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt

# %%
import deepinv as dinv
from torchvision.transforms import ToTensor, Compose, CenterCrop
import torch.nn.functional as F

save_dir = (
    "/Users/tl255879/Documents/research/repos/deepinv-PRs/hackaton_v2/data/urban100"
)

torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Define base train dataset
dataset = dinv.datasets.Urban100HR(
    save_dir, download=True, transform=Compose([ToTensor()])
)

# %%


# %%
print(dataset[0].shape)
print(dataset[0].dtype)

# %%
plt.figure()
plt.imshow(dataset[0].cpu().permute(1, 2, 0))

# %%
image = dataset[0]

# Finally, create a noisy version of the image with a fixed noise level sigma.
sigma = 0.2
noisy_image = image + sigma * torch.randn_like(image)

# %%


# %%
plt.figure()
plt.imshow(noisy_image.cpu().permute(1, 2, 0))

# %%

# dncnn = dinv.models.DnCNN(pretrained='download')

drunet = dinv.models.DRUNet()

# %%

# denoised_image = dncnn(noisy_image.unsqueeze(0))

# plt.figure()
# plt.imshow(denoised_image[0].permute(1,2,0).cpu().detach().numpy())
# plt.title("DnCNN Denoised Image")
# plt.show()


# %%

drunet_denoised = drunet(noisy_image.unsqueeze(0), sigma=sigma)

# %%

plt.figure()
plt.imshow(drunet_denoised[0].permute(1, 2, 0).cpu().detach().numpy())
plt.title("DRUNet Denoised Image")
plt.show()

# %%
noisy_image.shape

# %%
input_img = noisy_image.unsqueeze(0)

# %%
receptive_field_radius = 32
B, C, H, W = input_img.shape


# %%
def create_tiled_windows_and_masks(
    image, patch_size, receptive_field_radius, overlap_strategy="reflect"
):
    """
    Create tiled windows for processing large images with models that have receptive fields.

    Args:
        image (torch.Tensor): Input image of shape (B, C, H, W) or (C, H, W)
        patch_size (int or tuple): Size of the non-overlapping patches
        receptive_field_radius (int): Radius of the model's receptive field
        overlap_strategy (str): How to handle boundaries ('reflect', 'constant', 'edge')

    Returns:
        windows (list): List of big windows to feed to the model
        masks (list): List of masks to crop the output to match the original patches
        patch_positions (list): List of (top, left, bottom, right) positions for reassembly
    """
    # Handle both (C, H, W) and (B, C, H, W) inputs
    if image.ndim == 3:
        image = image.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    B, C, H, W = image.shape

    # Handle patch_size as int or tuple
    if isinstance(patch_size, int):
        patch_h, patch_w = patch_size, patch_size
    else:
        patch_h, patch_w = patch_size

    # Calculate number of patches
    n_patches_h = (H + patch_h - 1) // patch_h  # Ceiling division
    n_patches_w = (W + patch_w - 1) // patch_w

    # Calculate window size (patch + receptive field padding)
    window_h = patch_h + 2 * receptive_field_radius
    window_w = patch_w + 2 * receptive_field_radius

    windows = []
    masks = []
    patch_positions = []

    for i in range(n_patches_h):
        for j in range(n_patches_w):
            # Calculate patch coordinates
            patch_top = i * patch_h
            patch_left = j * patch_w
            patch_bottom = min(patch_top + patch_h, H)
            patch_right = min(patch_left + patch_w, W)

            # Calculate window coordinates (with padding for receptive field)
            window_top = patch_top - receptive_field_radius
            window_left = patch_left - receptive_field_radius
            window_bottom = patch_bottom + receptive_field_radius
            window_right = patch_right + receptive_field_radius

            # Extract window with boundary handling
            if overlap_strategy == "reflect":
                # Use torch.nn.functional.pad for reflection padding

                # Calculate padding needed
                pad_top = max(0, -window_top)
                pad_left = max(0, -window_left)
                pad_bottom = max(0, window_bottom - H)
                pad_right = max(0, window_right - W)

                # Adjust window coordinates to valid range
                window_top_clamped = max(0, window_top)
                window_left_clamped = max(0, window_left)
                window_bottom_clamped = min(H, window_bottom)
                window_right_clamped = min(W, window_right)

                # Extract the valid part of the window
                window = image[
                    :,
                    :,
                    window_top_clamped:window_bottom_clamped,
                    window_left_clamped:window_right_clamped,
                ]

                # Apply reflection padding if needed
                if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
                    window = F.pad(
                        window,
                        (pad_left, pad_right, pad_top, pad_bottom),
                        mode="reflect",
                    )

            elif overlap_strategy == "constant":
                # Zero padding

                pad_top = max(0, -window_top)
                pad_left = max(0, -window_left)
                pad_bottom = max(0, window_bottom - H)
                pad_right = max(0, window_right - W)

                window_top_clamped = max(0, window_top)
                window_left_clamped = max(0, window_left)
                window_bottom_clamped = min(H, window_bottom)
                window_right_clamped = min(W, window_right)

                window = image[
                    :,
                    :,
                    window_top_clamped:window_bottom_clamped,
                    window_left_clamped:window_right_clamped,
                ]

                if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
                    window = F.pad(
                        window,
                        (pad_left, pad_right, pad_top, pad_bottom),
                        mode="constant",
                        value=0,
                    )

            elif overlap_strategy == "edge":
                # Edge/replicate padding

                pad_top = max(0, -window_top)
                pad_left = max(0, -window_left)
                pad_bottom = max(0, window_bottom - H)
                pad_right = max(0, window_right - W)

                window_top_clamped = max(0, window_top)
                window_left_clamped = max(0, window_left)
                window_bottom_clamped = min(H, window_bottom)
                window_right_clamped = min(W, window_right)

                window = image[
                    :,
                    :,
                    window_top_clamped:window_bottom_clamped,
                    window_left_clamped:window_right_clamped,
                ]

                if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
                    window = F.pad(
                        window,
                        (pad_left, pad_right, pad_top, pad_bottom),
                        mode="replicate",
                    )

            # Create mask for cropping the output
            # The mask indicates where to crop from the model output to get the original patch
            mask_top = receptive_field_radius
            mask_left = receptive_field_radius

            # Handle edge cases where the patch might be smaller than expected
            actual_patch_h = patch_bottom - patch_top
            actual_patch_w = patch_right - patch_left

            mask_bottom = mask_top + actual_patch_h
            mask_right = mask_left + actual_patch_w

            mask = (mask_top, mask_left, mask_bottom, mask_right)

            # Store patch position for reassembly
            patch_position = (patch_top, patch_left, patch_bottom, patch_right)

            # Remove batch dimension if input was 3D
            if squeeze_output:
                window = window.squeeze(0)

            windows.append(window)
            masks.append(mask)
            patch_positions.append(patch_position)

    return windows, masks, patch_positions


# %%
def reassemble_from_patches(processed_windows, masks, patch_positions, original_shape):
    """
    Reassemble processed patches back into a full image.

    Args:
        processed_windows (list): List of processed windows from the model
        masks (list): List of masks to crop the windows
        patch_positions (list): List of patch positions for reassembly
        original_shape (tuple): Original image shape (B, C, H, W) or (C, H, W)

    Returns:
        torch.Tensor: Reassembled image
    """
    # Determine if we need batch dimension
    if len(original_shape) == 3:
        C, H, W = original_shape
        output = torch.zeros(
            (C, H, W),
            dtype=processed_windows[0].dtype,
            device=processed_windows[0].device,
        )
        has_batch = False
    else:
        B, C, H, W = original_shape
        output = torch.zeros(
            (B, C, H, W),
            dtype=processed_windows[0].dtype,
            device=processed_windows[0].device,
        )
        has_batch = True

    for window, mask, patch_pos in zip(processed_windows, masks, patch_positions):
        # Extract the relevant patch from the processed window using the mask
        mask_top, mask_left, mask_bottom, mask_right = mask

        if has_batch:
            if window.ndim == 3:  # Add batch dimension if needed
                window = window.unsqueeze(0)
            cropped_patch = window[:, :, mask_top:mask_bottom, mask_left:mask_right]
        else:
            if window.ndim == 4:  # Remove batch dimension if needed
                window = window.squeeze(0)
            cropped_patch = window[:, mask_top:mask_bottom, mask_left:mask_right]

        # Place the cropped patch in the output image
        patch_top, patch_left, patch_bottom, patch_right = patch_pos

        if has_batch:
            output[:, :, patch_top:patch_bottom, patch_left:patch_right] = cropped_patch
        else:
            output[:, patch_top:patch_bottom, patch_left:patch_right] = cropped_patch

    return output


# %%
def process_large_image_with_tiling(
    model,
    image,
    patch_size,
    receptive_field_radius,
    overlap_strategy="reflect",
    device=None,
    **model_kwargs,
):
    """
    Process a large image using tiling to handle memory constraints and receptive fields.

    Args:
        model: The neural network model to apply
        image (torch.Tensor): Input image of shape (B, C, H, W) or (C, H, W)
        patch_size (int or tuple): Size of the non-overlapping patches
        receptive_field_radius (int): Radius of the model's receptive field
        overlap_strategy (str): How to handle boundaries ('reflect', 'constant', 'edge')
        device: Device to run the model on (if None, uses image device)
        **model_kwargs: Additional keyword arguments to pass to the model

    Returns:
        torch.Tensor: Processed image with the same shape as input
    """
    if device is None:
        device = image.device

    original_shape = image.shape

    # Create tiled windows and masks
    windows, masks, patch_positions = create_tiled_windows_and_masks(
        image, patch_size, receptive_field_radius, overlap_strategy
    )

    # Process each window through the model
    processed_windows = []

    for window in windows:
        # Move window to device if needed
        if window.device != device:
            window = window.to(device)

        # Ensure window has batch dimension for model
        if window.ndim == 3:
            window = window.unsqueeze(0)

        # Apply model
        with torch.no_grad():
            processed_window = model(window, **model_kwargs)

        processed_windows.append(processed_window)

    # Reassemble the processed patches
    result = reassemble_from_patches(
        processed_windows, masks, patch_positions, original_shape
    )

    return result


# %%
# Test the tiling functions
print(f"Original image shape: {noisy_image.shape}")
print(f"Receptive field radius: {receptive_field_radius}")

# Define patch size (smaller than image for demonstration)
patch_size = 128

# Test creating windows and masks
windows, masks, patch_positions = create_tiled_windows_and_masks(
    noisy_image, patch_size, receptive_field_radius, overlap_strategy="reflect"
)

print(f"Number of patches created: {len(windows)}")
print(f"First window shape: {windows[0].shape}")
print(f"First mask (top, left, bottom, right): {masks[0]}")
print(f"First patch position (top, left, bottom, right): {patch_positions[0]}")

# Calculate expected window size
expected_window_size = patch_size + 2 * receptive_field_radius
print(f"Expected window size: {expected_window_size} x {expected_window_size}")
print(f"Actual window size: {windows[0].shape[-2]} x {windows[0].shape[-1]}")

# %%


# %%
# Test the complete tiling process with DRUNet
print("Processing with tiling...")
tiled_result = process_large_image_with_tiling(
    model=drunet,
    image=noisy_image,
    patch_size=128,  # Use smaller patches
    receptive_field_radius=receptive_field_radius,
    overlap_strategy="reflect",
    sigma=sigma,  # Pass sigma to DRUNet
)

print(f"Original noisy image shape: {noisy_image.shape}")
print(f"Tiled result shape: {tiled_result.shape}")

# Compare with direct processing (if memory allows)
print("Processing without tiling...")
direct_result = drunet(noisy_image.unsqueeze(0), sigma=sigma).squeeze(0)
print(f"Direct result shape: {direct_result.shape}")

# Calculate difference
diff = torch.abs(tiled_result - direct_result)
print(f"Maximum absolute difference: {diff.max():.6f}")
print(f"Mean absolute difference: {diff.mean():.6f}")

# %%
# Visualize the results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original noisy image
axes[0, 0].imshow(noisy_image.permute(1, 2, 0).cpu().numpy())
axes[0, 0].set_title("Original Noisy Image")
axes[0, 0].axis("off")

# Direct processing result
axes[0, 1].imshow(direct_result.permute(1, 2, 0).cpu().detach().numpy())
axes[0, 1].set_title("Direct Processing")
axes[0, 1].axis("off")

# Tiled processing result
axes[0, 2].imshow(tiled_result.permute(1, 2, 0).cpu().detach().numpy())
axes[0, 2].set_title("Tiled Processing")
axes[0, 2].axis("off")

# Difference map
diff_normalized = (diff / diff.max()).permute(1, 2, 0).cpu().detach().numpy()
axes[1, 0].imshow(diff_normalized)
axes[1, 0].set_title(f"Difference Map\n(Max: {diff.max():.6f})")
axes[1, 0].axis("off")

# Close-up of a region to see differences
crop_y, crop_x = 200, 300
crop_size = 150
crop_slice_y = slice(crop_y, crop_y + crop_size)
crop_slice_x = slice(crop_x, crop_x + crop_size)

# Clip values to [0,1] for proper display
direct_crop = torch.clamp(direct_result[:, crop_slice_y, crop_slice_x], 0, 1)
tiled_crop = torch.clamp(tiled_result[:, crop_slice_y, crop_slice_x], 0, 1)

axes[1, 1].imshow(direct_crop.permute(1, 2, 0).cpu().detach().numpy())
axes[1, 1].set_title("Direct (Crop)")
axes[1, 1].axis("off")

axes[1, 2].imshow(tiled_crop.permute(1, 2, 0).cpu().detach().numpy())
axes[1, 2].set_title("Tiled (Crop)")
axes[1, 2].axis("off")

plt.tight_layout()
plt.show()

print(f"The tiling approach produces very similar results to direct processing.")
print(
    f"Small differences are expected at patch boundaries due to the receptive field handling."
)

# %%
# Example usage with different configurations
print("=" * 60)
print("TILING FUNCTION USAGE EXAMPLES")
print("=" * 60)

# Example 1: Very small patches for memory-constrained environments
print("\n1. Small patches (64x64) for very large images:")
small_patch_windows, small_patch_masks, small_patch_positions = (
    create_tiled_windows_and_masks(
        noisy_image,
        patch_size=64,
        receptive_field_radius=32,
        overlap_strategy="reflect",
    )
)
print(f"   Number of patches: {len(small_patch_windows)}")
print(
    f"   Window size: {small_patch_windows[0].shape[-2]}x{small_patch_windows[0].shape[-1]}"
)

# Example 2: Different overlap strategies
print("\n2. Different overlap strategies:")
for strategy in ["reflect", "constant", "edge"]:
    windows, _, _ = create_tiled_windows_and_masks(
        noisy_image,
        patch_size=128,
        receptive_field_radius=32,
        overlap_strategy=strategy,
    )
    print(f"   {strategy:10}: {len(windows)} patches")

# Example 3: Non-square patches
print("\n3. Non-square patches:")
rect_windows, rect_masks, rect_positions = create_tiled_windows_and_masks(
    noisy_image,
    patch_size=(100, 150),
    receptive_field_radius=32,
    overlap_strategy="reflect",
)
print(f"   Rectangular patches (100x150): {len(rect_windows)} patches")
print(f"   Window size: {rect_windows[0].shape[-2]}x{rect_windows[0].shape[-1]}")

print("\n4. Complete processing example:")
print(
    "   This is how you would use it for a very large image that doesn't fit in memory:"
)
print("   ```python")
print("   # For a huge image")
print("   result = process_large_image_with_tiling(")
print("       model=drunet,")
print("       image=huge_image,")
print("       patch_size=256,  # Adjust based on your GPU memory")
print("       receptive_field_radius=32,")
print("       overlap_strategy='reflect',")
print("       sigma=0.2")
print("   )")
print("   ```")


# %%
def visualize_tiling_strategy(
    image,
    patch_size,
    receptive_field_radius,
    overlap_strategy="reflect",
    max_patches_to_show=16,
):
    """
    Visualize the tiling strategy showing windows and patches positions.

    Args:
        image: Input image tensor
        patch_size: Size of patches
        receptive_field_radius: Receptive field radius
        overlap_strategy: Padding strategy
        max_patches_to_show: Maximum number of patches to highlight (for clarity)
    """
    # Get the tiling information
    windows, masks, patch_positions = create_tiled_windows_and_masks(
        image, patch_size, receptive_field_radius, overlap_strategy
    )

    # Image dimensions
    if image.ndim == 3:
        C, H, W = image.shape
    else:
        B, C, H, W = image.shape

    # Create the visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    img_display = (
        image.cpu().permute(1, 2, 0)
        if image.ndim == 3
        else image[0].cpu().permute(1, 2, 0)
    )
    img_display = torch.clamp(img_display, 0, 1)  # Clamp for display

    axes[0].imshow(img_display.numpy())
    axes[0].set_title("Original Image")
    axes[0].set_xlabel(f"Width: {W} pixels")
    axes[0].set_ylabel(f"Height: {H} pixels")

    # Show patches (non-overlapping regions)
    axes[1].imshow(img_display.numpy())

    # Color palette for different patches
    colors = plt.cm.Set3(
        np.linspace(0, 1, min(len(patch_positions), max_patches_to_show))
    )

    for i, (patch_pos, color) in enumerate(
        zip(patch_positions[:max_patches_to_show], colors)
    ):
        patch_top, patch_left, patch_bottom, patch_right = patch_pos

        # Draw patch boundary
        rect = plt.Rectangle(
            (patch_left, patch_top),
            patch_right - patch_left,
            patch_bottom - patch_top,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            alpha=0.8,
        )
        axes[1].add_patch(rect)

        # Add patch number
        axes[1].text(
            patch_left + 5,
            patch_top + 15,
            f"{i+1}",
            color=color,
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    axes[1].set_title(
        f"Patches (Non-overlapping)\nPatch size: {patch_size}×{patch_size}"
    )
    axes[1].set_xlabel(f"Total patches: {len(patch_positions)}")
    axes[1].set_ylabel("Height (pixels)")

    # Show windows (patches + receptive field padding)
    axes[2].imshow(img_display.numpy())

    for i, (patch_pos, color) in enumerate(
        zip(patch_positions[:max_patches_to_show], colors)
    ):
        patch_top, patch_left, patch_bottom, patch_right = patch_pos

        # Calculate window boundaries
        window_top = patch_top - receptive_field_radius
        window_left = patch_left - receptive_field_radius
        window_bottom = patch_bottom + receptive_field_radius
        window_right = patch_right + receptive_field_radius

        # Clamp to image boundaries for visualization
        window_top_vis = max(0, window_top)
        window_left_vis = max(0, window_left)
        window_bottom_vis = min(H, window_bottom)
        window_right_vis = min(W, window_right)

        # Draw window boundary (larger rectangle)
        window_rect = plt.Rectangle(
            (window_left_vis, window_top_vis),
            window_right_vis - window_left_vis,
            window_bottom_vis - window_top_vis,
            linewidth=2,
            edgecolor=color,
            facecolor=color,
            alpha=0.15,
        )
        axes[2].add_patch(window_rect)

        # Draw patch boundary (inner rectangle)
        patch_rect = plt.Rectangle(
            (patch_left, patch_top),
            patch_right - patch_left,
            patch_bottom - patch_top,
            linewidth=3,
            edgecolor=color,
            facecolor="none",
            alpha=0.9,
        )
        axes[2].add_patch(patch_rect)

        # Add patch number
        axes[2].text(
            patch_left + 5,
            patch_top + 15,
            f"{i+1}",
            color=color,
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
        )

    window_size = patch_size + 2 * receptive_field_radius
    axes[2].set_title(
        f"Windows (Patches + Receptive Field)\nWindow size: {window_size}×{window_size}"
    )
    axes[2].set_xlabel(f"Receptive field radius: {receptive_field_radius}")
    axes[2].set_ylabel("Height (pixels)")

    # Add legends
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="red", lw=3, label="Patch boundary"),
        Line2D([0], [0], color="red", lw=2, alpha=0.5, label="Window boundary"),
    ]
    axes[2].legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.show()

    # Print summary information
    print(f"Tiling Strategy Summary:")
    print(f"├─ Image size: {H} × {W}")
    print(f"├─ Patch size: {patch_size} × {patch_size}")
    print(f"├─ Receptive field radius: {receptive_field_radius}")
    print(f"├─ Window size: {window_size} × {window_size}")
    print(f"├─ Total patches: {len(patch_positions)}")
    print(f"├─ Overlap strategy: {overlap_strategy}")
    print(
        f"└─ Showing first {min(len(patch_positions), max_patches_to_show)} patches in visualization"
    )


# Visualize the current tiling strategy
visualize_tiling_strategy(
    noisy_image, patch_size, receptive_field_radius, "reflect", max_patches_to_show=12
)


# %%
def create_tiling_concept_diagram(patch_size=128, receptive_field_radius=32):
    """
    Create a conceptual diagram showing how patches and windows work.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Diagram 1: Single patch and window relationship
    ax1 = axes[0]

    # Draw the window (larger rectangle)
    window_size = patch_size + 2 * receptive_field_radius
    window_rect = plt.Rectangle(
        (0, 0),
        window_size,
        window_size,
        linewidth=3,
        edgecolor="blue",
        facecolor="lightblue",
        alpha=0.3,
    )
    ax1.add_patch(window_rect)

    # Draw the patch (inner rectangle)
    patch_rect = plt.Rectangle(
        (receptive_field_radius, receptive_field_radius),
        patch_size,
        patch_size,
        linewidth=3,
        edgecolor="red",
        facecolor="lightcoral",
        alpha=0.5,
    )
    ax1.add_patch(patch_rect)

    # Add dimensions
    ax1.annotate(
        "",
        xy=(0, -10),
        xytext=(window_size, -10),
        arrowprops=dict(arrowstyle="<->", lw=2),
    )
    ax1.text(
        window_size / 2,
        -20,
        f"Window: {window_size}×{window_size}",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    ax1.annotate(
        "",
        xy=(receptive_field_radius, -35),
        xytext=(receptive_field_radius + patch_size, -35),
        arrowprops=dict(arrowstyle="<->", lw=2, color="red"),
    )
    ax1.text(
        receptive_field_radius + patch_size / 2,
        -45,
        f"Patch: {patch_size}×{patch_size}",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="red",
    )

    # Receptive field annotations
    ax1.annotate(
        "",
        xy=(-5, receptive_field_radius),
        xytext=(-5, receptive_field_radius + patch_size),
        arrowprops=dict(arrowstyle="<->", lw=2, color="green"),
    )
    ax1.text(
        -15,
        receptive_field_radius + patch_size / 2,
        f"RF: {receptive_field_radius}",
        ha="right",
        va="center",
        fontsize=10,
        rotation=90,
        color="green",
    )

    ax1.set_xlim(-50, window_size + 20)
    ax1.set_ylim(-60, window_size + 20)
    ax1.set_aspect("equal")
    ax1.set_title(
        "Single Patch and Window Relationship", fontsize=14, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)

    # Add legend
    legend_elements = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="lightblue",
            edgecolor="blue",
            label="Window (fed to model)",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="lightcoral",
            edgecolor="red",
            label="Patch (output region)",
        ),
        plt.Line2D([0], [0], color="green", lw=2, label="Receptive field padding"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right")

    # Diagram 2: Multiple patches tiling
    ax2 = axes[1]

    # Create a grid showing how patches tile
    grid_patches = 4  # 4x4 grid for illustration
    total_size = grid_patches * patch_size

    colors = plt.cm.Set3(np.linspace(0, 1, grid_patches * grid_patches))

    for i in range(grid_patches):
        for j in range(grid_patches):
            # Patch position
            patch_x = j * patch_size
            patch_y = i * patch_size
            color = colors[i * grid_patches + j]

            # Draw patch
            patch_rect = plt.Rectangle(
                (patch_x, patch_y),
                patch_size,
                patch_size,
                linewidth=2,
                edgecolor="black",
                facecolor=color,
                alpha=0.6,
            )
            ax2.add_patch(patch_rect)

            # Draw window outline (only for first few patches to avoid clutter)
            if i < 2 and j < 2:
                window_x = patch_x - receptive_field_radius
                window_y = patch_y - receptive_field_radius
                window_rect = plt.Rectangle(
                    (window_x, window_y),
                    window_size,
                    window_size,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                    linestyle="--",
                    alpha=0.8,
                )
                ax2.add_patch(window_rect)

            # Add patch number
            ax2.text(
                patch_x + patch_size / 2,
                patch_y + patch_size / 2,
                f"{i*grid_patches + j + 1}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

    ax2.set_xlim(-receptive_field_radius - 10, total_size + receptive_field_radius + 10)
    ax2.set_ylim(-receptive_field_radius - 10, total_size + receptive_field_radius + 10)
    ax2.set_aspect("equal")
    ax2.set_title(
        "Non-overlapping Patch Tiling\n(Dashed lines show windows for patches 1-4)",
        fontsize=14,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3)

    # Add annotations for the concept
    ax2.text(
        total_size / 2,
        -receptive_field_radius - 30,
        "Each patch is processed with its surrounding context\n(receptive field padding)",
        ha="center",
        va="top",
        fontsize=11,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()

    print("Tiling Concept Explanation:")
    print("=" * 50)
    print(f"1. PATCHES (red/colored squares):")
    print(f"   • Size: {patch_size} × {patch_size} pixels")
    print(f"   • Non-overlapping regions that tile the entire image")
    print(f"   • Final output is assembled from these regions")
    print()
    print(f"2. WINDOWS (blue/dashed outlines):")
    print(f"   • Size: {window_size} × {window_size} pixels")
    print(f"   • Input to the neural network model")
    print(f"   • Include {receptive_field_radius}-pixel padding around each patch")
    print()
    print(f"3. RECEPTIVE FIELD PADDING:")
    print(f"   • Ensures the model has enough context to process patch edges properly")
    print(f"   • Without this, edges of patches would have artifacts")
    print(f"   • Padding strategy (reflect/constant/edge) handles image boundaries")


# Create the conceptual diagram
create_tiling_concept_diagram(patch_size, receptive_field_radius)

# %%
