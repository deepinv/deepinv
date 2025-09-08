# %% [markdown]
# ## Tile strategies

# %%
import os
import copy
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

import deepinv as dinv
from torchvision.transforms import ToTensor, Compose, CenterCrop
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from pathlib import Path

save_dir = (
    Path(__file__).parent / "data/urban100"
)

torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Define base train dataset
dataset = dinv.datasets.Urban100HR(
    save_dir, download=True, transform=Compose([ToTensor()])
)

image = dataset[0]

# Finally, create a noisy version of the image with a fixed noise level sigma.
sigma = 0.2
noisy_image = image + sigma * torch.randn_like(image)

drunet = dinv.models.DRUNet()

# drunet_denoised = drunet(noisy_image.unsqueeze(0), sigma=sigma)

input_img = noisy_image.unsqueeze(0)

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



# ddp_windows.py



# ---------- Dataset over a list of windows ----------
class WindowListDataset(Dataset):
    """
    Wraps a Python list of window tensors shaped [C, H, W] (CPU).
    Yields (index, tensor) so we can restore original order after sharding.
    """
    def __init__(self, windows: List[torch.Tensor]):
        assert len(windows) > 0, "windows list is empty"
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int):
        return idx, self.windows[idx]


# ---------- Collate that pads variable-size windows per batch ----------
def collate_pad(batch: List[Tuple[int, torch.Tensor]]):
    """
    batch: list of (idx, [C,H,W] tensor)
    Returns:
      ids: 1D LongTensor of indices
      x  : [B,C,Hmax,Wmax] tensor (zero-padded)
      shapes: list of (h, w) for optional cropping of outputs
    """
    ids, tiles = zip(*batch)
    ids = torch.as_tensor(ids, dtype=torch.long)

    C = tiles[0].shape[0]
    Hmax = max(t.shape[1] for t in tiles)
    Wmax = max(t.shape[2] for t in tiles)

    x = tiles[0].new_zeros((len(tiles), C, Hmax, Wmax))
    shapes = []
    for i, t in enumerate(tiles):
        _, h, w = t.shape
        x[i, :, :h, :w] = t
        shapes.append((h, w))
    return ids, x, shapes


# ---------- DDP setup/teardown ----------
def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return local_rank, dist.get_rank(), dist.get_world_size()

def cleanup_ddp():
    dist.barrier()
    dist.destroy_process_group()


# ---------- Inference over windows with DDP ----------
@torch.no_grad()
def ddp_infer_windows(model: torch.nn.Module,
                      windows: List[torch.Tensor],
                      batch_size: int = 8,
                      num_workers: int = 4,
                      use_amp: bool = True):
    """
    Returns (only on rank 0): list of output tensors aligned to the input windows order.
    Other ranks return None.
    """
    local_rank, rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # Build dataset/loader with DistributedSampler (no shuffling; we preserve order via indices)
    dataset = WindowListDataset(windows)
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_pad,
    )

    # Move model to this GPU and wrap with DDP
    model = copy.deepcopy(model).to(device).eval()
    ddp_model = DDP(model, device_ids=[device.index], output_device=device.index)

    local_results: List[Tuple[int, torch.Tensor]] = []
    autocast = torch.cuda.amp.autocast

    for ids, x, shapes in loader:
        x = x.pin_memory().to(device, non_blocking=True)
        with torch.inference_mode(), autocast(enabled=use_amp, dtype=torch.float16):
            y = ddp_model(x)  # expect [B,C,Hmax,Wmax] here for this example model
        y = y.float().cpu()

        # Optional: crop back to original (h,w) if your model preserves spatial size
        # (this block is safe: if shapes match, we crop; otherwise we keep full y)
        if y.ndim == 4 and y.shape[-2:] == x.shape[-2:]:
            for k, (h, w) in enumerate(shapes):
                y[k] = y[k, :, :h, :w]

        for k, idx in enumerate(ids.tolist()):
            local_results.append((idx, y[k]))

    # Gather variable-length results to rank 0
    gather_list = [None] * world_size if rank == 0 else None
    dist.gather_object(local_results, gather_list, dst=0)

    outputs = None
    if rank == 0:
        # Flatten and reorder by original indices
        flat = [pair for worker in gather_list for pair in worker]
        n = len(windows)
        outputs = [None] * n
        for idx, out in flat:
            outputs[idx] = out
        # (optional) sanity check
        assert all(o is not None for o in outputs), "Missing some outputs after gather."

    cleanup_ddp()
    return outputs


# ---------- Demo entrypoint ----------
if __name__ == "__main__":
    # Example windows (CPU): mix of sizes to show padding works
    torch.manual_seed(0)

    # Run with: torchrun --nproc_per_node=<NUM_GPUS> ddp_windows.py
    outs = ddp_infer_windows(drunet, windows, batch_size=4, num_workers=4, use_amp=True)

    # Only rank 0 receives outputs
    if outs is not None:
        print(f"Got {len(outs)} outputs. Example shape[0]: {tuple(outs[0].shape)}")


# %%
