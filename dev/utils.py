import os
import copy
from typing import List, Tuple

import numpy as np
import torch

# Optional matplotlib import
import matplotlib.pyplot as plt


import deepinv as dinv
from torchvision.transforms import ToTensor, Compose, CenterCrop
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from pathlib import Path


def setup_distributed():
    """
    Initialize distributed training environment.
    Handles single GPU, multi-GPU single node, and multi-node scenarios.
    """
    # Get distributed training parameters from environment variables (set by torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"Setup: LOCAL_RANK={local_rank}, RANK={rank}, WORLD_SIZE={world_size}")

    # Set default environment variables if not set
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355"

    # Determine device and backend
    if torch.cuda.is_available():
        # Handle case where LOCAL_RANK might be >= number of available GPUs
        num_gpus = torch.cuda.device_count()
        if local_rank >= num_gpus:
            print(
                f"Warning: LOCAL_RANK {local_rank} >= available GPUs {num_gpus}, using GPU 0"
            )
            local_rank = local_rank % num_gpus

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
        print(f"Using CUDA device: {device}")
    else:
        device = torch.device("cpu")
        backend = "gloo"
        print(f"Using CPU device: {device}")

    # Initialize the process group only if we have multiple processes
    if world_size > 1:
        try:
            dist.init_process_group(
                backend=backend, init_method="env://", rank=rank, world_size=world_size
            )
            print(
                f"Rank {rank}/{world_size}: Initialized process group with device {device}"
            )
        except Exception as e:
            print(f"Failed to initialize distributed process group: {e}")
            print("Falling back to single process mode")
            world_size = 1
            rank = 0
    else:
        print(f"Single process mode with device {device}")

    return device, rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
            print("Distributed process group cleaned up")
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")


def print_gpu_info():
    """Print GPU information for debugging."""
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = int(os.environ.get("RANK", 0))
        num_gpus = torch.cuda.device_count()

        # Handle case where LOCAL_RANK might be >= available GPUs
        if local_rank >= num_gpus:
            local_rank = local_rank % num_gpus

        print(f"Rank {rank}, Local Rank {local_rank}")
        print(f"CUDA Device Count: {num_gpus}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name()}")
        print(
            f"Device Memory: {torch.cuda.get_device_properties(local_rank).total_memory // 1024**3} GB"
        )
    else:
        print("CUDA not available")


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


# ---------- DDP setup/teardown (FIXED VERSION) ----------
def setup_ddp():
    """
    FIXED VERSION: Properly setup DDP to avoid NCCL errors.
    """
    device, rank, world_size, local_rank = setup_distributed()
    return local_rank, rank, world_size, device


def cleanup_ddp():
    """Clean up DDP resources safely."""
    try:
        if dist.is_initialized():
            dist.barrier()
            cleanup_distributed()
        else:
            print("No distributed process group to clean up")
    except Exception as e:
        print(f"Warning: Error during DDP cleanup: {e}")
        # Force cleanup even if there's an error
        try:
            cleanup_distributed()
        except:
            pass


# ---------- Inference over windows with DDP (FIXED VERSION) ----------
@torch.no_grad()
def ddp_infer_windows(
    model: torch.nn.Module,
    windows: List[torch.Tensor],
    batch_size: int = 8,
    num_workers: int = 4,
    use_amp: bool = True,
    **model_kwargs,
):
    """
    FIXED VERSION: Returns (only on rank 0): list of output tensors aligned to the input windows order.
    Other ranks return None. Handles single GPU, multi-GPU single node, and multi-node scenarios.

    Key fixes:
    - Proper device assignment using LOCAL_RANK
    - Correct DDP initialization
    - Better error handling
    - Support for single GPU scenario
    """
    try:
        local_rank, rank, world_size, device = setup_ddp()

        print(f"Rank {rank}: Using device {device} (local_rank: {local_rank})")
        print_gpu_info()

        # For single GPU or non-distributed scenario, handle differently
        if world_size == 1:
            # Single process mode - process all windows directly
            print(f"Single process mode: processing {len(windows)} windows")

            # Move model to device
            model = copy.deepcopy(model).to(device).eval()

            # Create simple DataLoader without DistributedSampler
            dataset = WindowListDataset(windows)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                collate_fn=collate_pad,
            )

            local_results: List[Tuple[int, torch.Tensor]] = []

            if torch.cuda.is_available() and use_amp:
                autocast = torch.cuda.amp.autocast
            else:
                from contextlib import nullcontext

                autocast = nullcontext

            print(f"Processing {len(loader)} batches...")

            for batch_idx, (ids, x, shapes) in enumerate(loader):
                if batch_idx % 10 == 0:
                    print(f"Processing batch {batch_idx}/{len(loader)}")

                x = x.to(device, non_blocking=True)

                with torch.inference_mode():
                    if torch.cuda.is_available() and use_amp:
                        with autocast(enabled=True, dtype=torch.float16):
                            y = model(x, **model_kwargs)
                    else:
                        y = model(x, **model_kwargs)

                y = y.float().cpu()

                # Optional: crop back to original (h,w) if your model preserves spatial size
                if y.ndim == 4 and y.shape[-2:] == x.shape[-2:]:
                    # Handle variable-sized outputs properly
                    for k, (h, w) in enumerate(shapes):
                        cropped = y[k, :, :h, :w].clone()
                        idx = ids[k].item()
                        local_results.append((idx, cropped))
                else:
                    # If shapes don't match, use full outputs
                    for k, idx in enumerate(ids.tolist()):
                        local_results.append((idx, y[k]))

            # Prepare outputs in correct order
            outputs = [None] * len(windows)
            for idx, out in local_results:
                outputs[idx] = out

            print(f"Successfully processed all {len(windows)} windows")
            return outputs

        else:
            # Multi-process distributed mode
            print(f"Distributed mode: rank {rank}/{world_size}")

            # Build dataset/loader with DistributedSampler (no shuffling; we preserve order via indices)
            dataset = WindowListDataset(windows)
            sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=(num_workers > 0),
                collate_fn=collate_pad,
            )

            # Move model to this GPU and wrap with DDP
            model = copy.deepcopy(model).to(device).eval()

            # FIXED: Use local_rank directly instead of device.index
            print(f"Rank: {rank}, Local: {local_rank}, device_count {torch.cuda.device_count()}")
            ddp_model = DDP(
                model,
                device_ids=[local_rank] if torch.cuda.is_available() else None,
                output_device=local_rank if torch.cuda.is_available() else None,
            )

            local_results: List[Tuple[int, torch.Tensor]] = []

            if torch.cuda.is_available() and use_amp:
                autocast = torch.cuda.amp.autocast
            else:
                # Fallback for CPU or when AMP is disabled
                from contextlib import nullcontext

                autocast = nullcontext

            print(f"Rank {rank}: Processing {len(loader)} batches...")

            for batch_idx, (ids, x, shapes) in enumerate(loader):
                if batch_idx % 10 == 0:
                    print(f"Rank {rank}: Processing batch {batch_idx}/{len(loader)}")

                x = x.to(device, non_blocking=True)

                with torch.inference_mode():
                    if torch.cuda.is_available() and use_amp:
                        with autocast(enabled=True, dtype=torch.float16):
                            y = ddp_model(x, **model_kwargs)
                    else:
                        y = ddp_model(x, **model_kwargs)

                y = y.float().cpu()

                # Optional: crop back to original (h,w) if your model preserves spatial size
                # (this block is safe: if shapes match, we crop; otherwise we keep full y)
                if y.ndim == 4 and y.shape[-2:] == x.shape[-2:]:
                    # Handle variable-sized outputs properly for distributed processing
                    for k, (h, w) in enumerate(shapes):
                        cropped = y[k, :, :h, :w].clone()
                        idx = ids[k].item()
                        local_results.append((idx, cropped))
                else:
                    # If shapes don't match, use full outputs
                    for k, idx in enumerate(ids.tolist()):
                        local_results.append((idx, y[k]))

            print(f"Rank {rank}: Completed processing, gathering results...")

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
                missing_outputs = [i for i, o in enumerate(outputs) if o is None]
                if missing_outputs:
                    print(f"Warning: Missing outputs for indices: {missing_outputs}")
                else:
                    print(f"Successfully processed all {n} windows")

            return outputs

    except Exception as e:
        print(
            f"Rank {rank if 'rank' in locals() else 'unknown'}: Error in ddp_infer_windows: {e}"
        )
        raise e
    finally:
        cleanup_ddp()
