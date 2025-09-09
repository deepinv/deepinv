# %% [markdown]
# ## Tile strategies

#!/usr/bin/env python3
"""
Multi-GPU Distributed Data Parallel (DDP) example for tiled image processing.

This script shows how to properly distribute tiled image processing across multiple GPUs
to avoid NCCL conflicts and efficiently process large images.

Usage:
    # Single node, multiple GPUs (using torchrun)
    torchrun --nproc_per_node=4 example_ddp.py

    # Multiple nodes (using torchrun)
    torchrun --nnodes=2 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=HOST_NODE_ADDR example_ddp.py

    # SLURM cluster (using the provided slurm_example.sh)
    sbatch slurm_example.sh

Key fixes for NCCL errors:
1. Proper GPU device assignment using LOCAL_RANK
2. Correct device mapping in DDP initialization
3. Environment variable setup for distributed training
4. Proper cleanup and error handling
5. SLURM-compatible environment variable handling
"""

import os
import sys

sys.path.append(str(Path(__file__).parent / "utils.py"))

import numpy as np
import torch
import matplotlib.pyplot as plt

import deepinv as dinv
from torchvision.transforms import ToTensor, Compose, CenterCrop

from pathlib import Path

from utils import *

# ---------- Demo entrypoint (FIXED VERSION) ----------
if __name__ == "__main__":
    print("Starting distributed tiled image processing...")

    # Print environment info for debugging
    print(f"Environment variables:")
    print(f"  RANK: {os.environ.get('RANK', 'not set')}")
    print(f"  LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'not set')}")
    print(f"  WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'not set')}")
    print(f"  MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'not set')}")
    print(f"  MASTER_PORT: {os.environ.get('MASTER_PORT', 'not set')}")

    # Print SLURM info if available
    if "SLURM_JOB_ID" in os.environ:
        print(f"SLURM Job Information:")
        print(f"  Job ID: {os.environ.get('SLURM_JOB_ID', 'not set')}")
        print(f"  Node ID: {os.environ.get('SLURM_NODEID', 'not set')}")
        print(f"  Proc ID: {os.environ.get('SLURM_PROCID', 'not set')}")
        print(f"  Local ID: {os.environ.get('SLURM_LOCALID', 'not set')}")
        print(f"  Node list: {os.environ.get('SLURM_NODELIST', 'not set')}")

    # Set seed for reproducibility
    torch.manual_seed(0)

    # Data loading setup

    # save_dir = Path(__file__).parent / "data/urban100"
    save_dir = (
        "/lustre/fswork/projects/rech/fio/ulx23va/projects/deepinv_PR/data/urban100"
    )

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Define base train dataset
    dataset = dinv.datasets.Urban100HR(
        save_dir, download=False, transform=Compose([ToTensor()])
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

    # Run with: torchrun --nproc_per_node=<NUM_GPUS> example_ddp.py
    # Or with SLURM: sbatch slurm_example.sh
    try:
        outs = ddp_infer_windows(
            drunet, windows, batch_size=4, num_workers=2, use_amp=True, sigma=sigma
        )

        # Only rank 0 receives outputs
        if outs is not None:
            print(
                f"SUCCESS: Got {len(outs)} outputs. Example shape[0]: {tuple(outs[0].shape)}"
            )
        else:
            print("Non-master rank completed successfully")

    except Exception as e:
        print(f"FAILED: {e}")
        raise e


# %%
