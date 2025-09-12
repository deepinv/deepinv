#!/usr/bin/env python3
"""
Multi-GPU demonstration script for tiled image processing.

This script shows exactly how the GPU assignment and work distribution works
in multi-GPU scenarios.
"""

import os
import sys
from pathlib import Path

# Add the current directory to path to import utils
sys.path.append(str(Path(__file__).parent))

import torch
import deepinv as dinv
from torchvision.transforms import ToTensor, Compose

from utils import *


def demonstrate_gpu_assignment():
    """Demonstrate how GPU assignment works in multi-GPU setup"""
    print("=" * 60)
    print("GPU ASSIGNMENT DEMONSTRATION")
    print("=" * 60)

    # Get environment variables set by torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"Process Information:")
    print(f"  RANK (global): {rank}")
    print(f"  LOCAL_RANK (local): {local_rank}")
    print(f"  WORLD_SIZE (total processes): {world_size}")
    print(f"  Process ID: {os.getpid()}")

    if torch.cuda.is_available():
        print(f"\nGPU Information:")
        print(f"  Total GPUs available: {torch.cuda.device_count()}")
        print(f"  Current GPU device: {torch.cuda.current_device()}")
        print(f"  Assigned GPU: cuda:{local_rank}")
        print(f"  GPU name: {torch.cuda.get_device_name(local_rank)}")
        print(
            f"  GPU memory: {torch.cuda.get_device_properties(local_rank).total_memory // 1024**3} GB"
        )
    else:
        print(f"\nNo CUDA GPUs available - using CPU")

    print(f"\nWork Distribution:")
    print(
        f"  This process will handle windows: {local_rank}, {local_rank + world_size}, {local_rank + 2*world_size}, ..."
    )

    return local_rank, rank, world_size


def simulate_multi_gpu_workload():
    """Simulate a workload to show how it's distributed across GPUs"""
    local_rank, rank, world_size = demonstrate_gpu_assignment()

    # Create dummy data
    total_windows = 48  # Example: 48 image windows to process

    # Simulate how DistributedSampler would distribute work
    windows_per_gpu = total_windows // world_size
    remainder = total_windows % world_size

    # Calculate start and end indices for this process
    start_idx = rank * windows_per_gpu + min(rank, remainder)
    end_idx = start_idx + windows_per_gpu + (1 if rank < remainder else 0)

    my_windows = list(range(start_idx, end_idx))

    print(f"\nWork Distribution for {total_windows} windows across {world_size} GPUs:")
    print(f"  Process {rank} (GPU {local_rank}): handles windows {my_windows}")
    print(f"  Total windows for this process: {len(my_windows)}")

    # Simulate processing time
    import time

    print(f"\nSimulating processing...")
    time.sleep(1)  # Simulate work

    print(f"Process {rank}: Completed processing {len(my_windows)} windows")

    return my_windows


def main():
    """Main demonstration function"""
    print("🚀 MULTI-GPU TILED IMAGE PROCESSING DEMONSTRATION")

    # Show GPU assignment
    my_windows = simulate_multi_gpu_workload()

    # If you want to run actual image processing, uncomment below:
    """
    # Load real data and process
    save_dir = "/path/to/your/data"
    dataset = dinv.datasets.Urban100HR(save_dir, download=False, transform=Compose([ToTensor()]))
    image = dataset[0]
    sigma = 0.2
    noisy_image = image + sigma * torch.randn_like(image)
    
    # Create model
    drunet = dinv.models.DRUNet()
    
    # Create windows
    receptive_field_radius = 32
    patch_size = 128
    
    windows, masks, patch_positions = create_tiled_windows_and_masks(
        noisy_image, patch_size, receptive_field_radius, overlap_strategy="reflect"
    )
    
    # Process with distributed inference
    outputs = ddp_infer_windows(
        drunet, windows, batch_size=4, num_workers=2, use_amp=True, sigma=sigma
    )
    
    if outputs is not None:
        print(f"✅ SUCCESS: Processed {len(outputs)} windows")
    """

    print(f"\n✅ Process {os.environ.get('RANK', 0)} completed successfully!")


if __name__ == "__main__":
    main()
