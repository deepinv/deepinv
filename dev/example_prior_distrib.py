#!/usr/bin/env python3
"""
Distributed Prior Example with DRUNet Denoiser

Usage:
    # Single process
    uv run python dev/example_prior_distrib.py
    
    # Multi-process with torchrun
    uv run torchrun --nproc_per_node=2 dev/example_prior_distrib.py
    uv run torchrun --nproc_per_node=4 dev/example_prior_distrib.py

This script demonstrates the distributed framework for image denoising using
a DRUNet-based prior with 2D tiling strategy.

Key Features Demonstrated:
- DistributedContext: Handles process group initialization and device management
- DistributedSignal: Manages synchronized signal with automatic synchronization
- DistributedPrior: Distributes prior computation across processes using tiling
- 2D Tiling Strategy: Splits large images into smaller overlapping patches
- Plug-and-Play Prior: Uses DRUNet denoiser as a proximal operator

The example:
1. Downloads Urban100 dataset and selects a test image
2. Adds Gaussian noise to create a denoising problem
3. Creates a PnP prior using DRUNet denoiser
4. Sets up distributed computation with 2D tiling
5. Applies the prior in a distributed fashion across multiple processes
6. Compares distributed vs direct computation results
7. Saves a visualization showing the comparison

The distributed approach automatically:
- Splits the image into overlapping patches (tiles)
- Distributes patches across available processes using round-robin
- Applies the denoiser to each patch locally on each process
- Gathers and reassembles the results using the reduction function
- Synchronizes the final result across all processes

This enables processing very large images that wouldn't fit in GPU memory
by processing smaller patches in parallel across multiple devices/processes.
"""

import os
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.transforms import ToTensor, Compose

# Import deepinv components
import deepinv as dinv
from deepinv.optim.prior import PnP

# Import the distributed framework
from deepinv.distrib.distrib_framework import (
    DistributedContext,
    DistributedSignal,
    DistributedPrior,
)
from deepinv.distrib.utils import tiling_splitting_strategy, tiling_reduce_fn


def create_drunet_prior(device="cpu"):
    """Create a PnP prior using DRUNet denoiser."""
    # Load DRUNet model
    drunet = dinv.models.DRUNet()
    drunet = drunet.to(device)
    
    # Create PnP prior
    pnp_prior = PnP(denoiser=drunet)
    return pnp_prior


def load_urban100_test_image(device="cpu"):
    """Load a test image from Urban100 dataset."""
    # Set up save directory
    save_dir = Path(__file__).parent / "data" / "urban100"
    
    # Set seeds for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    # Define dataset
    dataset = dinv.datasets.Urban100HR(
        save_dir, download=True, transform=Compose([ToTensor()])
    )
    
    # Get first image
    image = dataset[0].to(device)
    
    # Add noise
    sigma = 0.2
    noisy_image = image + sigma * torch.randn_like(image)
    
    return image, noisy_image, sigma


def run_direct_denoising(prior, noisy_image, sigma, device="cpu"):
    """Run direct denoising without tiling for comparison."""
    with torch.no_grad():
        # Add batch dimension
        input_batch = noisy_image.unsqueeze(0)
        
        # Apply prior (which is just denoising in this case)
        denoised_batch = prior.prox(input_batch, sigma_denoiser=sigma)
        
        # Remove batch dimension
        denoised = denoised_batch.squeeze(0)
    
    return denoised


def main():
    """Main function demonstrating distributed prior with 2D tiling."""
    
    # Configuration - can be modified based on available memory
    use_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
    
    # Check available GPU memory and force CPU if insufficient
    if use_gpu:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        # For this demo, we need significant GPU memory due to the large image and DRUNet
        if gpu_memory_gb < 12:  # Less than 12GB, use CPU to be safe
            use_gpu = False
    
    patch_size = 256      # Larger patches on GPU
    receptive_field_radius = 64  # Larger RF on GPU
    non_overlap = True  # Allow overlap due to receptive field

    # Initialize distributed context
    with DistributedContext(sharding="round_robin", seed=42) as ctx:
        # Use GPU if available and not in a memory-constrained environment
        if not use_gpu:
            ctx.device = torch.device("cpu")

        if ctx.rank == 0:
            print(f"Running Distributed Prior Example")
            print(f"Processes: {ctx.world_size}")
            print(f"Device: {ctx.device}")
            print(f"Distributed: {ctx.is_dist}")
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"GPU Memory: {gpu_memory_gb:.1f} GB")
                if not use_gpu:
                    print(f"Using CPU due to limited GPU memory")
        
        # Load test image and create noisy version
        if ctx.rank == 0:
            print(f"Loading Urban100 dataset...")
        
        clean_image, noisy_image, sigma = load_urban100_test_image(ctx.device)
        B, C, H, W = 1, *noisy_image.shape  # Add batch dimension for signal shape
        
        if ctx.rank == 0:
            print(f"Image shape: {noisy_image.shape}")
            print(f"Noise level: {sigma}")
        
        # Create DRUNet-based prior
        if ctx.rank == 0:
            print(f"Creating DRUNet-based PnP prior...")
        
        pnp_prior = create_drunet_prior(ctx.device)
        
        # Create distributed signal from noisy image
        signal = DistributedSignal(ctx, shape=(B, C, H, W))
        signal.update_(noisy_image.unsqueeze(0))  # Add batch dimension
        
        if ctx.rank == 0:
            print(f"Created distributed signal with shape: {signal.shape}")
        
        # Create distributed prior with 2D tiling strategy
        if ctx.rank == 0:
            print(f"Setting up distributed prior with 2D tiling...")
            print(f"Patch size: {patch_size}x{patch_size}")
            print(f"Receptive field radius: {receptive_field_radius}")
        
        distributed_prior = DistributedPrior(
            ctx=ctx,
            prior=pnp_prior,
            splitting_strategy=tiling_splitting_strategy,
            signal_shape=(B, C, H, W),
            reduce_fn=tiling_reduce_fn,
            splitting_kwargs={
                "patch_size": patch_size,
                "receptive_field_radius": receptive_field_radius,
                "stride": None,  # Non-overlapping patches
                "hw_dims": (-2, -1),  # Height and width dimensions
                "non_overlap": non_overlap,  # Allow overlap due to receptive field
            },
        )
        
        if ctx.rank == 0:
            print(f"Local split indices (showing first 10): {distributed_prior.local_split_indices[:10]}...")
            print(f"Total splits: {distributed_prior.num_splits}")
            print(f"Patches per process: ~{distributed_prior.num_splits // ctx.world_size}")
        
        # Run distributed prior operation (prox)
        if ctx.rank == 0:
            print(f"Running distributed prior computation...")
        
        # Time the distributed computation
        start_time = torch.cuda.Event(enable_timing=True) if use_gpu else None
        end_time = torch.cuda.Event(enable_timing=True) if use_gpu else None
        
        if use_gpu:
            start_time.record()
        
        # Apply the distributed prior
        distributed_result_tensor = distributed_prior.prox(
            signal, 
            sigma_denoiser=sigma
        )
        
        if use_gpu:
            end_time.record()
            torch.cuda.synchronize()
            distributed_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        
        # Update signal with result
        signal.update_(distributed_result_tensor)
        
        if ctx.rank == 0:
            print(f"Distributed computation completed!")
            if use_gpu:
                print(f"Distributed computation time: {distributed_time:.2f}s")
        
        # For comparison, run direct denoising on rank 0
        if ctx.rank == 0:
            print(f"Running direct denoising for comparison...")
            
            if use_gpu:
                start_time.record()
            
            direct_result = run_direct_denoising(pnp_prior, noisy_image, sigma, ctx.device)
            
            if use_gpu:
                end_time.record()
                torch.cuda.synchronize()
                direct_time = start_time.elapsed_time(end_time) / 1000.0
                print(f"Direct computation time: {direct_time:.2f}s")
                print(f"Speedup: {direct_time / distributed_time:.2f}x")
            
            # Extract result from distributed signal (remove batch dimension)
            distributed_result = signal.data.squeeze(0)
            
            # Compute differences
            diff = torch.abs(distributed_result - direct_result)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            std_diff = diff.std().item()
            
            print(f"\nComparison Results:")
            print(f"Maximum absolute difference: {max_diff:.6f}")
            print(f"Mean absolute difference: {mean_diff:.6f}")
            print(f"Standard deviation of difference: {std_diff:.6f}")
            
            # Compute PSNR for both results
            def psnr(img1, img2):
                mse = torch.mean((img1 - img2) ** 2)
                if mse == 0:
                    return float('inf')
                return 20 * torch.log10(1.0 / torch.sqrt(mse))
            
            psnr_distributed = psnr(clean_image, distributed_result)
            psnr_direct = psnr(clean_image, direct_result)
            psnr_noisy = psnr(clean_image, noisy_image)
            
            print(f"\nPSNR Results:")
            print(f"Noisy image: {psnr_noisy:.2f} dB")
            print(f"Direct denoising: {psnr_direct:.2f} dB")
            print(f"Distributed denoising: {psnr_distributed:.2f} dB")
            print(f"PSNR difference: {abs(psnr_distributed - psnr_direct):.4f} dB")
            
            # Save visualization
            print(f"\nSaving visualization...")
            save_visualization(
                clean_image, noisy_image, direct_result, distributed_result, diff, non_overlap
            )
            
            print(f"Distributed prior example completed successfully!")
            
            # Verify the results are very close
            if max_diff < 0.1:  # Threshold for acceptable difference
                print(f"✓ Distributed and direct results match well (max diff: {max_diff:.6f})")
            else:
                print(f"⚠ Large difference detected (max diff: {max_diff:.6f})")
                
            # Print summary
            print(f"\n" + "="*60)
            print(f"DISTRIBUTED PRIOR EXAMPLE SUMMARY")
            print(f"="*60)
            print(f"Image size: {H}x{W}")
            print(f"Patch size: {patch_size}x{patch_size}")
            print(f"Receptive field: {receptive_field_radius}")
            print(f"Total patches: {distributed_prior.num_splits}")
            print(f"Processes: {ctx.world_size}")
            print(f"Device: {ctx.device}")
            print(f"Max difference: {max_diff:.6f}")
            print(f"PSNR improvement: {psnr_distributed - psnr_noisy:.2f} dB")
            if use_gpu and 'distributed_time' in locals() and 'direct_time' in locals():
                print(f"Speedup: {direct_time / distributed_time:.2f}x")
            print(f"="*60)


def save_visualization(clean, noisy, direct, distributed, diff, non_overlap):
    """Save visualization comparing all results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Convert to numpy for visualization
    def to_numpy(tensor):
        return tensor.permute(1, 2, 0).cpu().detach().numpy()
    
    # Original clean image
    axes[0, 0].imshow(to_numpy(clean))
    axes[0, 0].set_title("Clean Image")
    axes[0, 0].axis("off")
    
    # Noisy image
    axes[0, 1].imshow(to_numpy(noisy))
    axes[0, 1].set_title("Noisy Image")
    axes[0, 1].axis("off")
    
    # Direct denoising result
    axes[0, 2].imshow(to_numpy(torch.clamp(direct, 0, 1)))
    axes[0, 2].set_title("Direct Denoising")
    axes[0, 2].axis("off")
    
    # Distributed denoising result
    axes[1, 0].imshow(to_numpy(torch.clamp(distributed, 0, 1)))
    axes[1, 0].set_title("Distributed Denoising")
    axes[1, 0].axis("off")
    
    # Difference map
    diff_normalized = diff / diff.max()
    axes[1, 1].imshow(to_numpy(diff_normalized))
    axes[1, 1].set_title(f"Difference Map\n(Max: {diff.max():.6f})")
    axes[1, 1].axis("off")
    
    # Close-up comparison
    crop_size = 150
    H, W = clean.shape[-2:]  # Get height and width from image
    crop_y, crop_x = H//2 - crop_size//2, W//2 - crop_size//2
    crop_slice_y = slice(crop_y, crop_y + crop_size)
    crop_slice_x = slice(crop_x, crop_x + crop_size)
    
    direct_crop = torch.clamp(direct[:, crop_slice_y, crop_slice_x], 0, 1)
    distributed_crop = torch.clamp(distributed[:, crop_slice_y, crop_slice_x], 0, 1)
    
    # Create side-by-side comparison
    comparison = torch.cat([direct_crop, distributed_crop], dim=2)
    axes[1, 2].imshow(to_numpy(comparison))
    axes[1, 2].set_title("Direct | Distributed\n(Crop)")
    axes[1, 2].axis("off")
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / f"distributed_prior_results_nonoverlap_{non_overlap}.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
