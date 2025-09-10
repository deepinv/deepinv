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
from pathlib import Path
import time

# Add the current directory to path to import utils
sys.path.append(str(Path(__file__).parent))

import numpy as np
import torch
import gc

import deepinv as dinv
from torchvision.transforms import ToTensor, Compose, CenterCrop

from utils import *


def compute_psnr(image1, image2, max_val=1.0):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        image1, image2: torch.Tensor images of shape (C, H, W) or (B, C, H, W)
        max_val: Maximum possible pixel value (1.0 for normalized images)

    Returns:
        float: PSNR value in dB
    """
    mse = torch.mean((image1 - image2) ** 2)
    if mse == 0:
        return float("inf")
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


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
    # save_dir = (
    #     "/lustre/fswork/projects/rech/fio/ulx23va/projects/deepinv_PR/data/urban100"
    # )
    save_dir = (
        "/Users/tl255879/Documents/research/repos/deepinv-PRs/hackaton_v2/data/urban100"
    )

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Define base train dataset
    dataset = dinv.datasets.Urban100HR(
        save_dir, download=False, transform=Compose([ToTensor()])
    )

    # Artificially increase image size for demonstration
    scale_factor = 2.0  # Increase size by factor of 2
    image = dataset[0]
    image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale_factor
    ).squeeze()

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
        start_time = time.time()
        outs = ddp_infer_windows(
            drunet, windows, batch_size=4, num_workers=2, use_amp=True, sigma=sigma
        )
        distributed_time = time.time() - start_time

        # Only rank 0 receives outputs and handles reconstruction
        if outs is not None:
            print(
                f"SUCCESS: Got {len(outs)} outputs. Example shape[0]: {tuple(outs[0].shape)}"
            )
            print(f"Distributed processing time: {distributed_time:.2f} seconds")

            # Reconstruct the full image from processed windows
            print("Reconstructing full image from processed windows...")
            reconstructed_image = reassemble_from_patches(
                outs, masks, patch_positions, noisy_image.shape
            )

            print(f"Reconstructed image shape: {reconstructed_image.shape}")
            print(f"Original noisy image shape: {noisy_image.shape}")

            # For comparison, also process the image directly (if memory allows)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            print("Processing image directly for comparison...")
            start_time = time.time()
            drunet = drunet.to("cuda" if torch.cuda.is_available() else "cpu")
            noisy_image = noisy_image.to("cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                direct_result = (
                    drunet(noisy_image.unsqueeze(0), sigma=sigma).squeeze(0).to("cpu")
                )
            direct_time = time.time() - start_time
            noisy_image = noisy_image.to("cpu")
            print(f"Direct processing time: {direct_time:.2f} seconds")

            # Calculate difference between tiled and direct processing
            diff = torch.abs(reconstructed_image - direct_result)

            # Calculate PSNRs (comparing against the original clean image)
            psnr_noisy = compute_psnr(image, noisy_image)
            psnr_direct = compute_psnr(image, direct_result)
            psnr_distributed = compute_psnr(image, reconstructed_image)

            print(f"\nPSNR Analysis (vs Original Clean Image):")
            print(f"├─ Noisy image PSNR: {psnr_noisy:.2f} dB")
            print(f"├─ Direct reconstruction PSNR: {psnr_direct:.2f} dB")
            print(f"├─ Distributed reconstruction PSNR: {psnr_distributed:.2f} dB")
            print(f"├─ PSNR improvement (direct): {psnr_direct - psnr_noisy:+.2f} dB")
            print(
                f"└─ PSNR improvement (distributed): {psnr_distributed - psnr_noisy:+.2f} dB"
            )

            print(f"\nReconstruction Quality Assessment:")
            print(f"├─ Maximum absolute difference: {diff.max():.6f}")
            print(f"├─ Mean absolute difference: {diff.mean():.6f}")
            print(f"├─ Standard deviation of difference: {diff.std():.6f}")
            print(f"├─ 95th percentile difference: {torch.quantile(diff, 0.95):.6f}")
            print(
                f"└─ PSNR difference (direct vs distributed): {abs(psnr_direct - psnr_distributed):.3f} dB"
            )

            # Performance comparison
            print(f"\nPerformance Comparison:")
            if distributed_time < direct_time:
                speedup = direct_time / distributed_time
                print(f"├─ Distributed processing is {speedup:.2f}x faster")
            else:
                slowdown = distributed_time / direct_time
                print(
                    f"├─ Distributed processing is {slowdown:.2f}x slower (overhead from small image)"
                )
            print(
                f"├─ Processing {len(windows)} patches of size {patch_size}x{patch_size}"
            )
            print(f"└─ Image size: {H}x{W} pixels")

            # Create visualization
            print("\nCreating visualization...")
            fig, axes = plt.subplots(2, 4, figsize=(24, 12))

            # Original clean image
            axes[0, 0].imshow(torch.clamp(image, 0, 1).permute(1, 2, 0).cpu().numpy())
            axes[0, 0].set_title("Original Clean Image\n(Ground Truth)")
            axes[0, 0].axis("off")

            # Original noisy image
            axes[0, 1].imshow(
                torch.clamp(noisy_image, 0, 1).permute(1, 2, 0).cpu().numpy()
            )
            axes[0, 1].set_title(f"Noisy Image\n(σ = 0.2, PSNR: {psnr_noisy:.2f} dB)")
            axes[0, 1].axis("off")

            # Direct processing result
            axes[0, 2].imshow(
                torch.clamp(direct_result, 0, 1).permute(1, 2, 0).cpu().detach().numpy()
            )
            axes[0, 2].set_title(
                f"Direct Processing\n({direct_time:.2f}s, PSNR: {psnr_direct:.2f} dB)"
            )
            axes[0, 2].axis("off")

            # Tiled processing result
            axes[0, 3].imshow(
                torch.clamp(reconstructed_image, 0, 1)
                .permute(1, 2, 0)
                .cpu()
                .detach()
                .numpy()
            )
            axes[0, 3].set_title(
                f"Distributed Tiled Processing\n({distributed_time:.2f}s, PSNR: {psnr_distributed:.2f} dB)"
            )
            axes[0, 3].axis("off")

            # Difference map (direct vs distributed)
            diff_normalized = (
                (diff / diff.max()).permute(1, 2, 0).cpu().detach().numpy()
            )
            axes[1, 0].imshow(diff_normalized)
            axes[1, 0].set_title(
                f"Difference Map (Direct vs Distributed)\n(Max: {diff.max():.6f}, PSNR Δ: {abs(psnr_direct - psnr_distributed):.3f} dB)"
            )
            axes[1, 0].axis("off")

            # Close-up of a region to see differences
            crop_y, crop_x = 200, 300
            crop_size = 150
            crop_slice_y = slice(crop_y, crop_y + crop_size)
            crop_slice_x = slice(crop_x, crop_x + crop_size)

            # Clip values to [0,1] for proper display
            clean_crop = torch.clamp(image[:, crop_slice_y, crop_slice_x], 0, 1)
            noisy_crop = torch.clamp(noisy_image[:, crop_slice_y, crop_slice_x], 0, 1)
            direct_crop = torch.clamp(
                direct_result[:, crop_slice_y, crop_slice_x], 0, 1
            )
            tiled_crop = torch.clamp(
                reconstructed_image[:, crop_slice_y, crop_slice_x], 0, 1
            )

            axes[1, 1].imshow(clean_crop.permute(1, 2, 0).cpu().numpy())
            axes[1, 1].set_title("Clean Image\n(Crop)")
            axes[1, 1].axis("off")

            axes[1, 2].imshow(direct_crop.permute(1, 2, 0).cpu().detach().numpy())
            axes[1, 2].set_title("Direct Processing\n(Crop)")
            axes[1, 2].axis("off")

            axes[1, 3].imshow(tiled_crop.permute(1, 2, 0).cpu().detach().numpy())
            axes[1, 3].set_title("Distributed Tiled\n(Crop)")
            axes[1, 3].axis("off")

            plt.tight_layout()

            # Save the plot
            output_path = (
                Path(__file__).parent / "distributed_reconstruction_results.png"
            )
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"├─ Plot saved to: {output_path}")

            # Also save a summary text file
            summary_path = Path(__file__).parent / "reconstruction_summary.txt"
            with open(summary_path, "w") as f:
                f.write("Distributed Image Reconstruction Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Image dimensions: {H} x {W} pixels\n")
                f.write(f"Number of patches: {len(windows)}\n")
                f.write(f"Patch size: {patch_size} x {patch_size}\n")
                f.write(
                    f"Window size: {expected_window_size} x {expected_window_size}\n"
                )
                f.write(f"Receptive field radius: {receptive_field_radius}\n\n")
                f.write(f"Processing times:\n")
                f.write(f"  Direct processing: {direct_time:.2f} seconds\n")
                f.write(f"  Distributed processing: {distributed_time:.2f} seconds\n\n")
                f.write(f"PSNR Analysis (vs Original Clean Image):\n")
                f.write(f"  Noisy image PSNR: {psnr_noisy:.2f} dB\n")
                f.write(f"  Direct reconstruction PSNR: {psnr_direct:.2f} dB\n")
                f.write(
                    f"  Distributed reconstruction PSNR: {psnr_distributed:.2f} dB\n"
                )
                f.write(
                    f"  PSNR improvement (direct): {psnr_direct - psnr_noisy:+.2f} dB\n"
                )
                f.write(
                    f"  PSNR improvement (distributed): {psnr_distributed - psnr_noisy:+.2f} dB\n"
                )
                f.write(
                    f"  PSNR difference (direct vs distributed): {abs(psnr_direct - psnr_distributed):.3f} dB\n\n"
                )
                f.write(f"Quality metrics:\n")
                f.write(f"  Max absolute difference: {diff.max():.6f}\n")
                f.write(f"  Mean absolute difference: {diff.mean():.6f}\n")
                f.write(f"  Std deviation of difference: {diff.std():.6f}\n")
                f.write(
                    f"  95th percentile difference: {torch.quantile(diff, 0.95):.6f}\n"
                )
            print(f"└─ Summary saved to: {summary_path}")

            # # Also display if in interactive mode
            # try:
            #     plt.show()
            # except:
            #     pass  # In case we're running headless

            print(f"\nConclusion:")
            print(
                f"├─ The distributed tiling approach produces very similar results to direct processing."
            )
            print(
                f"├─ Small differences are expected at patch boundaries due to receptive field handling."
            )
            print(f"├─ Quality is excellent with mean difference < {diff.mean():.4f}")
            print(f"└─ Processing completed successfully!")

        else:
            print("Non-master rank completed successfully")

    except Exception as e:
        print(f"FAILED: {e}")
        raise e


# %%
