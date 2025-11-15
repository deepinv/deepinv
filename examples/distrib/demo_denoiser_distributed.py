"""
Distributed Denoiser with Image Tiling
=======================================

This example demonstrates how to distribute a denoiser across multiple processes
using image tiling for large-scale image processing.

**Usage:**

.. code-block:: bash

    # Single process
    python examples/distrib/demo_denoiser_distributed.py

    # Multi-process with torchrun (2 processes)
    python -m torch.distributed.run --nproc_per_node=2 examples/distrib/demo_denoiser_distributed.py

**Key Features:**

- Distribute denoising across multiple processes using image tiling
- Automatic patch extraction and reassembly
- Memory-efficient processing of large images

**Key Steps:**

1. Load a large test image
2. Add noise to create a noisy observation
3. Initialize distributed context
4. Configure tiling parameters
5. Distribute denoiser with dinv.distrib.distribute()
6. Apply distributed denoising
7. Visualize results and compute metrics
"""

import torch
from deepinv.models import DRUNet
from deepinv.utils.demo import load_example
from deepinv.utils.plotting import plot
from deepinv.loss.metric import PSNR

# Import distributed framework
from deepinv.distrib import DistributedContext, distribute


def create_noisy_image(device, img_size=(1024, 1024), noise_sigma=0.1, seed=42):
    """
    Create a noisy test image.

    :param device: Device to create image on
    :param tuple img_size: Size of the image (H, W)
    :param float noise_sigma: Standard deviation of Gaussian noise
    :param int seed: Random seed for reproducible noise
    :returns: Tuple of (clean_image, noisy_image, noise_sigma)
    """
    # Load example image
    clean_image = load_example(
        "CBSD_0010.png", grayscale=False, device=device, img_size=img_size
    )

    # Set seed for reproducible noise
    torch.manual_seed(seed)
    
    # Add Gaussian noise
    noise = torch.randn_like(clean_image) * noise_sigma
    noisy_image = clean_image + noise

    # Clip to valid range
    noisy_image = torch.clamp(noisy_image, 0, 1)

    return clean_image, noisy_image, noise_sigma


def main():
    """Run distributed denoiser demonstration."""

    # ============================================================================
    # CONFIGURATION
    # ============================================================================

    img_size = (1024, 1024)  # Large image for demonstrating tiling
    noise_sigma = 0.1
    patch_size = 256  # Size of each patch
    receptive_field_size = 32  # Overlap for smooth boundaries

    # ============================================================================
    # DISTRIBUTED CONTEXT
    # ============================================================================

    # Initialize distributed context (handles single and multi-process automatically)
    with DistributedContext(seed=42) as ctx:

        if ctx.rank == 0:
            print("=" * 70)
            print("🚀 Distributed Denoiser Demo")
            print("=" * 70)
            print(f"\n📊 Running on {ctx.world_size} process(es)")
            print(f"   Device: {ctx.device}")

        # ============================================================================
        # STEP 1: Create test image with noise
        # ============================================================================

        clean_image, noisy_image, sigma = create_noisy_image(
            ctx.device, img_size=img_size, noise_sigma=noise_sigma
        )

        # Compute input PSNR (create metric on all ranks for consistency)
        psnr_metric = PSNR()
        input_psnr = psnr_metric(noisy_image, clean_image).item()

        if ctx.rank == 0:
            print(f"\n✅ Created test image")
            print(f"   Image shape: {clean_image.shape}")
            print(f"   Noise sigma: {sigma}")
            print(f"   Input PSNR: {input_psnr:.2f} dB")

        # ============================================================================
        # STEP 2: Load denoiser model
        # ============================================================================

        if ctx.rank == 0:
            print(f"\n🔄 Loading DRUNet denoiser...")

        denoiser = DRUNet(pretrained="download").to(ctx.device)

        if ctx.rank == 0:
            print(f"   ✅ Denoiser loaded")

        # ============================================================================
        # STEP 3: Distribute denoiser with tiling configuration
        # ============================================================================

        if ctx.rank == 0:
            print(f"\n🔧 Configuring distributed denoiser")
            print(f"   Patch size: {patch_size}x{patch_size}")
            print(f"   Receptive field radius: {receptive_field_size}")
            print(f"   Tiling strategy: smart_tiling")

        distributed_denoiser = distribute(
            denoiser,
            ctx,
            patch_size=patch_size,
            receptive_field_size=receptive_field_size,
        )

        if ctx.rank == 0:
            print(f"   ✅ Distributed denoiser created")

        # ============================================================================
        # STEP 4: Apply distributed denoising
        # ============================================================================

        if ctx.rank == 0:
            print(f"\n🔄 Applying distributed denoising...")

        with torch.no_grad():
            denoised_image = distributed_denoiser(noisy_image, sigma=sigma)

        if ctx.rank == 0:
            print(f"   ✅ Denoising completed")
            print(f"   Output shape: {denoised_image.shape}")

        # Compare with non-distributed result (only on rank 0)
        if ctx.rank == 0:
            print(f"\n🔍 Comparing with non-distributed denoising...")
            with torch.no_grad():
                denoised_ref = denoiser(noisy_image, sigma=sigma)
            
            diff = torch.abs(denoised_image - denoised_ref)
            mean_diff = diff.mean().item()
            max_diff = diff.max().item()
            
            print(f"   Mean absolute difference: {mean_diff:.2e}")
            print(f"   Max absolute difference:  {max_diff:.2e}")
            
            # Check that differences are small (due to tiling boundary effects)
            # The distributed version uses tiling with overlapping patches and blending,
            # which can produce slightly different results at patch boundaries.
            # These differences are typically very small (< 0.01 mean, < 0.5 max).
            tolerance_mean = 0.01
            tolerance_max = 0.5
            assert mean_diff < tolerance_mean, f"Mean difference too large: {mean_diff:.4f} (tolerance: {tolerance_mean})"
            assert max_diff < tolerance_max, f"Max difference too large: {max_diff:.4f} (tolerance: {tolerance_max})"
            print(f"   ✅ Results are very close (within tolerance)!")

        # ============================================================================
        # STEP 5: Compute metrics and visualize results (only on rank 0)
        # ============================================================================

        if ctx.rank == 0:
            # Compute output PSNR
            output_psnr = psnr_metric(denoised_image, clean_image).item()
            psnr_improvement = output_psnr - input_psnr

            print(f"\n📊 Results:")
            print(f"   Input PSNR:  {input_psnr:.2f} dB")
            print(f"   Output PSNR: {output_psnr:.2f} dB")
            print(f"   Improvement: {psnr_improvement:.2f} dB")

            # Plot results
            plot(
                [clean_image, noisy_image, denoised_image],
                titles=[
                    "Clean Image",
                    f"Noisy (PSNR: {input_psnr:.2f} dB)",
                    f"Denoised (PSNR: {output_psnr:.2f} dB)",
                ],
                save_fn="distributed_denoiser_result.png",
                figsize=(15, 4),
            )

            # Plot zoom on a region to see details
            # Extract a 256x256 patch from center
            h, w = clean_image.shape[-2:]
            y_start, x_start = h // 2 - 128, w // 2 - 128
            y_end, x_end = y_start + 256, x_start + 256

            clean_patch = clean_image[..., y_start:y_end, x_start:x_end]
            noisy_patch = noisy_image[..., y_start:y_end, x_start:x_end]
            denoised_patch = denoised_image[..., y_start:y_end, x_start:x_end]

            plot(
                [clean_patch, noisy_patch, denoised_patch],
                titles=["Clean (zoom)", "Noisy (zoom)", "Denoised (zoom)"],
                save_fn="distributed_denoiser_zoom.png",
                figsize=(15, 4),
            )

            print(f"\n✅ Demo completed successfully!")
            print(f"   Results saved to:")
            print(f"   - distributed_denoiser_result.png")
            print(f"   - distributed_denoiser_zoom.png")
            print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
