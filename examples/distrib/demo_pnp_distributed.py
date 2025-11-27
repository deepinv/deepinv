"""
Distributed Plug-and-Play (PnP) Reconstruction
==============================================

This example demonstrates how to use the distributed framework for PnP reconstruction
using the simplified distribute() API. The framework automatically distributes physics
operators and denoisers across multiple processes.

**Usage:**

.. code-block:: bash

    # Single process
    python examples/distrib/demo_pnp_distributed.py

    # Multi-process with torchrun (2 processes)
    python -m torch.distributed.run --nproc_per_node=2 examples/distrib/demo_pnp_distributed.py

**Key Features:**

- Distribute multiple physics operators across processes
- Distribute denoiser with image tiling
- PnP algorithm with distributed components
- L2 data fidelity gradient computed using data_fidelity.grad()

**Key Steps:**

1. Create stacked physics operators and measurements with reproducible noise
2. Initialize distributed context
3. Distribute physics with dinv.distrib.distribute()
4. Distribute denoiser with tiling configuration
5. Create PnP prior and L2 data fidelity
6. Run PnP iterations using data_fidelity.grad() for gradient computation
7. Visualize results and track convergence
"""

import torch
import torch.nn.functional as F
from deepinv.physics import GaussianNoise, stack
from deepinv.physics.blur import Blur, gaussian_blur
from deepinv.utils.demo import load_example
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.loss.metric import PSNR
from deepinv.utils.plotting import plot
from deepinv.models import DRUNet
from typing import cast

# Import the distributed framework
from deepinv.distrib import DistributedContext, distribute, DistributedLinearPhysics


# ============================================================================
# DATA SETUP
# ============================================================================


def create_physics_and_measurements(device, img_size=1024, seed=42):
    """
    Create stacked physics operators and measurements using example images.

    :param device: Device to create operators on
    :param tuple img_size: Size of the image (H, W)
    :param int seed: Random seed for reproducible noise generation

    :returns: Tuple of (stacked_physics, measurements, clean_image)
    """
    # Load example image in original size
    img = load_example("CBSD_0010.png", grayscale=False, device=device)

    # Resize image so that max dimension equals img_size
    _, _, h, w = img.shape
    max_dim = max(h, w)

    if max_dim != img_size:
        scale_factor = img_size / max_dim
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)

        clean_image = F.interpolate(
            img, size=(new_h, new_w), mode="bicubic", align_corners=False
        )
    else:
        clean_image = img

    # Create different Gaussian blur kernels
    kernels = [
        gaussian_blur(sigma=1.0, device=str(device)),  # Small blur
        gaussian_blur(sigma=2.0, device=str(device)),  # Medium blur
        gaussian_blur(
            sigma=(1.5, 3.0), angle=30, device=str(device)
        ),  # Anisotropic blur
    ]

    # Noise levels for each operator
    noise_levels = [0.03, 0.05, 0.04]

    # Create physics operators
    physics_list = []

    for i, (kernel, noise_level) in enumerate(zip(kernels, noise_levels)):
        # Create blur operator with circular padding
        blur_op = Blur(filter=kernel, padding="circular", device=str(device))

        # Set the noise model with reproducible random generator
        rng = torch.Generator(device=device).manual_seed(seed + i)
        blur_op.noise_model = GaussianNoise(sigma=noise_level, rng=rng)
        blur_op = blur_op.to(device)

        physics_list.append(blur_op)

    # Stack physics operators into a single operator
    stacked_physics = stack(*physics_list)

    # Generate measurements (returns a TensorList)
    measurements = stacked_physics(clean_image)

    return stacked_physics, measurements, clean_image


def main():
    """Run distributed PnP reconstruction."""

    # ============================================================================
    # CONFIGURATION
    # ============================================================================

    num_iterations = 20
    step_size = 0.5
    denoiser_sigma = 0.05
    img_size = 512
    patch_size = 256
    receptive_field_size = 64

    # ============================================================================
    # DISTRIBUTED CONTEXT
    # ============================================================================

    # Initialize distributed context (handles single and multi-process automatically)
    with DistributedContext(seed=42) as ctx:

        if ctx.rank == 0:
            print("=" * 70)
            print("Distributed PnP Reconstruction")
            print("=" * 70)
            print(f"\nRunning on {ctx.world_size} process(es)")
            print(f"   Device: {ctx.device}")

        # ============================================================================
        # STEP 1: Create stacked physics operators and measurements
        # ============================================================================

        stacked_physics, measurements, clean_image = create_physics_and_measurements(
            ctx.device, img_size=img_size
        )

        if ctx.rank == 0:
            print(f"\nCreated stacked physics with {len(stacked_physics)} operators")
            print(f"   Image shape: {clean_image.shape}")
            print(f"   Measurements type: {type(measurements).__name__}")

        # ============================================================================
        # STEP 2: Distribute physics operators
        # ============================================================================

        if ctx.rank == 0:
            print(f"\n🔧 Distributing physics operators...")

        distributed_physics = distribute(stacked_physics, ctx)

        if ctx.rank == 0:
            print(f"   Distributed physics created")
            print(
                f"   Local operators on this rank: {len(distributed_physics.local_indexes)}"
            )

        # ============================================================================
        # STEP 3: Create L2 data fidelity
        # ============================================================================

        data_fidelity = L2()

        if ctx.rank == 0:
            print(f"\nCreated L2 data fidelity")

        # ============================================================================
        # STEP 4: Distribute denoiser with tiling
        # ============================================================================

        if ctx.rank == 0:
            print(f"\n🔧 Loading and distributing denoiser...")
            print(f"   Patch size: {patch_size}x{patch_size}")
            print(f"   Receptive field radius: {receptive_field_size}")

        denoiser = DRUNet(pretrained="download").to(ctx.device)

        distributed_denoiser = distribute(
            denoiser,
            ctx,
            patch_size=patch_size,
            receptive_field_size=receptive_field_size,
        )

        if ctx.rank == 0:
            print(f"   Distributed denoiser created")

        # ============================================================================
        # STEP 5: Create PnP prior with distributed denoiser
        # ============================================================================

        prior = PnP(denoiser=distributed_denoiser)

        if ctx.rank == 0:
            print(f"\nCreated PnP prior with distributed denoiser")

        # ============================================================================
        # STEP 6: Run distributed PnP algorithm
        # ============================================================================

        if ctx.rank == 0:
            print(f"\nRunning PnP reconstruction ({num_iterations} iterations)...")

        # Initialize reconstruction with zeros
        x = torch.zeros_like(clean_image)

        # Track PSNR (only on rank 0)
        psnr_metric = PSNR()
        psnr_history = []

        # PnP iterations
        with torch.no_grad():
            for it in range(num_iterations):
                # Data fidelity gradient step using the data_fidelity.grad() method
                grad = data_fidelity.grad(x, measurements, distributed_physics)

                # Gradient descent step
                x = x - step_size * grad

                # Denoising step (proximal operator of prior)
                x = prior.prox(x, sigma_denoiser=denoiser_sigma)

                # Compute PSNR on rank 0
                if ctx.rank == 0:
                    psnr_val = psnr_metric(x, clean_image).item()
                    psnr_history.append(psnr_val)

                    if it == 0 or (it + 1) % 5 == 0:
                        print(
                            f"   Iteration {it+1}/{num_iterations}, PSNR: {psnr_val:.2f} dB"
                        )

        # ============================================================================
        # STEP 7: Compare with non-distributed PnP (only on rank 0)
        # ============================================================================

        if ctx.rank == 0:
            print(f"\nComparing with non-distributed PnP reconstruction...")

            # Run non-distributed PnP
            x_ref = torch.zeros_like(clean_image)
            with torch.no_grad():
                for it in range(num_iterations):
                    # Data fidelity gradient step using data_fidelity.grad()
                    grad_ref = data_fidelity.grad(x_ref, measurements, stacked_physics)
                    x_ref = x_ref - step_size * grad_ref

                    # Denoising step
                    x_ref = denoiser(x_ref, sigma=denoiser_sigma)

            # Compare results
            diff = torch.abs(x - x_ref)
            mean_diff = diff.mean().item()
            max_diff = diff.max().item()

            psnr_ref = psnr_metric(x_ref, clean_image).item()
            psnr_dist = psnr_metric(x, clean_image).item()

            print(f"   Non-distributed final PSNR: {psnr_ref:.2f} dB")
            print(f"   Distributed final PSNR:     {psnr_dist:.2f} dB")
            print(f"   PSNR difference:             {abs(psnr_dist - psnr_ref):.2f} dB")
            print(f"   Mean absolute difference:    {mean_diff:.2e}")
            print(f"   Max absolute difference:     {max_diff:.2e}")

            # Check that results are close
            assert (
                abs(psnr_dist - psnr_ref) < 1.0
            ), f"PSNR difference too large: {abs(psnr_dist - psnr_ref):.2f} dB"
            print(f"   Results match well!")

        # ============================================================================
        # STEP 8: Visualize results (only on rank 0)
        # ============================================================================

        if ctx.rank == 0:
            print(f"\nReconstruction completed!")
            print(f"   Final PSNR: {psnr_history[-1]:.2f} dB")

            # Plot results
            plot(
                [clean_image, measurements[0], x],
                titles=["Ground Truth", "Measurement (first)", "Reconstruction"],
                save_fn="distributed_pnp_result.png",
                figsize=(12, 4),
            )

            # Plot convergence curve
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 5))
            plt.plot(
                range(1, num_iterations + 1), psnr_history, marker="o", linewidth=2
            )
            plt.xlabel("Iteration", fontsize=12)
            plt.ylabel("PSNR (dB)", fontsize=12)
            plt.title("PnP Reconstruction Convergence", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("distributed_pnp_convergence.png", dpi=150)
            plt.close()

            print(f"\nResults saved:")
            print(f"   - distributed_pnp_result.png")
            print(f"   - distributed_pnp_convergence.png")
            print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
