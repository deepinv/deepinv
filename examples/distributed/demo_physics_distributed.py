"""
Distributed Physics Operators
==============================

This example demonstrates how to distribute physics operators across multiple processes
for parallel computation of forward and adjoint operations.

**Usage:**

.. code-block:: bash

    # Single process
    python examples/distrib/demo_physics_distributed.py

    # Multi-process with torchrun (2 processes)
    python -m torch.distributed.run --nproc_per_node=2 examples/distrib/demo_physics_distributed.py

**Key Features:**

- Distribute multiple operators across processes
- Parallel forward operations (A)
- Parallel adjoint operations (A^T)
- Parallel composition (A^T A)
- Automatic result assembly from distributed processes

**Key Steps:**

1. Create multiple physics operators with different blur kernels
2. Stack them using dinv.physics.stack()
3. Initialize distributed context
4. Distribute physics with dinv.distributed.distribute()
5. Apply forward, adjoint, and composition operations
6. Visualize results
"""

# %%
import torch
import torch.nn.functional as F
from deepinv.physics import Blur, stack
from deepinv.physics.blur import gaussian_blur
from deepinv.utils.demo import load_example
from deepinv.utils.plotting import plot

# Import distributed framework
from deepinv.distributed import DistributedContext, distribute


# %%
def create_stacked_physics(device, img_size=1024):
    """
    Create stacked physics operators with different Gaussian blur kernels.

    :param device: Device to create operators on
    :param tuple img_size: Size of the image (H, W)
    :returns: Tuple of (stacked_physics, clean_image)
    """
    # Load example image
    img = load_example(
        "CBSD_0010.png", grayscale=False, device=device, img_size=img_size
    )

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
        gaussian_blur(sigma=2.5, device=str(device)),  # Medium blur
        gaussian_blur(
            sigma=(1.5, 3.5), angle=30, device=str(device)
        ),  # Anisotropic blur
    ]

    # Create physics operators (without noise for exact comparison)
    physics_list = []

    for kernel in kernels:
        # Create blur operator with circular padding
        blur_op = Blur(filter=kernel, padding="circular", device=str(device))
        blur_op = blur_op.to(device)

        physics_list.append(blur_op)

    # Stack physics operators into a single operator
    stacked_physics = stack(*physics_list)

    return stacked_physics, clean_image


"""Run distributed physics demonstration."""
# %%
# ============================================================================
# CONFIGURATION
# ============================================================================

img_size = 512


# %%
# ============================================================================
# DISTRIBUTED CONTEXT
# ============================================================================

# Initialize distributed context (handles single and multi-process automatically)
with DistributedContext(seed=42) as ctx:

    if ctx.rank == 0:
        print("=" * 70)
        print("Distributed Physics Operators Demo")
        print("=" * 70)
        print(f"\nRunning on {ctx.world_size} process(es)")
        print(f"   Device: {ctx.device}")

    # ============================================================================
    # STEP 1: Create stacked physics operators
    # ============================================================================

    stacked_physics, clean_image = create_stacked_physics(ctx.device, img_size=img_size)

    if ctx.rank == 0:
        print(f"\nCreated stacked physics with {len(stacked_physics)} operators")
        print(f"   Image shape: {clean_image.shape}")
        print(
            f"   Operator types: {[type(p).__name__ for p in stacked_physics.physics_list]}"
        )

    # ============================================================================
    # STEP 2: Distribute physics across processes
    # ============================================================================

    distributed_physics = distribute(stacked_physics, ctx)

    if ctx.rank == 0:
        print(f"\n🔧 Distributed physics created")
        print(
            f"   Local operators on this rank: {len(distributed_physics.local_indexes)}"
        )

    # ============================================================================
    # STEP 3: Test forward operation (A)
    # ============================================================================

    if ctx.rank == 0:
        print(f"\n🔄 Testing forward operation (A)...")

    # Apply distributed forward operation
    measurements = distributed_physics(clean_image)

    # Compare with non-distributed result (only on rank 0)
    measurements_ref = None
    if ctx.rank == 0:
        print(f"   Output type: {type(measurements).__name__}")
        print(f"   Number of measurements: {len(measurements)}")
        for i, m in enumerate(measurements):
            print(f"   Measurement {i} shape: {m.shape}")

        print(f"\n🔍 Comparing with non-distributed forward operation...")
        measurements_ref = stacked_physics(clean_image)

        max_diff = 0.0
        mean_diff = 0.0
        for i in range(len(measurements)):
            diff = torch.abs(measurements[i] - measurements_ref[i])
            max_diff = max(max_diff, diff.max().item())
            mean_diff += diff.mean().item()
        mean_diff /= len(measurements)

        print(f"   Mean absolute difference: {mean_diff:.2e}")
        print(f"   Max absolute difference:  {max_diff:.2e}")

        # Assert exact equality (should be zero for deterministic operations)
        assert (
            max_diff < 1e-6
        ), f"Distributed forward operation differs from non-distributed: max diff = {max_diff}"
        print(f"   Results match exactly!")

    # ============================================================================
    # STEP 4: Test adjoint operation (A^T)
    # ============================================================================

    if ctx.rank == 0:
        print(f"\nTesting adjoint operation (A^T)...")

    # Apply adjoint operation
    adjoint_result = distributed_physics.A_adjoint(measurements)

    if ctx.rank == 0:
        print(f"   Output shape: {adjoint_result.shape}")
        print(f"   Output norm: {torch.norm(adjoint_result).item():.4f}")

        # Compare with non-distributed result
        print(f"\nComparing with non-distributed adjoint operation...")
        assert measurements_ref is not None
        adjoint_ref = stacked_physics.A_adjoint(measurements_ref)
        diff = torch.abs(adjoint_result - adjoint_ref)
        print(f"   Mean absolute difference: {diff.mean().item():.2e}")
        print(f"   Max absolute difference:  {diff.max().item():.2e}")

        # Assert exact equality
        assert (
            diff.max().item() < 1e-6
        ), f"Distributed adjoint differs from non-distributed: max diff = {diff.max().item()}"
        print(f"   Results match exactly!")

    # ============================================================================
    # STEP 5: Test composition (A^T A)
    # ============================================================================

    if ctx.rank == 0:
        print(f"\nTesting composition (A^T A)...")

    # Apply composition
    ata_result = distributed_physics.A_adjoint_A(clean_image)

    if ctx.rank == 0:
        print(f"   Output shape: {ata_result.shape}")
        print(f"   Output norm: {torch.norm(ata_result).item():.4f}")

        # Compare with non-distributed result
        print(f"\nComparing with non-distributed A^T A operation...")
        ata_ref = stacked_physics.A_adjoint_A(clean_image)
        diff = torch.abs(ata_result - ata_ref)
        print(f"   Mean absolute difference: {diff.mean().item():.2e}")
        print(f"   Max absolute difference:  {diff.max().item():.2e}")

        # Assert exact equality
        assert (
            diff.max().item() < 1e-6
        ), f"Distributed A^T A differs from non-distributed: max diff = {diff.max().item()}"
        print(f"   Results match exactly!")

    # ============================================================================
    # STEP 6: Visualize results (only on rank 0)
    # ============================================================================

    if ctx.rank == 0:
        print(f"\nVisualizing results...")

        # Plot original image and measurements
        images_to_plot = [clean_image] + [m for m in measurements]
        titles = ["Original Image"] + [
            f"Measurement {i+1}" for i in range(len(measurements))
        ]

        plot(
            images_to_plot,
            titles=titles,
            save_fn="distributed_physics_forward.png",
            figsize=(15, 4),
        )

        # Plot adjoint and A^T A results
        # Normalize for visualization
        adjoint_vis = (adjoint_result - adjoint_result.min()) / (
            adjoint_result.max() - adjoint_result.min() + 1e-8
        )
        ata_vis = (ata_result - ata_result.min()) / (
            ata_result.max() - ata_result.min() + 1e-8
        )

        plot(
            [clean_image, adjoint_vis, ata_vis],
            titles=["Original", r"$A^T(y)$", r"$A^T A(x)$"],
            save_fn="distributed_physics_adjoint.png",
            figsize=(12, 4),
        )

        print(f"\n Demo completed successfully!")
        print(f"   Results saved to:")
        print(f"   - distributed_physics_forward.png")
        print(f"   - distributed_physics_adjoint.png")
        print("\n" + "=" * 70)
