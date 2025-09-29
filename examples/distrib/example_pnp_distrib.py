"""
Distributed Framework Tutorial
===============================

This example demonstrates how to use DeepInverse's distributed framework for solving inverse problems
across multiple processes. The distributed framework allows you to:

1. **Scale reconstruction algorithms** across multiple GPUs or CPU cores
2. **Distribute forward operators** (physics) across processes
3. **Apply priors distributedly** using smart tiling strategies
4. **Maintain consistency** between distributed and single-process results

The example walks through a **Plug-and-Play (PnP)** reconstruction algorithm using multiple
complementary blur operators distributed across processes.

**Usage:**

.. code-block:: bash

    # Single process
    python examples/distrib/example_pnp_distrib.py

    # Multi-process with torchrun
    torchrun --nproc_per_node=2 examples/distrib/example_pnp_distrib.py

**Key Concepts:**

- **DistributedContext**: Manages process groups and device selection
- **DistributedLinearPhysics**: Distributes forward operators across processes
- **DistributedDataFidelity**: Computes data fidelity terms distributedly
- **DistributedPrior**: Applies denoising priors with spatial tiling
- **DistributedSignal**: Synchronized signal representation across processes
"""

# %%
import os
import time
import matplotlib.pyplot as plt
import torch
import deepinv as dinv
from pathlib import Path
from typing import List, Tuple
from torchvision.transforms import ToTensor, Compose

# Import physics and loss components
from deepinv.physics import GaussianNoise
from deepinv.physics.blur import Blur
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.loss.metric import PSNR

# Import the distributed framework
from deepinv.distrib.distrib_framework import (
    DistributedContext,
    DistributedLinearPhysics,
    DistributedDataFidelity,
    DistributedMeasurements,
    DistributedSignal,
    DistributedPrior,
)

# %%
# Step 1: Setting up the Distributed Context
# ===========================================
#
# The :class:`deepinv.distrib.DistributedContext` is the foundation of the distributed framework.
# It handles:
#
# - **Process group initialization**: Automatically detects RANK/WORLD_SIZE environment variables
# - **Device selection**: Chooses appropriate GPU/CPU based on LOCAL_RANK
# - **Backend selection**: NCCL for GPU communication, Gloo for CPU
# - **Data sharding**: Distributes work across processes using round-robin or block strategies

# %%
# Step 2: Creating Physics Operators
# ===================================
#
# We'll create multiple complementary blur operators that will be distributed across processes.
# Each process will handle a subset of these operators.


def create_blur_kernels(device: torch.device) -> List[torch.Tensor]:
    """
    Create different complementary blur kernels for distributed physics.

    This creates 4 different blur kernels that simulate different degradation types:
    - Gaussian blur (simulates camera defocus)
    - Motion blur (simulates camera shake)
    - Edge detection (simulates high-pass filtering)
    - Box blur (simulates simple averaging)

    Args:
        device: PyTorch device to create kernels on

    Returns:
        List of blur kernels as tensors
    """
    kernels = []

    # 1. Gaussian blur kernel (5x5)
    gaussian_kernel = torch.zeros((1, 1, 5, 5), device=device)
    center = 2
    sigma = 1.0
    for i in range(5):
        for j in range(5):
            x, y = i - center, j - center
            gaussian_kernel[0, 0, i, j] = torch.exp(
                torch.tensor(-(x**2 + y**2) / (2 * sigma**2), device=device)
            )
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    kernels.append(gaussian_kernel)

    # 2. Motion blur kernel (horizontal)
    motion_kernel = torch.zeros((1, 1, 5, 5), device=device)
    motion_kernel[0, 0, 2, :] = 1.0 / 5.0  # Horizontal line
    kernels.append(motion_kernel)

    # 3. Edge detection kernel (Laplacian)
    edge_kernel = (
        torch.tensor(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], device=device, dtype=torch.float32
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    kernels.append(edge_kernel)

    # 4. Box blur kernel (3x3)
    box_kernel = torch.ones((1, 1, 3, 3), device=device) / 9.0
    kernels.append(box_kernel)

    return kernels


# %%
# Step 3: Loading Test Data
# =========================
#
# We'll use an image from the Urban100 dataset for this demonstration.


def load_test_image(device: torch.device, image_idx: int = 0) -> torch.Tensor:
    """
    Load a test image from Urban100 dataset.

    Args:
        device: Device to load image on
        image_idx: Index of image to load from dataset

    Returns:
        Clean test image tensor of shape (C, H, W)
    """
    # Create data directory
    save_dir = Path(__file__).parent / "data" / "urban100"

    # Set seeds for reproducibility
    torch.manual_seed(42)

    # Load Urban100 dataset
    dataset = dinv.datasets.Urban100HR(
        save_dir, download=True, transform=Compose([ToTensor()])
    )

    image = dataset[image_idx].to(device)

    return image


# %%
# Step 4: Creating Physics and Measurements
# ==========================================
#
# Now we create multiple physics operators and generate corresponding noisy measurements.


def create_physics_and_measurements(
    image: torch.Tensor, device: torch.device, noise_sigma: float = 0.05
) -> Tuple[List[Blur], List[torch.Tensor]]:
    """
    Create multiple blur physics operators and corresponding noisy measurements.

    This simulates a scenario where we have multiple complementary measurements
    of the same scene, each degraded by different blur kernels and noise.

    Args:
        image: Clean ground truth image
        device: Device to create operators on
        noise_sigma: Base noise level for measurements

    Returns:
        Tuple of (physics_list, measurements_list)
    """
    blur_kernels = create_blur_kernels(device)
    physics_list = []
    measurements_list = []

    for i, kernel in enumerate(blur_kernels):
        # Create blur physics operator
        physics = Blur(filter=kernel, device=device)

        # Add different noise levels for each measurement
        physics.noise_model = GaussianNoise(sigma=noise_sigma * (0.5 + 0.5 * i))

        # Generate noisy measurement with fixed seed for reproducibility
        torch.manual_seed(42 + i)
        measurement = physics(image.unsqueeze(0))

        physics_list.append(physics)
        measurements_list.append(measurement)

    return physics_list, measurements_list


# %%
# Step 5: Single-Process PnP Baseline
# ====================================
#
# First, let's implement a standard single-process PnP algorithm for comparison.


def run_single_process_pnp(
    clean_image: torch.Tensor,
    physics_list: List[Blur],
    measurements_list: List[torch.Tensor],
    num_iterations: int = 10,
    lr: float = 0.01,
    sigma_denoiser: float = 0.05,
) -> Tuple[torch.Tensor, List[float]]:
    """
    Run PnP algorithm in single process for baseline comparison.

    The PnP algorithm alternates between:
    1. Data fidelity gradient step: x = x - lr * ∇f(x) where f(x) = ||Ax - y||²
    2. Denoising step: x = denoise(x)

    Args:
        clean_image: Ground truth for PSNR computation
        physics_list: List of physics operators
        measurements_list: List of corresponding measurements
        num_iterations: Number of PnP iterations
        lr: Learning rate for gradient step
        sigma_denoiser: Noise level for denoiser

    Returns:
        Tuple of (final_reconstruction, psnr_history)
    """
    device = clean_image.device

    # Initialize reconstruction with noisy version
    x = torch.zeros_like(clean_image.unsqueeze(0))

    # Create denoiser
    drunet = dinv.models.DRUNet(pretrained="download").to(device)
    pnp_prior = PnP(denoiser=drunet)

    # Metrics
    psnr_metric = PSNR()
    psnr_history = []

    with torch.no_grad():
        # PnP iterations
        for it in range(num_iterations):
            # Data fidelity gradient step
            grad = torch.zeros_like(x)
            for physics, measurement in zip(physics_list, measurements_list):
                # Compute gradient: ∇f(x) = A^T(Ax - y)
                residual = physics.A(x) - measurement
                grad += physics.A_adjoint(residual)
            grad = grad / len(physics_list)  # Average gradients

            # Gradient step
            x = x - lr * grad

            # Denoising step (proximal operator of denoising prior)
            with torch.no_grad():
                x = pnp_prior.prox(x, sigma_denoiser=sigma_denoiser)

            # Compute PSNR
            psnr_val = psnr_metric(x, clean_image.unsqueeze(0)).item()
            psnr_history.append(psnr_val)

            if it == 0 or (it + 1) % 5 == 0:
                print(
                    f"  Single-process PnP iteration {it+1}/{num_iterations}, PSNR: {psnr_val:.2f} dB"
                )

        return x.squeeze(0), psnr_history


# %%
# Step 6: Distributed PnP Algorithm
# ==================================
#
# Now we implement the same algorithm using the distributed framework.


def run_distributed_pnp_example():
    """
    Main function demonstrating the distributed PnP algorithm.

    This function shows how to:
    1. Set up distributed context
    2. Create distributed physics operators
    3. Set up distributed measurements and signals
    4. Run distributed PnP algorithm
    5. Compare with single-process baseline
    """

    # Configuration
    num_iterations = 20
    lr = 0.1
    denoiser_sigma = 0.02
    noise_sigma = 0.1
    patch_size = 256
    receptive_field_radius = 64

    print("=" * 80)
    print("🚀 Distributed Framework Tutorial: Plug-and-Play Algorithm")
    print("=" * 80)

    # %%
    # Initialize Distributed Context
    # ==============================
    #
    # The context manager automatically handles process group initialization,
    # device selection, and cleanup.

    with DistributedContext(sharding="round_robin", seed=42) as ctx:

        print(f"\n📊 Process Information:")
        print(f"  • World size: {ctx.world_size} process(es)")
        print(f"  • Current rank: {ctx.rank}")
        print(f"  • Device: {ctx.device}")
        print(f"  • Distributed: {ctx.is_dist}")

        # Load test image (all processes load the same image)
        clean_image = load_test_image(ctx.device, image_idx=0)
        B, C, H, W = 1, *clean_image.shape

        if ctx.rank == 0:
            print(f"\n📷 Loaded test image: {clean_image.shape}")

        # %%
        # Create Physics Operators and Measurements
        # ==========================================

        physics_list, measurements_list = create_physics_and_measurements(
            clean_image, ctx.device, noise_sigma=noise_sigma
        )
        num_operators = len(physics_list)

        if ctx.rank == 0:
            print(f"\n🔬 Created {num_operators} physics operators:")
            print(f"  • Gaussian blur, Motion blur, Edge detection, Box blur")
            print(f"  • Each with different noise levels")

        # %%
        # Set up Distributed Components
        # ==============================
        #
        # Now we create the distributed versions of our components.

        # Factory functions for distributed framework
        def factory_physics(idx, device, shared):
            """Factory to create physics operators by index"""
            return physics_list[idx]

        def factory_data_fidelity(idx, device, shared):
            """Factory to create L2 data fidelity by index"""
            return L2()

        def factory_measurements(idx, device, shared):
            """Factory to access measurements by index"""
            return measurements_list[idx]

        if ctx.rank == 0:
            print(f"\n🔧 Setting up distributed components...")

        # Distributed physics operators
        distributed_physics = DistributedLinearPhysics(
            ctx, num_ops=num_operators, factory=factory_physics
        )

        # Distributed measurements
        distributed_measurements = DistributedMeasurements(
            ctx, num_items=num_operators, factory=factory_measurements
        )

        # Distributed signal (synchronized across processes)
        distributed_signal = DistributedSignal(ctx, shape=(B, C, H, W))

        # Distributed data fidelity
        distributed_df = DistributedDataFidelity(
            ctx,
            distributed_physics,
            distributed_measurements,
            data_fidelity_factory=factory_data_fidelity,
            reduction="mean",
        )

        # Distributed prior (DRUNet with 2D tiling)
        drunet_prior = PnP(
            denoiser=dinv.models.DRUNet(pretrained="download").to(ctx.device)
        )

        distributed_prior = DistributedPrior(
            ctx=ctx,
            prior=drunet_prior,
            strategy="smart_tiling",
            signal_shape=(B, C, H, W),
            strategy_kwargs={
                "patch_size": patch_size,
                "receptive_field_radius": receptive_field_radius,
                "overlap": True,
            },
        )

        if ctx.rank == 0:
            print(
                f"  ✅ Distributed physics: {len(distributed_physics.local_idx)} local operators"
            )
            print(
                f"  ✅ Distributed measurements: {len(distributed_measurements.local)} local measurements"
            )
            print(f"  ✅ Distributed signal: shape {distributed_signal.shape}")
            print(
                f"  ✅ Distributed prior: tiling with {patch_size}x{patch_size} patches"
            )

        # %%
        # Run Distributed PnP Algorithm
        # ==============================

        if ctx.rank == 0:
            print(
                f"\n🔄 Running distributed PnP algorithm ({num_iterations} iterations)..."
            )

        psnr_metric = PSNR()
        psnr_history_distributed = []

        with torch.no_grad():
            for it in range(num_iterations):
                # Data fidelity gradient step (distributed across processes)
                grad = distributed_df.grad(distributed_signal)

                # Gradient step
                new_data = distributed_signal.data - lr * grad
                distributed_signal.update_(new_data)

                # Denoising step (distributed prior with tiling)
                denoised = distributed_prior.prox(
                    distributed_signal, sigma_denoiser=denoiser_sigma
                )
                distributed_signal.update_(denoised)

                # Compute PSNR (only on rank 0)
                if ctx.rank == 0:
                    psnr_val = psnr_metric(
                        distributed_signal.data, clean_image.unsqueeze(0)
                    ).item()
                    psnr_history_distributed.append(psnr_val)

                    if it == 0 or (it + 1) % 5 == 0:
                        print(
                            f"  Distributed PnP iteration {it+1}/{num_iterations}, PSNR: {psnr_val:.2f} dB"
                        )

        # %%
        # Run Single-Process Comparison
        # ==============================

        single_psnr_history = []

        if ctx.rank == 0:  # Only run on main process
            print(f"\n🔄 Running single-process PnP for comparison...")

            single_recon, single_psnr_history = run_single_process_pnp(
                clean_image,
                physics_list,
                measurements_list,
                num_iterations=num_iterations,
                lr=lr,
                sigma_denoiser=denoiser_sigma,
            )

        # %%
        # Results and Visualization
        # =========================

        if ctx.rank == 0:
            print(f"\n📊 Results Summary:")
            print(f"  • Distributed final PSNR: {psnr_history_distributed[-1]:.2f} dB")
            print(f"  • Single-process final PSNR: {single_psnr_history[-1]:.2f} dB")
            print(
                f"  • PSNR difference: {abs(psnr_history_distributed[-1] - single_psnr_history[-1]):.4f} dB"
            )

            # Create visualization
            save_results_visualization(
                clean_image=clean_image,
                distributed_recon=distributed_signal.data.squeeze(0),
                single_recon=single_recon,
                psnr_distributed=psnr_history_distributed,
                psnr_single=single_psnr_history,
                measurements_list=measurements_list,
                ctx=ctx,
            )

            print(f"\n💾 Results saved to 'distributed_pnp_tutorial_results.png'")
            print("=" * 80)


# %%
# Step 7: Visualization and Analysis
# ===================================


def save_results_visualization(
    clean_image: torch.Tensor,
    distributed_recon: torch.Tensor,
    single_recon: torch.Tensor,
    psnr_distributed: List[float],
    psnr_single: List[float],
    measurements_list: List[torch.Tensor],
    ctx,
):
    """
    Save comprehensive visualization comparing distributed vs single-process results.
    """

    def to_numpy(tensor):
        """Convert tensor to numpy for visualization"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)
        if tensor.shape[-1] == 1:
            tensor = tensor.squeeze(-1)

        # Clip to valid range for visualization
        tensor = torch.clamp(tensor, 0, 1)
        return tensor.detach().cpu().numpy()

    # Create figure
    fig = plt.figure(figsize=(16, 10))

    # Row 1: Input data
    ax1 = plt.subplot(3, 4, 1)
    plt.imshow(to_numpy(clean_image), cmap="gray")
    plt.title("Ground Truth", fontsize=12, fontweight="bold")
    plt.axis("off")

    ax2 = plt.subplot(3, 4, 2)
    plt.imshow(to_numpy(measurements_list[0].squeeze(0)), cmap="gray")
    plt.title("Measurement 1\n(Gaussian Blur)", fontsize=11)
    plt.axis("off")

    ax3 = plt.subplot(3, 4, 3)
    plt.imshow(to_numpy(measurements_list[1].squeeze(0)), cmap="gray")
    plt.title("Measurement 2\n(Motion Blur)", fontsize=11)
    plt.axis("off")

    ax4 = plt.subplot(3, 4, 4)
    # Show first measurement as pseudo-inverse baseline
    pinv_recon = measurements_list[0].squeeze(0)
    # Resize to match ground truth if needed
    if pinv_recon.shape != clean_image.shape:
        pinv_recon = torch.nn.functional.interpolate(
            pinv_recon.unsqueeze(0),
            size=clean_image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    plt.imshow(to_numpy(pinv_recon), cmap="gray")
    psnr_pinv = PSNR()(pinv_recon.unsqueeze(0), clean_image.unsqueeze(0)).item()
    plt.title(f"Pseudo-inverse\nPSNR: {psnr_pinv:.2f} dB", fontsize=11)
    plt.axis("off")

    # Row 2: Reconstructions
    ax5 = plt.subplot(3, 4, 5)
    plt.imshow(to_numpy(distributed_recon), cmap="gray")
    dist_psnr = psnr_distributed[-1]
    plt.title(
        f"Distributed PnP\nPSNR: {dist_psnr:.2f} dB", fontsize=11, fontweight="bold"
    )
    plt.axis("off")

    ax6 = plt.subplot(3, 4, 6)
    plt.imshow(to_numpy(single_recon), cmap="gray")
    single_psnr = psnr_single[-1]
    plt.title(f"Single-process PnP\nPSNR: {single_psnr:.2f} dB", fontsize=11)
    plt.axis("off")

    # Difference map
    ax7 = plt.subplot(3, 4, 7)
    diff = torch.abs(distributed_recon - single_recon)
    diff_np = to_numpy(diff)
    if diff_np.ndim == 3:
        diff_np = diff_np.mean(axis=2)
    plt.imshow(diff_np, cmap="hot")
    plt.title(f"Difference Map\nMax: {diff.max():.4f}", fontsize=11)
    plt.axis("off")
    plt.colorbar(shrink=0.8)

    # Crop comparison
    ax8 = plt.subplot(3, 4, 8)
    crop_size = 64
    H, W = clean_image.shape[-2:]
    crop_y, crop_x = H // 2 - crop_size // 2, W // 2 - crop_size // 2
    crop_slice = (slice(crop_y, crop_y + crop_size), slice(crop_x, crop_x + crop_size))

    clean_crop = clean_image[..., crop_slice[0], crop_slice[1]]
    dist_crop = distributed_recon[..., crop_slice[0], crop_slice[1]]

    # Convert to grayscale if RGB
    if clean_crop.dim() == 3 and clean_crop.shape[0] == 3:
        clean_crop = clean_crop.mean(0)
        dist_crop = dist_crop.mean(0)

    comparison = torch.cat([clean_crop, dist_crop], dim=-1)
    plt.imshow(to_numpy(comparison), cmap="gray")
    plt.title("GT | Distributed\n(Crop)", fontsize=11)
    plt.axis("off")

    # Row 3: Analysis plots
    ax9 = plt.subplot(3, 2, 5)
    iterations = range(len(psnr_distributed))
    plt.plot(
        iterations,
        psnr_distributed,
        "b-",
        linewidth=2,
        label=f"Distributed (World Size: {ctx.world_size})",
    )
    plt.plot(iterations, psnr_single, "r--", linewidth=2, label="Single-process")
    plt.axhline(
        y=psnr_pinv, color="g", linestyle=":", linewidth=2, label="Pseudo-inverse"
    )
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("PSNR (dB)", fontsize=12)
    plt.title("PSNR Evolution", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Performance comparison
    ax10 = plt.subplot(3, 2, 6)
    methods = ["Pseudo-inv", "Single-proc", "Distributed"]
    psnrs = [psnr_pinv, single_psnr, dist_psnr]
    colors = ["green", "red", "blue"]

    bars = plt.bar(methods, psnrs, color=colors, alpha=0.7)
    plt.ylabel("PSNR (dB)", fontsize=12)
    plt.title("Final PSNR Comparison", fontsize=12, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, psnr in zip(bars, psnrs):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{psnr:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    save_path = Path(__file__).parent / "distributed_pnp_tutorial_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# %%
# Run the Example
# ===============

if __name__ == "__main__":
    run_distributed_pnp_example()

# %%
# 🎉 Congratulations!
# ====================
#
# You've successfully learned how to use DeepInverse's distributed framework!
#
# **What we covered:**
#
# 1. **DistributedContext**: Process group management and device selection
# 2. **DistributedLinearPhysics**: Distributing forward operators across processes
# 3. **DistributedMeasurements**: Handling distributed measurement data
# 4. **DistributedSignal**: Synchronized signal representation
# 5. **DistributedDataFidelity**: Distributed gradient computation
# 6. **DistributedPrior**: Spatial tiling for distributed denoising
#
# **Key Benefits:**
#
# - **Scalability**: Algorithms scale across multiple processes/GPUs
# - **Consistency**: Results match single-process implementations
# - **Flexibility**: Easy to adapt existing algorithms
# - **Efficiency**: Reduced memory usage per process through smart sharding
#
# **What's Next?**
#
# - Adapt your own inverse problem algorithms to the distributed framework
# - Scale to larger problems and more processes for real speedups
