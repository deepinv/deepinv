#!/usr/bin/env python3
"""
Distributed Plug-and-Play (PnP) Algorithm Example

Usage:
    # Single process
    uv run python dev/example_pnp_distrib.py
    
    # Multi-process with torchrun
    uv run torchrun --nproc_per_node=2 dev/example_pnp_distrib.py
    uv run torchrun --nproc_per_node=4 dev/example_pnp_distrib.py

This script demonstrates a distributed Plug-and-Play (PnP) algorithm that combines:
1. Distributed physics operators: Multiple complementary blur kernels distributed across processes
2. Distributed PnP algorithm: L2 data fidelity gradient step + distributed denoising prior step
3. Performance comparison: Distributed vs single-process execution with PSNR tracking

The example showcases:
- DistributedContext: Process group initialization and device management
- DistributedLinearPhysics: Multiple blur operators (Gaussian, motion, edge detection) distributed across processes
- DistributedDataFidelity: L2 data fidelity for gradient steps
- DistributedPrior: DRUNet-based denoising with 2D tiling strategy
- Performance tracking: PSNR over iterations, timing comparisons, and visualizations

The PnP algorithm alternates between:
1. Data fidelity gradient step: x = x - lr * ∇f(x) where f(x) = ||Ax - y||²
2. Denoising step: x = denoise(x) using distributed DRUNet prior

Results include PSNR plots, reconstruction comparisons, and timing analysis.
"""
import os
import time
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Compose

import torch
import deepinv as dinv
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
from deepinv.distrib.utils import tiling_splitting_strategy, tiling_reduce_fn


def create_blur_kernels(device: torch.device) -> List[torch.Tensor]:
    """
    Create different complementary blur kernels.
    
    Returns:
        List of blur kernels: [gaussian_blur, motion_blur, edge_detection, box_blur]
    """
    kernels = []
    
    # 1. Gaussian blur kernel (5x5)
    gaussian_kernel = torch.zeros((1, 1, 5, 5), device=device)
    center = 2
    sigma = 1.0
    for i in range(5):
        for j in range(5):
            x, y = i - center, j - center
            gaussian_kernel[0, 0, i, j] = torch.exp(torch.tensor(-(x**2 + y**2) / (2 * sigma**2), device=device))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    kernels.append(gaussian_kernel)
    
    # 2. Motion blur kernel (horizontal)
    motion_kernel = torch.zeros((1, 1, 5, 5), device=device)
    motion_kernel[0, 0, 2, :] = 1.0 / 5.0  # Horizontal line
    motion_kernel /= motion_kernel.sum()
    kernels.append(motion_kernel)
    
    # 3. Edge detection kernel (Laplacian)
    edge_kernel = torch.tensor([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernels.append(edge_kernel)
    
    # 4. Box blur kernel (3x3)
    box_kernel = torch.ones((1, 1, 3, 3), device=device) / 9.0
    box_kernel /= box_kernel.sum()
    kernels.append(box_kernel)
    
    return kernels


def load_urban100_image(device: torch.device, image_idx: int = 0) -> torch.Tensor:
    """Load an image from Urban100 dataset."""
    save_dir = Path(__file__).parent / "data" / "urban100"
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Define dataset
    dataset = dinv.datasets.Urban100HR(
        save_dir, download=True, transform=Compose([ToTensor()])
    )
    
    # Get specified image and resize to smaller size for memory efficiency
    image = dataset[image_idx].to(device)
    
    # Force smaller size for this demo
    target_size = 256  # Smaller fixed size
    image = torch.nn.functional.adaptive_avg_pool2d(
        image.unsqueeze(0), (target_size, target_size)
    ).squeeze(0)
    
    return image


def create_physics_and_measurements(
    image: torch.Tensor, 
    device: torch.device,
    noise_sigma: float = 0.05
) -> Tuple[List[Blur], List[torch.Tensor]]:
    """
    Create multiple blur physics operators and corresponding measurements.
    
    Args:
        image: Clean ground truth image
        device: Device to create operators on
        noise_sigma: Noise level for measurements
        
    Returns:
        physics_list: List of blur physics operators
        measurements_list: List of corresponding noisy measurements
    """
    blur_kernels = create_blur_kernels(device)
    physics_list = []
    measurements_list = []
    
    for i, kernel in enumerate(blur_kernels):
        # Create blur physics operator
        physics = Blur(filter=kernel, device=device)
        
        # Add noise model with different sigma for each operator
        physics.noise_model = GaussianNoise(sigma=noise_sigma * (0.5 + 0.5 * i))
        
        # Generate noisy measurement
        torch.manual_seed(42 + i)  # Different noise for each measurement
        measurement = physics(image.unsqueeze(0))
        
        physics_list.append(physics)
        measurements_list.append(measurement)
    
    return physics_list, measurements_list


def create_drunet_prior(device: torch.device) -> PnP:
    """Create a PnP prior using DRUNet denoiser."""
    drunet = dinv.models.DRUNet(pretrained="download").to(device)
    pnp_prior = PnP(denoiser=drunet)
    return pnp_prior


def compute_pseudo_inverse_baseline(
    physics_list: List[Blur], 
    measurements_list: List[torch.Tensor],
    signal_shape: Tuple[int, ...]
) -> torch.Tensor:
    """
    Compute a simple pseudo-inverse baseline reconstruction.
    
    This uses the adjoint of the first physics operator as a baseline.
    """
    device = measurements_list[0].device
    
    # Use first physics operator for pseudo-inverse
    physics = physics_list[0]
    measurement = measurements_list[0]
    
    # Simple adjoint reconstruction
    x_pinv = physics.A_adjoint(measurement)
    
    return x_pinv


def run_single_process_pnp(
    clean_image: torch.Tensor,
    physics_list: List[Blur],
    measurements_list: List[torch.Tensor],
    num_iterations: int = 20,
    lr: float = 0.01,
    sigma_denoiser: float = 0.05,
    device: torch.device = None
) -> Tuple[torch.Tensor, List[float]]:
    """
    Run PnP algorithm in single process mode for comparison.
    
    Returns:
        reconstruction: Final reconstructed image
        psnr_history: PSNR values over iterations
    """
    if device is None:
        device = clean_image.device
    
    # Initialize reconstruction - use same initialization as distributed version
    torch.manual_seed(42)  # Ensure reproducible initialization
    x = clean_image.unsqueeze(0) + 0.1 * torch.randn_like(clean_image.unsqueeze(0))
    
    # Create denoiser
    drunet = dinv.models.DRUNet(pretrained="download").to(device)
    pnp_prior = PnP(denoiser=drunet)
    
    # PSNR metric
    psnr_metric = PSNR()
    psnr_history = []
    
    # PnP iterations
    for it in range(num_iterations):
        # Data fidelity gradient step
        grad = torch.zeros_like(x)
        for physics, measurement in zip(physics_list, measurements_list):
            # Compute residual and gradient
            residual = physics.A(x) - measurement
            grad += physics.A_adjoint(residual)
        grad = grad / len(physics_list)  # average, not sum
        
        # Gradient step
        x = x - lr * grad
        
        # Denoising step
        with torch.no_grad():
            x = pnp_prior.prox(x, sigma_denoiser=sigma_denoiser)
        
        # Compute PSNR
        psnr_val = psnr_metric(x, clean_image.unsqueeze(0)).item()
        psnr_history.append(psnr_val)
    
    return x.squeeze(0), psnr_history


def main():
    """Main function demonstrating distributed PnP algorithm."""
    
    # Configuration
    num_iterations = 5  # Reduced for faster testing
    lr = 0.0001
    denoiser_sigma = 0.04
    noise_sigma = 0.1
    patch_size = 128  # Smaller patches for memory efficiency
    receptive_field_radius = 32  # Smaller receptive field
    
    print("="*80)
    print("Distributed Plug-and-Play (PnP) Algorithm Example")
    print("="*80)
    
    # Initialize distributed context
    with DistributedContext(sharding="round_robin", seed=42) as ctx:
        
        print(f"Running on {ctx.world_size} process(es)")
        print(f"Device: {ctx.device}")
        print(f"Distributed: {ctx.is_dist}")
        print(f"Rank: {ctx.rank}")
        
        # Load test image
        if ctx.rank == 0:
            print("\n1. Loading Urban100 dataset...")
        
        clean_image = load_urban100_image(ctx.device, image_idx=0)
        B, C, H, W = 1, *clean_image.shape
        
        if ctx.rank == 0:
            print(f"   Image shape: {clean_image.shape}")
        
        # Create physics operators and measurements
        if ctx.rank == 0:
            print("\n2. Creating complementary blur operators and measurements...")
        
        physics_list, measurements_list = create_physics_and_measurements(
            clean_image, ctx.device, noise_sigma=noise_sigma
        )
        num_operators = len(physics_list)
        
        if ctx.rank == 0:
            print(f"   Created {num_operators} blur operators:")
            blur_types = ["Gaussian", "Motion", "Edge Detection", "Box"]
            for i, blur_type in enumerate(blur_types[:num_operators]):
                print(f"     - Operator {i+1}: {blur_type} blur")
        
        # Factory functions for distributed framework
        def factory_physics(idx, device, shared):
            return physics_list[idx].to(device)
        
        def factory_data_fidelity(idx, device, shared):
            return L2().to(device)
        
        def factory_measurements(idx, device, shared):
            return measurements_list[idx].to(device)
        
        # Build distributed components
        if ctx.rank == 0:
            print(f"\n3. Building distributed components...")
        
        # Distributed physics operators
        distributed_physics = DistributedLinearPhysics(
            ctx, num_ops=num_operators, factory=factory_physics
        )
        
        # Distributed measurements
        distributed_measurements = DistributedMeasurements(
            ctx, num_items=num_operators, factory=factory_measurements
        )
        
        # Distributed signal
        distributed_signal = DistributedSignal(ctx, shape=(B, C, H, W))
        
        # Initialize with same noisy version as single process for fair comparison
        torch.manual_seed(42)  # Ensure reproducible initialization
        initial_noise = 0.1 * torch.randn_like(clean_image.unsqueeze(0))
        distributed_signal.update_(clean_image.unsqueeze(0) + initial_noise)
        
        # Distributed data fidelity
        distributed_df = DistributedDataFidelity(
            ctx, distributed_physics, distributed_measurements,
            data_fidelity_factory=factory_data_fidelity, reduction="mean"
        )
        
        # Distributed prior (denoising)
        if ctx.rank == 0:
            print(f"   Setting up distributed DRUNet prior with 2D tiling...")
            print(f"   Patch size: {patch_size}x{patch_size}")
            print(f"   Receptive field radius: {receptive_field_radius}")
        
        drunet_prior = create_drunet_prior(ctx.device)
        
        distributed_prior = DistributedPrior(
            ctx=ctx,
            prior=drunet_prior,
            splitting_strategy=tiling_splitting_strategy,
            signal_shape=(B, C, H, W),
            reduce_fn=tiling_reduce_fn,
            splitting_kwargs={
                "patch_size": patch_size,
                "receptive_field_radius": receptive_field_radius,
                "non_overlap": True,
            },
        )
        
        if ctx.rank == 0:
            print(f"   Local operators per rank: {len(distributed_physics.local_physics)}")
            print(f"   Local prior splits per rank: {len(distributed_prior.local_split_indices)}")
        
        # Run distributed PnP algorithm
        if ctx.rank == 0:
            print(f"\n4. Running distributed PnP algorithm ({num_iterations} iterations)...")

        psnr_metric = PSNR()
        psnr_history = []

        # Time the distributed computation
        start_time = time.time()

        for it in range(num_iterations):
            # Data fidelity gradient step
            grad = distributed_df.grad(distributed_signal)
            distributed_signal.data = distributed_signal.data - lr * grad

            # Denoising step using distributed prior
            distributed_signal.data = distributed_prior.prox(distributed_signal, sigma_denoiser=denoiser_sigma)

            # Compute PSNR on rank 0
            if ctx.rank == 0:
                current_psnr = psnr_metric(distributed_signal.data, clean_image.unsqueeze(0)).item()
                psnr_history.append(current_psnr)
                
                if it % 5 == 0 or it < 3:
                    print(f"   Iteration {it:2d}: PSNR = {current_psnr:.2f} dB")
        
        distributed_time = time.time() - start_time
        
        if ctx.rank == 0:
            print(f"   Distributed PnP completed in {distributed_time:.2f} seconds")
        
        # Run single-process comparison (only on rank 0)
        single_psnr_history = []
        single_time = 0
        
        if ctx.rank == 0:
            print(f"\n5. Running single-process PnP for comparison...")
            
            start_time = time.time()
            single_result, single_psnr_history = run_single_process_pnp(
                clean_image, physics_list, measurements_list, 
                num_iterations=num_iterations, lr=lr, sigma_denoiser=denoiser_sigma,
                device=ctx.device
            )
            single_time = time.time() - start_time
            
            print(f"   Single-process PnP completed in {single_time:.2f} seconds")
        
        # Compute baseline reconstruction (only on rank 0)
        if ctx.rank == 0:
            print(f"\n6. Computing pseudo-inverse baseline...")
            
            baseline_recon = compute_pseudo_inverse_baseline(
                physics_list, measurements_list, (B, C, H, W)
            )
            baseline_psnr = psnr_metric(baseline_recon, clean_image.unsqueeze(0)).item()
            
            print(f"   Pseudo-inverse PSNR: {baseline_psnr:.2f} dB")
        
        # Final results and visualization (only on rank 0)
        if ctx.rank == 0:
            print(f"\n7. Final Results:")
            print(f"   " + "="*50)
            
            distributed_final_psnr = psnr_history[-1] if psnr_history else 0
            single_final_psnr = single_psnr_history[-1] if single_psnr_history else 0
            
            print(f"   Baseline (Pseudo-inverse):  {baseline_psnr:.2f} dB")
            print(f"   Distributed PnP:           {distributed_final_psnr:.2f} dB")
            print(f"   Single-process PnP:        {single_final_psnr:.2f} dB")
            print(f"   PSNR difference:           {abs(distributed_final_psnr - single_final_psnr):.4f} dB")
            
            print(f"\n   Timing Comparison:")
            speedup = single_time / distributed_time if distributed_time > 0 else 1.0
            print(f"   Distributed time:          {distributed_time:.2f} seconds")
            print(f"   Single-process time:       {single_time:.2f} seconds")
            print(f"   Speedup factor:            {speedup:.2f}x")
            
            # Create visualizations
            print(f"\n8. Creating visualizations...")
            save_visualizations(
                clean_image=clean_image,
                measurements_list=measurements_list,
                baseline_recon=baseline_recon.squeeze(0),
                distributed_recon=distributed_signal.data.squeeze(0),
                single_recon=single_result,
                psnr_distributed=psnr_history,
                psnr_single=single_psnr_history,
                baseline_psnr=baseline_psnr,
                ctx=ctx
            )
            
            print(f"   Visualizations saved to 'distributed_pnp_results.png'")
            print(f"\nDistributed PnP example completed successfully!")


def save_visualizations(
    clean_image: torch.Tensor,
    measurements_list: List[torch.Tensor],
    baseline_recon: torch.Tensor,
    distributed_recon: torch.Tensor,
    single_recon: torch.Tensor,
    psnr_distributed: List[float],
    psnr_single: List[float],
    baseline_psnr: float,
    ctx
):
    """Save comprehensive visualizations of the results."""
    
    def to_numpy(tensor):
        """Convert tensor to numpy for visualization."""
        if tensor.dim() == 4:  # BCHW
            tensor = tensor.squeeze(0)  # Remove batch
        if tensor.dim() == 3 and tensor.shape[0] in [1, 3]:  # CHW
            tensor = tensor.permute(1, 2, 0)  # HWC
        return torch.clamp(tensor, 0, 1).cpu().detach().numpy()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Image reconstructions (top row)
    ax1 = plt.subplot(3, 4, 1)
    plt.imshow(to_numpy(clean_image), cmap='gray')
    plt.title("Ground Truth", fontsize=12, fontweight='bold')
    plt.axis('off')
    
    ax2 = plt.subplot(3, 4, 2)
    plt.imshow(to_numpy(measurements_list[0].squeeze(0)), cmap='gray')
    plt.title(f"Measurement 1\n(Gaussian Blur)", fontsize=12)
    plt.axis('off')
    
    ax3 = plt.subplot(3, 4, 3)
    plt.imshow(to_numpy(measurements_list[1].squeeze(0) if len(measurements_list) > 1 else measurements_list[0].squeeze(0)), cmap='gray')
    plt.title(f"Measurement 2\n(Motion Blur)", fontsize=12)
    plt.axis('off')
    
    ax4 = plt.subplot(3, 4, 4)
    plt.imshow(to_numpy(baseline_recon), cmap='gray')
    plt.title(f"Pseudo-inverse\nPSNR: {baseline_psnr:.2f} dB", fontsize=12)
    plt.axis('off')
    
    # Reconstructions comparison (middle row)
    ax5 = plt.subplot(3, 4, 5)
    plt.imshow(to_numpy(distributed_recon), cmap='gray')
    dist_psnr = psnr_distributed[-1] if psnr_distributed else 0
    plt.title(f"Distributed PnP\nPSNR: {dist_psnr:.2f} dB", fontsize=12, fontweight='bold')
    plt.axis('off')
    
    ax6 = plt.subplot(3, 4, 6)
    plt.imshow(to_numpy(single_recon), cmap='gray')
    single_psnr = psnr_single[-1] if psnr_single else 0
    plt.title(f"Single-process PnP\nPSNR: {single_psnr:.2f} dB", fontsize=12)
    plt.axis('off')
    
    # Difference map
    ax7 = plt.subplot(3, 4, 7)
    diff = torch.abs(distributed_recon - single_recon)
    diff_np = to_numpy(diff)
    if diff_np.ndim == 3:
        diff_np = diff_np.mean(axis=2)  # Convert to grayscale if needed
    plt.imshow(diff_np, cmap='hot')
    plt.title(f"Difference Map\nMax: {diff.max():.4f}", fontsize=12)
    plt.axis('off')
    plt.colorbar(shrink=0.8)
    
    # Close-up comparison
    ax8 = plt.subplot(3, 4, 8)
    crop_size = min(100, min(clean_image.shape[-2:]) // 2)
    H, W = clean_image.shape[-2:]
    crop_y, crop_x = H//2 - crop_size//2, W//2 - crop_size//2
    crop_slice_y = slice(crop_y, crop_y + crop_size)
    crop_slice_x = slice(crop_x, crop_x + crop_size)
    
    if clean_image.dim() == 3:  # CHW
        clean_crop = clean_image[:, crop_slice_y, crop_slice_x]
        dist_crop = distributed_recon[:, crop_slice_y, crop_slice_x]
    else:  # HW
        clean_crop = clean_image[crop_slice_y, crop_slice_x]
        dist_crop = distributed_recon[crop_slice_y, crop_slice_x]
    
    # Side-by-side comparison
    comparison = torch.cat([clean_crop, dist_crop], dim=-1)
    plt.imshow(to_numpy(comparison), cmap='gray')
    plt.title("GT | Distributed\n(Close-up)", fontsize=12)
    plt.axis('off')
    
    # PSNR evolution plot (bottom row, spanning 2 columns)
    ax9 = plt.subplot(3, 2, 5)
    iterations = range(len(psnr_distributed))
    plt.plot(iterations, psnr_distributed, 'b-', linewidth=2, label=f'Distributed (World Size: {ctx.world_size})')
    if psnr_single:
        plt.plot(iterations, psnr_single, 'r--', linewidth=2, label='Single Process')
    plt.axhline(y=baseline_psnr, color='g', linestyle=':', linewidth=2, label='Pseudo-inverse')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('PSNR Evolution During PnP Iterations', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Performance comparison chart (bottom right)
    ax10 = plt.subplot(3, 2, 6)
    methods = ['Pseudo-inv', 'Distributed', 'Single-proc']
    psnrs = [baseline_psnr, dist_psnr, single_psnr]
    colors = ['green', 'blue', 'red']
    
    bars = plt.bar(methods, psnrs, color=colors, alpha=0.7)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('Final PSNR Comparison', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, psnr in zip(bars, psnrs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{psnr:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    save_path = Path(__file__).parent / "distributed_pnp_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == "__main__":
    main()
