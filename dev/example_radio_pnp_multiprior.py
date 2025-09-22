#!/usr/bin/env python3
"""
Multi-Prior Distributed Radio Interferometry PnP Example

Usage:
    # Single process
    uv run python dev/example_radio_pnp_multiprior.py
    
    # Multi-process with torchrun
    uv run torchrun --nproc_per_node=2 dev/example_radio_pnp_multiprior.py
    uv run torchrun --nproc_per_node=4 dev/example_radio_pnp_multiprior.py

This script demonstrates distributed PnP for radio interferometry reconstruction
using multiple priors: TV, Wavelet, and DRUNet denoising with proper fallbacks.
"""
import os
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt

import torch
import deepinv as dinv
from deepinv.physics.radio import RadioInterferometry
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP, TVPrior
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


def create_multiprior(prior_type: str = "tv", device: torch.device = torch.device("cpu")):
    """
    Create different types of priors with fallback handling.
    """
    if prior_type == "tv":
        return TVPrior()
    
    elif prior_type == "wavelet":
        try:
            import pywt
            from deepinv.optim.prior import WaveletPrior
            return WaveletPrior(wv="db8", device=device)
        except ImportError:
            print("Warning: pywt not available, falling back to TV prior")
            return TVPrior()
        except Exception as e:
            print(f"Warning: WaveletPrior error ({e}), falling back to TV prior")
            return TVPrior()
    
    elif prior_type == "drunet":
        try:
            from deepinv.optim.prior import PnP
            model = dinv.models.DRUNet(in_channels=1, out_channels=1, pretrained="download").to(device)
            return PnP(denoiser=model)
        except Exception as e:
            print(f"Warning: DRUNet not available ({e}), falling back to TV prior")
            return TVPrior()
    
    else:
        return TVPrior()


def load_radio_data_multiprior(data_dir: str, ctx: DistributedContext) -> Tuple[torch.Tensor, List[RadioInterferometry], List[torch.Tensor]]:
    """Load radio interferometry data optimized for multi-prior processing."""
    data_path = Path(data_dir)
    
    # Required files
    uv_file = data_path / "uv_coordinates.npy"
    gt_file = data_path / "3c353_gdth.npy"
    weights_file = data_path / "briggs_weight.npy"
    
    if not all(f.exists() for f in [uv_file, gt_file, weights_file]):
        raise FileNotFoundError(f"Missing data files in {data_dir}")
    
    if ctx.rank == 0:
        print(f"Loading radio interferometry data from {data_dir}")
    
    # Load ground truth image
    image_gdth = np.load(gt_file, allow_pickle=True)
    if image_gdth.dtype == np.complex64 or image_gdth.dtype == np.complex128:
        image_gdth = np.abs(image_gdth)
    
    # Resize for memory efficiency - smaller for multi-prior testing
    if image_gdth.shape[0] > 128:
        import scipy.ndimage
        factor = image_gdth.shape[0] / 128
        image_gdth = scipy.ndimage.zoom(image_gdth, 1/factor, order=1)
    
    x_true = torch.from_numpy(image_gdth).float().unsqueeze(0).unsqueeze(0).to(ctx.device)
    img_shape = x_true.shape[-2:]
    
    if ctx.rank == 0:
        print(f"Ground truth shape: {x_true.shape}")
    
    # Load UV coordinates - fewer measurements for faster multi-prior testing
    uv = np.load(uv_file, allow_pickle=True)
    uv = torch.from_numpy(uv).to(ctx.device).float()
    if uv.shape[1] == 2:
        uv = uv.transpose(0, 1)
    
    # Use fewer measurements for multi-prior testing
    num_measurements = min(5000, uv.shape[1])
    indices = torch.randperm(uv.shape[1])[:num_measurements]
    uv = uv[:, indices]
    
    if ctx.rank == 0:
        print(f"Using {uv.shape[1]} measurements for multi-prior testing")
    
    # Load Briggs weights
    briggs_weight = np.load(weights_file, allow_pickle=True)
    briggs_weight = torch.from_numpy(briggs_weight).to(ctx.device).float().squeeze()
    briggs_weight = briggs_weight[indices]
    
    # Generate measurements
    full_physics = RadioInterferometry(
        img_size=img_shape,
        samples_loc=uv,
        real_projection=True,
        device=ctx.device,
    )
    
    tau = 0.5976 * 2e-3
    torch.manual_seed(42)
    y_full = full_physics.A(x_true)
    noise = (torch.randn_like(y_full) + 1j * torch.randn_like(y_full)) / np.sqrt(2)
    y_full = y_full + tau * noise.to(y_full.dtype)
    y_full *= briggs_weight / tau
    weights_full = briggs_weight / tau
    
    # Split into multiple operators
    num_operators = max(2, ctx.world_size)
    measurements_per_operator = num_measurements // num_operators
    
    if ctx.rank == 0:
        print(f"Splitting {num_measurements} measurements into {num_operators} operators")
    
    physics_list = []
    measurements_list = []
    
    for i in range(num_operators):
        start_idx = i * measurements_per_operator
        if i == num_operators - 1:
            end_idx = num_measurements
        else:
            end_idx = (i + 1) * measurements_per_operator
        
        # Extract subset
        uv_subset = uv[:, start_idx:end_idx]
        y_subset = y_full[:, :, start_idx:end_idx]
        weights_subset = weights_full[start_idx:end_idx]
        
        # Create physics operator
        physics = RadioInterferometry(
            img_size=img_shape,
            samples_loc=uv_subset,
            real_projection=True,
            device=ctx.device,
        )
        physics.setWeight(weights_subset)
        
        physics_list.append(physics)
        measurements_list.append(y_subset)
    
    return x_true, physics_list, measurements_list


def run_multiprior_pnp(
    ctx: DistributedContext,
    clean_image: torch.Tensor,
    physics_list: List[RadioInterferometry],
    measurements_list: List[torch.Tensor],
    prior_types: List[str] = ["tv", "wavelet"],
    num_iterations: int = 8,
    lr: float = 1e-6
) -> Dict[str, Tuple[torch.Tensor, List[float], float]]:
    """
    Run PnP with multiple priors and compare results.
    """
    results = {}
    
    for prior_type in prior_types:
        if ctx.rank == 0:
            print(f"\n--- Running PnP with {prior_type.upper()} prior ---")
        
        start_time = time.time()
        
        # Create prior
        prior = create_multiprior(prior_type, ctx.device)
        
        if ctx.is_dist:
            # Distributed version
            recon, psnr_history = run_distributed_pnp_with_prior(
                ctx, clean_image, physics_list, measurements_list, 
                prior, num_iterations, lr
            )
        else:
            # Single process version
            recon, psnr_history = run_single_pnp_with_prior(
                clean_image, physics_list, measurements_list,
                prior, num_iterations, lr
            )
        
        elapsed = time.time() - start_time
        
        if ctx.rank == 0:
            final_psnr = psnr_history[-1] if psnr_history else 0
            print(f"   {prior_type.upper()} prior: {final_psnr:.2f} dB in {elapsed:.2f}s")
        
        results[prior_type] = (recon, psnr_history, elapsed)
    
    return results


def run_distributed_pnp_with_prior(
    ctx: DistributedContext,
    clean_image: torch.Tensor,
    physics_list: List[RadioInterferometry],
    measurements_list: List[torch.Tensor],
    prior,
    num_iterations: int,
    lr: float
) -> Tuple[torch.Tensor, List[float]]:
    """Run distributed PnP with given prior."""
    B, C, H, W = clean_image.shape
    num_operators = len(physics_list)
    
    # Factory functions
    def factory_physics(idx, device, shared):
        return physics_list[idx].to(device)
    
    def factory_data_fidelity(idx, device, shared):
        return L2().to(device)
    
    def factory_measurements(idx, device, shared):
        return measurements_list[idx].to(device)
    
    # Build distributed components
    distributed_physics = DistributedLinearPhysics(
        ctx, num_ops=num_operators, factory=factory_physics
    )
    
    distributed_measurements = DistributedMeasurements(
        ctx, num_items=num_operators, factory=factory_measurements
    )
    
    distributed_signal = DistributedSignal(ctx, shape=(B, C, H, W))
    distributed_signal.update_(torch.zeros_like(clean_image))
    
    # Distributed data fidelity
    distributed_df = DistributedDataFidelity(
        ctx, distributed_physics, distributed_measurements,
        data_fidelity_factory=factory_data_fidelity, reduction="sum"
    )
    
    # Create distributed prior
    distributed_prior = DistributedPrior(
        ctx=ctx,
        prior=prior,
        splitting_strategy="tiling2d",
        signal_shape=(B, C, H, W),
        splitting_kwargs={
            "patch_size": 64,
            "receptive_field_radius": 8,
            "non_overlap": True,
        },
    )
    
    # PSNR tracking
    psnr_metric = PSNR()
    psnr_history = []
    
    # PnP iterations
    with torch.no_grad():
        for it in range(num_iterations):
            # Data fidelity gradient step
            grad = distributed_df.grad(distributed_signal)
            distributed_signal.data = distributed_signal.data - lr * grad
            
            # Prior step
            distributed_signal.data = distributed_prior.prox(distributed_signal, sigma_denoiser=0.1)
            
            # Compute PSNR on rank 0
            if ctx.rank == 0:
                current_psnr = psnr_metric(distributed_signal.data, clean_image).item()
                psnr_history.append(current_psnr)
    
    return distributed_signal.data, psnr_history


def run_single_pnp_with_prior(
    clean_image: torch.Tensor,
    physics_list: List[RadioInterferometry],
    measurements_list: List[torch.Tensor],
    prior,
    num_iterations: int,
    lr: float
) -> Tuple[torch.Tensor, List[float]]:
    """Run single process PnP with given prior."""
    # Initialize reconstruction
    x = torch.zeros_like(clean_image)
    
    # PSNR tracking
    psnr_metric = PSNR()
    psnr_history = []
    
    # PnP iterations
    for it in range(num_iterations):
        # Data fidelity gradient step
        grad = torch.zeros_like(x)
        for physics, measurement in zip(physics_list, measurements_list):
            forward = physics.A(x)
            residual = forward - measurement
            grad += physics.A_adjoint(residual)
        
        # Gradient step
        x = x - lr * grad
        
        # Prior step
        with torch.no_grad():
            x = prior.prox(x, gamma=0.1)
        
        # Compute PSNR
        psnr_val = psnr_metric(x, clean_image).item()
        psnr_history.append(psnr_val)
    
    return x, psnr_history


def save_multiprior_visualization(
    clean_image: torch.Tensor,
    results: Dict[str, Tuple[torch.Tensor, List[float], float]],
    baseline_psnr: float,
    ctx: DistributedContext
):
    """Save visualization comparing multiple priors."""
    if ctx.rank != 0:
        return
    
    def to_numpy(tensor):
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 3:
            tensor = tensor.squeeze(0)
        return torch.clamp(tensor, 0, 1).cpu().numpy()
    
    num_priors = len(results)
    fig, axes = plt.subplots(2, max(3, num_priors + 1), figsize=(4 * max(3, num_priors + 1), 8))
    
    # Ground truth
    axes[0, 0].imshow(to_numpy(clean_image), cmap='gray')
    axes[0, 0].set_title("Ground Truth")
    axes[0, 0].axis('off')
    
    # Reconstructions
    col = 1
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (prior_name, (recon, psnr_hist, elapsed)) in enumerate(results.items()):
        if col < axes.shape[1]:
            axes[0, col].imshow(to_numpy(recon), cmap='gray')
            final_psnr = psnr_hist[-1] if psnr_hist else 0
            axes[0, col].set_title(f"{prior_name.upper()} Prior\nPSNR: {final_psnr:.2f} dB")
            axes[0, col].axis('off')
            col += 1
    
    # Hide unused image subplots
    for j in range(col, axes.shape[1]):
        axes[0, j].axis('off')
    
    # PSNR evolution
    axes[1, 0].axhline(baseline_psnr, color='gray', linestyle=':', label='Baseline', linewidth=2)
    for i, (prior_name, (recon, psnr_hist, elapsed)) in enumerate(results.items()):
        color = colors[i % len(colors)]
        label = f"{prior_name.upper()} ({'Dist' if ctx.is_dist else 'Single'})"
        axes[1, 0].plot(psnr_hist, color=color, linewidth=2, label=label)
    
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('PSNR (dB)')
    axes[1, 0].set_title('PSNR Evolution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Final PSNR comparison
    prior_names = ['Baseline'] + [name.upper() for name in results.keys()]
    psnrs = [baseline_psnr] + [psnr_hist[-1] if psnr_hist else 0 for _, (_, psnr_hist, _) in results.items()]
    bar_colors = ['gray'] + colors[:len(results)]
    
    bars = axes[1, 1].bar(prior_names, psnrs, color=bar_colors, alpha=0.7)
    axes[1, 1].set_ylabel('PSNR (dB)')
    axes[1, 1].set_title('Final PSNR Comparison')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    for bar, psnr in zip(bars, psnrs):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{psnr:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Timing comparison
    if len(results) > 0:
        times = [elapsed for _, (_, _, elapsed) in results.items()]
        time_labels = [name.upper() for name in results.keys()]
        bars = axes[1, 2].bar(time_labels, times, color=colors[:len(results)], alpha=0.7)
        axes[1, 2].set_ylabel('Time (seconds)')
        axes[1, 2].set_title('Execution Time')
        axes[1, 2].grid(True, alpha=0.3, axis='y')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        for bar, t in zip(bars, times):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{t:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # Hide unused analysis subplots
    for j in range(3, axes.shape[1]):
        axes[1, j].axis('off')
    
    plt.tight_layout()
    save_path = Path(__file__).parent / "radio_pnp_multiprior.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved multi-prior visualization to {save_path}")


def main():
    """Main function demonstrating multi-prior distributed radio interferometry PnP."""
    
    print("="*80)
    print("Multi-Prior Distributed Radio Interferometry PnP Example")
    print("="*80)
    
    # Configuration
    data_dir = "./dev"
    num_iterations = 8
    lr = 1e-6
    
    # List of priors to test
    prior_types = ["drunet"]
    
    # Initialize distributed context
    with DistributedContext(sharding="round_robin", seed=42) as ctx:
        
        print(f"Running on {ctx.world_size} process(es)")
        print(f"Device: {ctx.device}")
        print(f"Distributed: {ctx.is_dist}")
        if ctx.is_dist:
            print(f"Rank: {ctx.rank}")
        
        try:
            # Load data
            if ctx.rank == 0:
                print(f"\n1. Loading radio interferometry data...")
            
            clean_image, physics_list, measurements_list = load_radio_data_multiprior(data_dir, ctx)

            if ctx.rank == 0:
                print(f"\n1. Loading radio interferometry data...")
            
            # Compute baseline
            if ctx.rank == 0:
                print(f"\n2. Computing pseudo-inverse baseline...")
            baseline_recon = physics_list[0].A_adjoint(measurements_list[0])
            
            if ctx.rank == 0:
                psnr_metric = PSNR()
                baseline_psnr = psnr_metric(baseline_recon, clean_image).item()
                print(f"   Pseudo-inverse PSNR: {baseline_psnr:.2f} dB")
            
            # Run multi-prior PnP
            if ctx.rank == 0:
                mode = "distributed" if ctx.is_dist else "single-process"
                print(f"\n3. Running {mode} PnP with multiple priors...")
                print(f"   Priors to test: {', '.join(p.upper() for p in prior_types)}")
            
            results = run_multiprior_pnp(
                ctx, clean_image, physics_list, measurements_list,
                prior_types, num_iterations, lr
            )
            
            # Summary and visualization
            if ctx.rank == 0:
                print(f"\n4. Multi-Prior Results Summary:")
                print(f"   " + "="*50)
                print(f"   Baseline PSNR:        {baseline_psnr:.2f} dB")
                
                for prior_name, (_, psnr_hist, elapsed) in results.items():
                    final_psnr = psnr_hist[-1] if psnr_hist else 0
                    improvement = final_psnr - baseline_psnr
                    print(f"   {prior_name.upper():8} PSNR:        {final_psnr:.2f} dB (+{improvement:.2f}) in {elapsed:.2f}s")
                
                print(f"\n5. Creating multi-prior visualization...")
                save_multiprior_visualization(clean_image, results, baseline_psnr, ctx)
                
                # Find best prior
                best_prior = max(results.items(), key=lambda x: x[1][1][-1] if x[1][1] else 0)
                best_psnr = best_prior[1][1][-1] if best_prior[1][1] else 0
                print(f"\n   Best performing prior: {best_prior[0].upper()} ({best_psnr:.2f} dB)")
                
                mode = "distributed" if ctx.is_dist else "single-process"
                print(f"\nMulti-prior {mode} radio interferometry PnP completed successfully!")
        
        except FileNotFoundError as e:
            if ctx.rank == 0:
                print(f"Error: {e}")
                print("Make sure the data files are in the dev/ directory")
        except Exception as e:
            if ctx.rank == 0:
                print(f"Unexpected error: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
