#!/usr/bin/env python3
"""
Radio Interferometry using Distributed Framework

Usage:
    # Single process
    uv run python dev/example_radio_distrib_framework.py
    
    # Multi-process with torchrun
    uv run torchrun --nproc_per_node=2 dev/example_radio_distrib_framework.py
    uv run torchrun --nproc_per_node=4 dev/example_radio_distrib_framework.py

This script demonstrates the new distributed framework for radio interferometry
inverse problems using real data from dev/ directory.

Features:
- Uses real radio interferometry data (3c353_gdth.npy, uv_coordinates.npy, briggs_weight.npy)
- Splits the problem into multiple physics operators for distribution
- Automatic device detection and backend selection (gloo/nccl)
- Gradient descent optimization with numerical stability
- Consistent results across single and multi-process execution

The distributed framework provides:
- DistributedContext: Handles process group initialization and device management
- DistributedLinearPhysics: Distributes physics operators across processes
- DistributedMeasurements: Distributes measurement data
- DistributedSignal: Manages synchronized reconstruction parameter
- DistributedDataFidelity: Computes distributed loss and gradients
"""

import os
import numpy as np
import torch
from pathlib import Path

# Import the radio interferometry physics
from deepinv.physics.radio import RadioInterferometry
from deepinv.optim.data_fidelity import L2

# Import the new distributed framework
from deepinv.distrib.distrib_framework import (
    DistributedContext,
    DistributedLinearPhysics,
    DistributedDataFidelity,
    DistributedMeasurements,
    DistributedSignal,
)


def load_real_radio_data(data_dir, ctx):
    """
    Load real radio interferometry data and split into multiple operators.
    
    Args:
        data_dir: Directory containing data files
        ctx: DistributedContext for device information
        
    Returns:
        ground_truth, physics_list, measurements_list
    """
    data_path = Path(data_dir)
    
    # Required files
    uv_file = data_path / "uv_coordinates.npy"
    gt_file = data_path / "3c353_gdth.npy"
    weights_file = data_path / "briggs_weight.npy"
    
    if not all(f.exists() for f in [uv_file, gt_file, weights_file]):
        raise FileNotFoundError(f"Missing data files in {data_dir}")
    
    if ctx.rank == 0:
        print(f"Loading real radio interferometry data from {data_dir}")
    
    # Load ground truth image
    image_gdth = np.load(gt_file, allow_pickle=True)
    x_true = torch.from_numpy(image_gdth).unsqueeze(0).unsqueeze(0).to(ctx.device)
    img_shape = x_true.shape[-2:]
    
    # Load UV coordinates
    uv = np.load(uv_file, allow_pickle=True)
    uv = torch.from_numpy(uv).to(ctx.device).float()
    # Ensure UV coordinates are (2, N) format
    if uv.shape[1] == 2:
        uv = uv.transpose(0, 1)
    
    # Load Briggs weights
    briggs_weight = np.load(weights_file, allow_pickle=True)
    briggs_weight = torch.from_numpy(briggs_weight).to(ctx.device).float().squeeze()
    
    # Noise parameters from the original demo
    tau = 0.5976 * 2e-3
    
    # Create the full physics operator to generate measurements
    if ctx.rank == 0:
        print(f"Creating full physics operator with {uv.shape[1]} measurements")
    
    full_physics = RadioInterferometry(
        img_size=img_shape,
        samples_loc=uv,
        real_projection=True,
        device=ctx.device,
    )
    
    # Generate measurements with noise (following the original demo)
    torch.manual_seed(42)  # For reproducibility
    y_full = full_physics.A(x_true)
    noise = (torch.randn_like(y_full) + 1j * torch.randn_like(y_full)) / np.sqrt(2)
    y_full = y_full + tau * noise.to(y_full.dtype)
    
    # Apply weighting (following the original demo)
    y_full *= briggs_weight / tau
    weights_full = (briggs_weight / tau).to(y_full.dtype)
    
    # Determine number of operators to create (more operators for better distribution)
    num_operators = max(4, ctx.world_size * 2)
    total_measurements = uv.shape[1]
    measurements_per_operator = total_measurements // num_operators
    
    if ctx.rank == 0:
        print(f"Splitting {total_measurements} measurements into {num_operators} operators")
        print(f"Approximately {measurements_per_operator} measurements per operator")
    
    physics_list = []
    measurements_list = []
    
    # Split the data into multiple operators
    for i in range(num_operators):
        start_idx = i * measurements_per_operator
        if i == num_operators - 1:
            # Last operator gets remaining measurements
            end_idx = total_measurements
        else:
            end_idx = (i + 1) * measurements_per_operator
        
        # Extract subset of UV coordinates and measurements
        uv_subset = uv[:, start_idx:end_idx]
        y_subset = y_full[:, :, start_idx:end_idx]
        weights_subset = weights_full[start_idx:end_idx]
        
        # Create physics operator for this subset
        physics = RadioInterferometry(
            img_size=img_shape,
            samples_loc=uv_subset,
            real_projection=True,
            device=ctx.device,
        )
        
        # Set weights for this subset
        physics.setWeight(weights_subset)
        
        physics_list.append(physics)
        measurements_list.append(y_subset)
        
        if ctx.rank == 0 and i < 3:  # Print info for first few operators
            print(f"  Operator {i+1}: {uv_subset.shape[1]} measurements")
    
    if ctx.rank == 0:
        print(f"Created {len(physics_list)} physics operators")
        print(f"Ground truth image shape: {x_true.shape}")
    
    return x_true, physics_list, measurements_list


def main():
    """Main function demonstrating radio interferometry with distributed framework."""
    
    # Configuration
    data_dir = "./dev"  # Directory containing real data files
    lr = 1e-6          # Learning rate (much smaller for stability)
    T = 50             # Number of iterations
    
    # Initialize distributed context
    with DistributedContext(sharding="round_robin", seed=42) as ctx:
        if ctx.rank == 0:
            print(f"Running on {ctx.world_size} process(es)")
            print(f"Device: {ctx.device}")
            print(f"Distributed: {ctx.is_dist}")
        
        # Load real radio interferometry data
        try:
            x_true, physics_list, measurements_list = load_real_radio_data(data_dir, ctx)
        except FileNotFoundError as e:
            if ctx.rank == 0:
                print(f"Error: {e}")
                print("Make sure the data files are in the dev/ directory:")
                print("  - 3c353_gdth.npy (ground truth)")
                print("  - uv_coordinates.npy (UV coordinates)")
                print("  - briggs_weight.npy (Briggs weights)")
            return
        
        # Extract image shape for signal initialization
        B, C, H, W = x_true.shape
        num_operators = len(physics_list)
        
        # Factory functions for the distributed framework
        def factory_physics(idx, device, shared):
            """Factory to create physics operator for index idx."""
            return physics_list[idx].to(device)
        
        def factory_data_fidelity(idx, device, shared):
            """Factory to create data fidelity for index idx."""
            return L2().to(device)
        
        def factory_measurements(idx, device, shared):
            """Factory to load measurements for index idx."""
            y = measurements_list[idx].to(device)
            return y
        
        # Build distributed components
        if ctx.rank == 0:
            print(f"Building distributed components for {num_operators} operators...")
        
        # Build distributed physics (local shards only)
        physics = DistributedLinearPhysics(
            ctx, num_ops=num_operators, factory=factory_physics,
        )
        
        # Build distributed measurements (local shards only)
        measurements = DistributedMeasurements(
            ctx, num_items=num_operators, factory=factory_measurements
        )
        
        # Build replicated signal - initialize with noisy version of ground truth
        signal = DistributedSignal(ctx, shape=(B, C, H, W))
        
        # Build distributed data fidelity
        df = DistributedDataFidelity(
            ctx, physics, measurements, 
            data_fidelity_factory=factory_data_fidelity, 
            reduction="sum"
        )

        if ctx.rank == 0:
            print(f"Local operators per rank: {len(physics.local_physics)}")
            print(f"Total operators: {num_operators}")
            print(f"Distributed indices: {physics.local_idx}")
            print(f"Starting gradient descent optimization...")
        
        # Simple gradient descent optimization
        initial_loss = None
        for it in range(T):
            # Compute loss and gradient
            loss = df.fn(signal)
            g = df.grad(signal)
            
            # Store initial loss for comparison
            if it == 0:
                initial_loss = loss.item()
            
            signal.data = signal.data - lr * g

            # Progress reporting
            if ctx.rank == 0 and (it % 10 == 0 or it < 5):
                print(f"Iteration {it:2d}, Loss: {loss.item():.6f}")
        
        # Final results
        if ctx.rank == 0:
            final_loss = loss.item()
            improvement = (initial_loss - final_loss) / initial_loss * 100
            print(f"\nOptimization completed!")
            print(f"Initial loss: {initial_loss:.6f}")
            print(f"Final loss:   {final_loss:.6f}")
            print(f"Improvement:  {improvement:.2f}%")
            
            # Compute reconstruction error
            recon_error = torch.norm(signal.data - x_true) / torch.norm(x_true)
            print(f"Reconstruction error (relative): {recon_error.item():.6f}")
            
            print("Radio interferometry reconstruction completed successfully!")


if __name__ == "__main__":
    main()
