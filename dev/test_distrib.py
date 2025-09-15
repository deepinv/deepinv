#!/usr/bin/env python3
"""
Comprehensive test suite for the distributed framework.
This script tests both single and multi-process functionality.
"""

import torch
import sys
import os
from deepinv.physics import GaussianNoise
from deepinv.physics.blur import Blur
from deepinv.physics.inpainting import Inpainting
from deepinv.physics.forward import StackedLinearPhysics
from deepinv.optim.data_fidelity import L2, StackedPhysicsDataFidelity
from deepinv.utils.tensorlist import TensorList

from deepinv.distrib.distrib_framework import (
    DistributedContext,
    DistributedLinearPhysics,
    DistributedDataFidelity,
    DistributedMeasurements,
    DistributedSignal,
)


def create_test_setup(device):
    """Create a test setup with multiple physics operators."""
    img_size = (1, 1, 16, 16)
    
    # Create ground truth signal
    torch.manual_seed(42)
    true_signal = torch.zeros(img_size, device=device)
    true_signal[0, 0, 6:10, 6:10] = 1.0
    true_signal[0, 0, 7:9, 7:9] = 0.5
    
    # Create physics operators
    physics_list = []
    
    # Physics 1: Blur
    blur_kernel = torch.ones((1, 1, 3, 3), device=device) / 9.0
    physics1 = Blur(filter=blur_kernel, device=device)
    physics1.noise_model = GaussianNoise(sigma=0.01)
    physics_list.append(physics1)
    
    # Physics 2: Inpainting
    mask_shape = img_size[1:]
    mask = torch.ones(mask_shape, device=device, dtype=torch.bool)
    mask[0, ::2, :] = False
    physics2 = Inpainting(img_size=mask_shape, mask=mask, device=device)
    physics2.noise_model = GaussianNoise(sigma=0.01)
    physics_list.append(physics2)
    
    # Physics 3: Identity
    physics3 = Blur(filter=torch.tensor([[[[1.0]]]], device=device), device=device)
    physics3.noise_model = GaussianNoise(sigma=0.01)
    physics_list.append(physics3)
    
    # Physics 4: Another blur
    blur_kernel2 = torch.tensor([[[[0.25, 0.5, 0.25]]]], device=device)
    physics4 = Blur(filter=blur_kernel2, device=device)
    physics4.noise_model = GaussianNoise(sigma=0.01)
    physics_list.append(physics4)
    
    # Create measurements
    measurements_list = [p(true_signal) for p in physics_list]
    
    return true_signal, physics_list, measurements_list


def test_distributed_vs_stacked_consistency():
    """Test that distributed and stacked implementations are consistent."""
    print("\\n=== Testing Distributed vs Stacked Consistency ===")
    
    with DistributedContext() as ctx:
        device = ctx.device
        true_signal, physics_list, measurements_list = create_test_setup(device)
        
        # Stacked implementation
        stacked_physics = StackedLinearPhysics(physics_list)
        
        # Distributed implementation  
        def factory_physics(idx, device, shared):
            return physics_list[idx].to(device)
        
        distributed_physics = DistributedLinearPhysics(
            ctx, num_ops=len(physics_list), factory=factory_physics
        )
        
        # Test forward consistency
        y_stacked = stacked_physics.A(true_signal)
        y_distributed = distributed_physics.A(true_signal)
        
        max_diff = max(torch.norm(y_stacked[i] - y_distributed[i]).item() for i in range(len(y_stacked)))
        print(f"Forward max difference: {max_diff:.2e}")
        assert max_diff < 1e-6, f"Forward difference too large: {max_diff}"
        
        # Test adjoint consistency
        x_adj_stacked = stacked_physics.A_adjoint(y_stacked)
        x_adj_distributed = distributed_physics.A_adjoint(y_stacked)
        
        adj_diff = torch.norm(x_adj_stacked - x_adj_distributed).item()
        print(f"Adjoint difference: {adj_diff:.2e}")
        assert adj_diff < 1e-6, f"Adjoint difference too large: {adj_diff}"
        
        # Test VJP consistency
        v = y_stacked  # Use measurements as test vectors
        vjp_stacked = stacked_physics.A_vjp(true_signal, v)
        vjp_distributed = distributed_physics.A_vjp(true_signal, v)
        
        vjp_diff = torch.norm(vjp_stacked - vjp_distributed).item()
        print(f"VJP difference: {vjp_diff:.2e}")
        assert vjp_diff < 1e-6, f"VJP difference too large: {vjp_diff}"
        
        print("✓ Consistency test passed!")


def test_data_fidelity_consistency():
    """Test that DistributedDataFidelity is consistent with StackedPhysicsDataFidelity."""
    print("\\n=== Testing Data Fidelity Consistency ===")
    
    with DistributedContext() as ctx:
        device = ctx.device
        true_signal, physics_list, measurements_list = create_test_setup(device)
        
        # Create test signal (noisy version of true signal)
        test_signal = true_signal.clone() + 0.1 * torch.randn_like(true_signal)
        
        # Stacked implementation
        stacked_physics = StackedLinearPhysics(physics_list)
        stacked_measurements = TensorList([m.clone() for m in measurements_list])
        stacked_data_fidelity_list = [L2() for _ in physics_list]
        stacked_data_fidelity = StackedPhysicsDataFidelity(stacked_data_fidelity_list)
        
        # Distributed implementation
        def factory_physics(idx, device, shared):
            return physics_list[idx].to(device)
        
        def factory_data_fidelity(idx, device, shared):
            return L2().to(device)
        
        def read_measurement(idx, device, shared):
            return measurements_list[idx].to(device)
        
        distributed_physics = DistributedLinearPhysics(
            ctx, num_ops=len(physics_list), factory=factory_physics
        )
        
        distributed_measurements = DistributedMeasurements(
            ctx, num_items=len(measurements_list), factory=read_measurement
        )
        
        distributed_signal = DistributedSignal(ctx, shape=test_signal.shape)
        distributed_signal.update_(test_signal)
        
        distributed_data_fidelity = DistributedDataFidelity(
            ctx, distributed_physics, distributed_measurements, 
            data_fidelity_factory=factory_data_fidelity
        )
        
        # Test loss consistency
        stacked_loss = stacked_data_fidelity.fn(test_signal, stacked_measurements, stacked_physics)
        distributed_loss = distributed_data_fidelity.fn(distributed_signal)
        
        loss_diff = torch.abs(stacked_loss - distributed_loss).item()
        print(f"Loss difference: {loss_diff:.2e}")
        assert loss_diff < 1e-6, f"Loss difference too large: {loss_diff}"
        
        # Test gradient consistency
        stacked_grad = stacked_data_fidelity.grad(test_signal, stacked_measurements, stacked_physics)
        distributed_grad = distributed_data_fidelity.grad(distributed_signal)
        
        grad_diff = torch.norm(stacked_grad - distributed_grad).item()
        print(f"Gradient difference: {grad_diff:.2e}")
        assert grad_diff < 1e-6, f"Gradient difference too large: {grad_diff}"
        
        print("✓ Data fidelity consistency test passed!")


def test_gradient_descent_convergence():
    """Test that distributed gradient descent converges."""
    print("\\n=== Testing Gradient Descent Convergence ===")
    
    with DistributedContext() as ctx:
        device = ctx.device
        true_signal, physics_list, measurements_list = create_test_setup(device)
        
        def factory_physics(idx, device, shared):
            return physics_list[idx].to(device)
        
        def factory_data_fidelity(idx, device, shared):
            return L2().to(device)
        
        def read_measurement(idx, device, shared):
            return measurements_list[idx].to(device)
        
        # Create distributed components
        physics = DistributedLinearPhysics(
            ctx, num_ops=len(physics_list), factory=factory_physics
        )
        
        measurements = DistributedMeasurements(
            ctx, num_items=len(measurements_list), factory=read_measurement
        )
        
        signal = DistributedSignal(ctx, shape=true_signal.shape)
        signal.update_(true_signal.clone() + 0.2 * torch.randn_like(true_signal))
        
        df = DistributedDataFidelity(
            ctx, physics, measurements, data_fidelity_factory=factory_data_fidelity
        )
        
        # Run gradient descent
        lr = 0.01
        initial_loss = df.fn(signal).item()
        
        for _ in range(10):
            grad = df.grad(signal)
            signal.data = signal.data - lr * grad

        final_loss = df.fn(signal).item()
        reduction = (initial_loss - final_loss) / initial_loss * 100
        
        print(f"Initial loss: {initial_loss:.6f}")
        print(f"Final loss: {final_loss:.6f}")
        print(f"Loss reduction: {reduction:.2f}%")
        
        assert reduction > 5, f"Loss reduction too small: {reduction:.2f}%"
        print("✓ Convergence test passed!")


def test_distributed_communication():
    """Test distributed communication and sharding."""
    print("\\n=== Testing Distributed Communication ===")
    
    with DistributedContext(sharding="round_robin") as ctx:
        num_ops = 6
        local_indices = ctx.local_indices(num_ops)
        
        print(f"Rank {ctx.rank}/{ctx.world_size}")
        print(f"Local indices: {local_indices}")
        print(f"Expected distribution for round_robin: {[i for i in range(num_ops) if i % ctx.world_size == ctx.rank]}")
        
        assert local_indices == [i for i in range(num_ops) if i % ctx.world_size == ctx.rank], \
            "Round robin sharding not working correctly"
        
        # Test block sharding
        ctx.sharding = "block"
        block_indices = ctx.local_indices(num_ops)
        per_rank = (num_ops + ctx.world_size - 1) // ctx.world_size
        start = ctx.rank * per_rank
        end = min(start + per_rank, num_ops)
        expected_block = list(range(start, end))
        
        print(f"Block indices: {block_indices}")
        print(f"Expected block: {expected_block}")
        
        assert block_indices == expected_block, "Block sharding not working correctly"
        
        print("✓ Communication test passed!")


def main():
    """Run all tests."""
    print("=== Distributed Framework Test Suite ===")
    
    try:
        test_distributed_vs_stacked_consistency()
        test_data_fidelity_consistency()
        test_gradient_descent_convergence() 
        test_distributed_communication()
        
        print("\\n🎉 All tests passed!")
        return 0
        
    except Exception as e:
        print(f"\\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
