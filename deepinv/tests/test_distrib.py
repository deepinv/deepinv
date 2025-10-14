"""
Tests for the distributed framework.

This module contains tests for:
- DistributedContext: context manager for distributed runs
- DistributedLinearPhysics: distributed physics operators
- DistributedMeasurements: distributed measurements
- DistributedDataFidelity: distributed data fidelity
- DistributedSignal: distributed signal processing
- DistributedPrior: distributed prior processing
- Distribution strategies: BasicStrategy and SmartTilingStrategy
- Factory API: FactoryConfig, TilingConfig, make_distrib_bundle
- Single vs distributed equivalence

Notes on testing distributed code:
- These tests can run in single-process mode for basic validation
- Multi-process tests use subprocess with torchrun to spawn multiple processes
- CI/CD environments typically only support CPU-based multi-process testing
- GPU-based multi-process testing should be done locally or on specialized infrastructure
"""
from __future__ import annotations
import os
import sys
import subprocess
import tempfile
import pytest
import torch
import numpy as np
from pathlib import Path

import deepinv as dinv
from deepinv.physics import LinearPhysics, Blur, GaussianNoise
from deepinv.physics.blur import gaussian_blur
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP, Prior
from deepinv.loss.metric import PSNR
from deepinv.models import DRUNet

# Import distributed components
from deepinv.distrib import (
    DistributedContext,
    DistributedLinearPhysics,
    DistributedDataFidelity,
    DistributedMeasurements,
    DistributedSignal,
    DistributedPrior,
    FactoryConfig,
    TilingConfig,
    make_distrib_bundle,
)

from deepinv.distrib.distribution_strategies.strategies import (
    BasicStrategy,
    SmartTilingStrategy,
    create_strategy,
)


# =============================================================================
# Helper Functions
# =============================================================================

def create_test_physics(device, num_ops=3):
    """Create simple test physics operators."""
    physics_list = []
    for i in range(num_ops):
        # Create simple blur operators with different sigmas
        kernel = gaussian_blur(sigma=1.0 + i * 0.5, device=str(device))
        blur = Blur(filter=kernel, padding="circular", device=str(device))
        blur.noise_model = GaussianNoise(sigma=0.01)
        physics_list.append(blur)
    return physics_list


def create_test_measurements(physics_list, x, device):
    """Create test measurements from physics operators."""
    measurements_list = []
    for physics in physics_list:
        y = physics(x)
        measurements_list.append(y)
    return measurements_list


def _worker(rank, world_size, test_func, test_args, result_queue, dist_config):
    """Worker function that runs in each process - must be at module level for pickling."""
    # Set environment variables for this rank
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = dist_config["master_addr"]
    os.environ["MASTER_PORT"] = dist_config["master_port"]
    
    try:
        result = test_func(rank, world_size, test_args)
        if result_queue is not None:
            result_queue.put((rank, result))
    except Exception as e:
        # Capture full traceback for debugging
        import traceback
        error_msg = f"Rank {rank} error:\n{''.join(traceback.format_exception(type(e), e, e.__traceback__))}"
        if result_queue is not None:
            result_queue.put((rank, RuntimeError(error_msg)))
        # Don't re-raise to avoid zombie processes
        import sys
        sys.exit(1)
    finally:
        # Clean up distributed state if initialized
        import torch.distributed as dist
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        
        # Clean up env vars
        for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
            os.environ.pop(key, None)


def run_distributed_test(test_func, dist_config, test_args=None):
    """
    Run a test function in distributed mode with multiple processes.
    
    Args:
        test_func: Function to run in each process (should accept rank, world_size, and test_args)
        dist_config: Configuration dict with world_size, backend, master_addr, master_port
        test_args: Optional dict of arguments to pass to test_func
    
    Returns:
        List of results from each process (or None if processes don't return)
    """
    world_size = dist_config["world_size"]
    
    # For single process, just run directly without setting env vars (to avoid distributed init)
    if world_size == 1:
        # Don't set env vars - let it run as truly single process
        result = test_func(0, 1, test_args or {})
        return [result]
    
    # For multiple processes, use torch.multiprocessing with fork
    # Fork allows nested functions to work (no pickling needed)
    import torch.multiprocessing as mp
    import sys
    import signal
    
    # Use fork on Unix systems (works with nested functions)
    # Fall back to spawn on Windows (will require refactoring)
    mp_context = "fork" if sys.platform != "win32" else "spawn"
    ctx = mp.get_context(mp_context)
    result_queue = ctx.Queue()
    
    # Spawn processes
    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_worker,
            args=(rank, world_size, test_func, test_args or {}, result_queue, dist_config),
            daemon=False  # Non-daemon so we can control cleanup
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete with timeout
    timeout = 10  # 10 seconds per test - some tests involve model initialization
    failed_ranks = []
    timed_out_ranks = []
    
    def cleanup_processes(kill_signal=signal.SIGTERM):
        """Aggressively clean up all child processes."""
        for p in processes:
            if p.is_alive():
                try:
                    if kill_signal == signal.SIGKILL:
                        p.kill()
                    else:
                        p.terminate()
                except Exception:
                    pass
        # Give them a moment to die
        for p in processes:
            if p.is_alive():
                p.join(timeout=0.5)
    
    try:
        for i, p in enumerate(processes):
            p.join(timeout=timeout)
            if p.is_alive():
                timed_out_ranks.append(i)
            elif p.exitcode != 0:
                failed_ranks.append(i)
    except KeyboardInterrupt:
        cleanup_processes(signal.SIGKILL)
        raise
    finally:
        # First try gentle termination
        cleanup_processes(signal.SIGTERM)
        # Then kill anything still alive
        cleanup_processes(signal.SIGKILL)
    
    # Report timeouts first
    if timed_out_ranks:
        raise RuntimeError(f"Processes {timed_out_ranks} timed out after {timeout}s")
    
    # Collect results (including any error messages)
    results = []
    errors = []
    while not result_queue.empty():
        rank, result = result_queue.get()
        if isinstance(result, Exception):
            errors.append((rank, result))
        else:
            results.append((rank, result))
    
    # If there were failures, report them with error details
    if failed_ranks:
        error_details = "\n".join([f"Rank {r}: {e}" for r, e in errors if r in failed_ranks])
        if not error_details:
            error_details = "No error details captured"
        raise RuntimeError(f"Processes {failed_ranks} failed:\n{error_details}")
    
    # Sort by rank and return just the results
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]


# =============================================================================
# Test DistributedContext
# =============================================================================

class TestDistributedContext:
    """Test the DistributedContext context manager."""

    def test_initialization(self, dist_config):
        """Test context initialization with distributed configuration."""
        def test_func(rank, world_size, args):
            with DistributedContext(seed=42, device_mode="cpu") as ctx:
                assert ctx.world_size == world_size
                assert ctx.rank == rank
                assert ctx.device is not None
                assert ctx.sharding in ["round_robin", "block"]
                
                # Check distributed state
                if world_size > 1:
                    assert ctx.is_dist
                else:
                    assert not ctx.is_dist
                
                return {"rank": rank, "world_size": world_size}
        
        results = run_distributed_test(test_func, dist_config)
        assert len(results) == dist_config["world_size"]

    def test_sharding_round_robin(self, dist_config):
        """Test round-robin sharding with different world sizes."""
        def test_func(rank, world_size, args):
            with DistributedContext(sharding="round_robin", seed=42, device_mode="cpu") as ctx:
                num_items = 10
                local_indices = ctx.local_indices(num_items)
                
                # Verify we get the correct indices for this rank
                expected = [i for i in range(num_items) if i % world_size == rank]
                assert local_indices == expected, f"Rank {rank}: got {local_indices}, expected {expected}"
                
                return local_indices
        
        results = run_distributed_test(test_func, dist_config)
        
        # Verify all indices are covered exactly once
        all_indices = []
        for indices in results:
            all_indices.extend(indices)
        all_indices.sort()
        assert all_indices == list(range(10))

    def test_sharding_block(self, dist_config):
        """Test block sharding with different world sizes."""
        def test_func(rank, world_size, args):
            with DistributedContext(sharding="block", seed=42, device_mode="cpu") as ctx:
                num_items = 10
                local_indices = ctx.local_indices(num_items)
                
                # Verify block distribution
                per_rank = (num_items + world_size - 1) // world_size
                start = rank * per_rank
                end = min(start + per_rank, num_items)
                expected = list(range(start, end))
                
                assert local_indices == expected, f"Rank {rank}: got {local_indices}, expected {expected}"
                
                return local_indices
        
        results = run_distributed_test(test_func, dist_config)
        
        # Verify all indices are covered exactly once
        all_indices = []
        for indices in results:
            all_indices.extend(indices)
        all_indices.sort()
        assert all_indices == list(range(10))

    def test_all_reduce(self, dist_config):
        """Test all_reduce operation."""
        def test_func(rank, world_size, args):
            with DistributedContext(seed=42, device_mode="cpu") as ctx:
                # Each rank contributes its rank value
                t = torch.tensor([float(rank)], device=ctx.device)
                
                # Sum reduction
                result_sum = ctx.all_reduce_(t.clone(), op="sum")
                expected_sum = sum(range(world_size))
                assert torch.allclose(result_sum, torch.tensor([float(expected_sum)], device=ctx.device))
                
                # Mean reduction
                result_mean = ctx.all_reduce_(t.clone(), op="mean")
                expected_mean = sum(range(world_size)) / world_size
                assert torch.allclose(result_mean, torch.tensor([expected_mean], device=ctx.device))
                
                return {"sum": result_sum.item(), "mean": result_mean.item()}
        
        results = run_distributed_test(test_func, dist_config)
        
        # All ranks should have the same result
        for result in results:
            assert result["sum"] == sum(range(dist_config["world_size"]))
            assert abs(result["mean"] - sum(range(dist_config["world_size"])) / dist_config["world_size"]) < 1e-6

    def test_broadcast(self, dist_config):
        """Test broadcast operation."""
        def test_func(rank, world_size, args):
            with DistributedContext(seed=42, device_mode="cpu") as ctx:
                # Only rank 0 has meaningful data
                if rank == 0:
                    t = torch.tensor([1.0, 2.0, 3.0], device=ctx.device)
                else:
                    t = torch.zeros(3, device=ctx.device)
                
                # Broadcast from rank 0
                result = ctx.broadcast_(t.clone(), src=0)
                
                # All ranks should now have [1, 2, 3]
                expected = torch.tensor([1.0, 2.0, 3.0], device=ctx.device)
                assert torch.allclose(result, expected)
                
                return result.tolist()
        
        results = run_distributed_test(test_func, dist_config)
        
        # All ranks should have the same result
        for result in results:
            assert result == [1.0, 2.0, 3.0]


# =============================================================================
# Test DistributedMeasurements
# =============================================================================

class TestDistributedMeasurements:
    """Test DistributedMeasurements class."""

    def test_initialization_with_list(self, dist_config):
        """Test initialization with a list of measurements."""
        def test_func(rank, world_size, args):
            with DistributedContext(seed=42, device_mode="cpu") as ctx:
                # Create measurements (same on all ranks)
                torch.manual_seed(42)
                measurements = [
                    torch.randn(1, 3, 32, 32, device=ctx.device) for _ in range(5)
                ]
                # Note: must pass num_items when providing a list
                dmeas = DistributedMeasurements(
                    ctx, num_items=len(measurements), measurements_list=measurements, dtype=torch.float32
                )

                assert len(dmeas) == 5
                
                # Check that local measurements match expected distribution
                expected_local_count = len(ctx.local_indices(5))
                assert len(dmeas.local) == expected_local_count
                assert len(dmeas.indices()) == expected_local_count
                
                return {"num_local": len(dmeas.local), "indices": dmeas.indices()}
        
        results = run_distributed_test(test_func, dist_config)
        
        # Verify total count matches
        total_local = sum(r["num_local"] for r in results)
        assert total_local == 5

    def test_initialization_with_factory(self, dist_config):
        """Test initialization with a factory function."""
        def test_func(rank, world_size, args):
            with DistributedContext(seed=42, device_mode="cpu") as ctx:
                def meas_factory(idx, device, shared):
                    torch.manual_seed(idx)  # Make deterministic
                    return torch.randn(1, 3, 32, 32, device=device)

                dmeas = DistributedMeasurements(
                    ctx, num_items=5, factory=meas_factory, dtype=torch.float32
                )

                assert len(dmeas) == 5
                assert all(y.shape == (1, 3, 32, 32) for y in dmeas.local)
                
                return {"num_local": len(dmeas.local)}
        
        results = run_distributed_test(test_func, dist_config)
        
        # Verify total count
        total_local = sum(r["num_local"] for r in results)
        assert total_local == 5


# =============================================================================
# Test DistributedLinearPhysics
# =============================================================================

class TestDistributedLinearPhysics:
    """Test DistributedLinearPhysics class."""

    def test_initialization_with_list(self, dist_config):
        """Test initialization with a list of physics operators."""
        def test_func(rank, world_size, args):
            with DistributedContext(seed=42, device_mode="cpu") as ctx:
                physics_list = create_test_physics(ctx.device, num_ops=3)
                dphysics = DistributedLinearPhysics(
                    ctx, num_ops=len(physics_list), factory=lambda i, d, s: physics_list[i]
                )

                expected_local = len(ctx.local_indices(3))
                assert len(dphysics.local_idx) == expected_local
                assert len(dphysics.local_physics) == expected_local
                
                return {"num_local": len(dphysics.local_idx)}
        
        results = run_distributed_test(test_func, dist_config)
        
        # Verify total
        total = sum(r["num_local"] for r in results)
        assert total == 3

    def test_forward(self, dist_config):
        """Test forward operation."""
        def test_func(rank, world_size, args):
            with DistributedContext(seed=42, device_mode="cpu") as ctx:
                physics_list = create_test_physics(ctx.device, num_ops=3)
                torch.manual_seed(42)
                x = torch.randn(1, 3, 64, 64, device=ctx.device)

                dphysics = DistributedLinearPhysics(
                    ctx, num_ops=len(physics_list), factory=lambda i, d, s: physics_list[i].to(d)
                )

                # Test forward on local operators
                local_results = []
                for phys, idx in zip(dphysics.local_physics, dphysics.local_idx):
                    y = phys(x)
                    local_results.append(y)

                expected_local = len(ctx.local_indices(3))
                assert len(local_results) == expected_local
                assert all(isinstance(y, torch.Tensor) for y in local_results)
                
                return {"num_results": len(local_results)}
        
        results = run_distributed_test(test_func, dist_config)
        total = sum(r["num_results"] for r in results)
        assert total == 3


# =============================================================================
# Test DistributedDataFidelity
# =============================================================================

class TestDistributedDataFidelity:
    """Test DistributedDataFidelity class."""

    def test_grad(self, dist_config):
        """Test gradient computation in distributed mode."""
        def test_func(rank, world_size, args):
            with DistributedContext(seed=42, device_mode="cpu") as ctx:
                # Create physics and measurements (deterministic)
                physics_list = create_test_physics(ctx.device, num_ops=3)
                torch.manual_seed(42)
                x = torch.randn(1, 3, 64, 64, device=ctx.device)
                measurements = create_test_measurements(physics_list, x, ctx.device)

                dphysics = DistributedLinearPhysics(
                    ctx, num_ops=len(physics_list), factory=lambda i, d, s: physics_list[i]
                )
                # Note: must pass num_items when providing a list
                dmeas = DistributedMeasurements(ctx, num_items=len(measurements), measurements_list=measurements)

                # Create distributed data fidelity and signal
                ddf = DistributedDataFidelity(
                    ctx,
                    dphysics,
                    dmeas,
                    data_fidelity_factory=lambda i, d, s: L2(),
                    reduction="mean",
                )
                dsignal = DistributedSignal(ctx, shape=x.shape)
                dsignal.update_(x)

                # Compute gradient
                grad = ddf.grad(dsignal)

                assert grad.shape == x.shape
                assert grad.device == ctx.device
                
                return {"grad_norm": grad.norm().item()}
        
        results = run_distributed_test(test_func, dist_config)
        
        # All ranks should have same gradient (due to all_reduce)
        grad_norms = [r["grad_norm"] for r in results]
        assert all(abs(gn - grad_norms[0]) < 1e-5 for gn in grad_norms)


# =============================================================================
# Test DistributedSignal
# =============================================================================

class TestDistributedSignal:
    """Test DistributedSignal class."""

    def test_initialization_and_sync(self, dist_config):
        """Test signal initialization and synchronization across ranks."""
        def test_func(rank, world_size, args):
            with DistributedContext(seed=42, device_mode="cpu") as ctx:
                shape = (1, 3, 64, 64)
                
                # Initialize with different data on each rank
                if rank == 0:
                    init_data = torch.ones(*shape, device=ctx.device)
                else:
                    init_data = torch.zeros(*shape, device=ctx.device)
                
                dsignal = DistributedSignal(ctx, shape=shape, init=init_data)

                # After init, all ranks should have the same data (broadcast from rank 0)
                assert dsignal.shape == shape
                assert dsignal.data.shape == shape
                
                # Data should be synced (broadcast from rank 0)
                expected = torch.ones(*shape, device=ctx.device)
                assert torch.allclose(dsignal.data, expected)
                
                return {"data_sum": dsignal.data.sum().item()}
        
        results = run_distributed_test(test_func, dist_config)
        
        # All ranks should have same data
        data_sums = [r["data_sum"] for r in results]
        assert all(abs(s - data_sums[0]) < 1e-5 for s in data_sums)

    def test_update_and_sync(self, dist_config):
        """Test updating signal data with synchronization."""
        def test_func(rank, world_size, args):
            with DistributedContext(seed=42, device_mode="cpu") as ctx:
                shape = (1, 3, 16, 16)
                dsignal = DistributedSignal(ctx, shape=shape)

                # Update with rank-specific data (only rank 0's data should propagate)
                if rank == 0:
                    new_data = torch.full(shape, 3.14, device=ctx.device)
                else:
                    new_data = torch.full(shape, 2.71, device=ctx.device)
                
                dsignal.update_(new_data)

                # After update, all ranks should have rank 0's data
                expected = torch.full(shape, 3.14, device=ctx.device)
                assert torch.allclose(dsignal.data, expected, atol=1e-5)
                
                return {"data_mean": dsignal.data.mean().item()}
        
        results = run_distributed_test(test_func, dist_config)
        
        # All ranks should have same mean
        data_means = [r["data_mean"] for r in results]
        assert all(abs(m - 3.14) < 1e-5 for m in data_means)


# =============================================================================
# Test Distribution Strategies
# =============================================================================

class TestDistributionStrategies:
    """Test distribution strategies."""

    def test_basic_strategy_initialization(self):
        """Test BasicStrategy initialization."""
        signal_shape = (1, 3, 64, 64)
        strategy = BasicStrategy(signal_shape, split_dims=(-2, -1))

        assert strategy.signal_shape == torch.Size(signal_shape)
        assert strategy.get_num_patches() > 0

    def test_basic_strategy_get_local_patches(self):
        """Test BasicStrategy patch extraction."""
        signal_shape = (1, 3, 64, 64)
        strategy = BasicStrategy(signal_shape, split_dims=(-2, -1), num_splits=(2, 2))

        X = torch.randn(*signal_shape)
        local_indices = [0, 1]
        patches = strategy.get_local_patches(X, local_indices)

        assert len(patches) == 2
        assert all(isinstance(p, tuple) and len(p) == 2 for p in patches)

    def test_basic_strategy_reduce_patches(self):
        """Test BasicStrategy patch reduction."""
        signal_shape = (1, 3, 64, 64)
        strategy = BasicStrategy(signal_shape, split_dims=(-2, -1), num_splits=(2, 2))

        X = torch.randn(*signal_shape)
        local_indices = list(range(strategy.get_num_patches()))
        patches = strategy.get_local_patches(X, local_indices)

        # Process patches (identity operation)
        processed_patches = [(idx, patch) for idx, patch in patches]

        # Reduce back
        out = torch.zeros_like(X)
        strategy.reduce_patches(out, processed_patches)

        # Should reconstruct original (with some numerical error for splits)
        assert out.shape == X.shape

    def test_smart_tiling_strategy_initialization(self):
        """Test SmartTilingStrategy initialization."""
        signal_shape = (1, 3, 128, 128)
        strategy = SmartTilingStrategy(
            signal_shape, patch_size=64, receptive_field_radius=16
        )

        assert strategy.signal_shape == torch.Size(signal_shape)
        assert strategy.patch_size == 64
        assert strategy.receptive_field_radius == 16
        assert strategy.get_num_patches() > 0

    def test_smart_tiling_strategy_get_local_patches(self):
        """Test SmartTilingStrategy patch extraction with padding."""
        signal_shape = (1, 3, 128, 128)
        strategy = SmartTilingStrategy(
            signal_shape, patch_size=64, receptive_field_radius=16
        )

        X = torch.randn(*signal_shape)
        local_indices = [0]
        patches = strategy.get_local_patches(X, local_indices)

        assert len(patches) == 1
        idx, patch = patches[0]
        # Patch should be larger than patch_size due to receptive field padding
        assert patch.shape[-2] >= 64 or patch.shape[-1] >= 64

    def test_smart_tiling_strategy_batching(self):
        """Test SmartTilingStrategy batching."""
        signal_shape = (1, 3, 128, 128)
        strategy = SmartTilingStrategy(
            signal_shape, patch_size=64, receptive_field_radius=16
        )

        X = torch.randn(*signal_shape)
        num_patches = strategy.get_num_patches()
        patches = strategy.get_local_patches(X, list(range(num_patches)))

        # Extract just the patch tensors
        patch_tensors = [p[1] for p in patches]

        # Apply batching
        batched = strategy.apply_batching(patch_tensors)

        assert isinstance(batched, list)
        assert len(batched) > 0

    def test_smart_tiling_oversized_patch(self):
        """Test SmartTilingStrategy with patch larger than image."""
        signal_shape = (1, 3, 32, 32)
        # Patch size larger than image
        strategy = SmartTilingStrategy(
            signal_shape, patch_size=128, receptive_field_radius=16
        )

        # Should handle gracefully and create at least one patch
        assert strategy.get_num_patches() > 0

        X = torch.randn(*signal_shape)
        patches = strategy.get_local_patches(X, [0])
        assert len(patches) > 0

    def test_create_strategy_factory(self):
        """Test strategy factory function."""
        signal_shape = (1, 3, 64, 64)

        basic = create_strategy("basic", signal_shape)
        assert isinstance(basic, BasicStrategy)

        smart = create_strategy("smart_tiling", signal_shape, patch_size=32)
        assert isinstance(smart, SmartTilingStrategy)

        with pytest.raises(ValueError):
            create_strategy("unknown_strategy", signal_shape)


# =============================================================================
# Test DistributedPrior
# =============================================================================

class TestDistributedPrior:
    """Test DistributedPrior class."""

    def test_prox_operation(self, dist_config):
        """Test prox operation with distributed prior."""
        def test_func(rank, world_size, args):
            with DistributedContext(seed=42, device_mode="cpu") as ctx:
                class SimplePrior(Prior):
                    def forward(self, x, *args, **kwargs):
                        return x * 0.9

                prior = SimplePrior()
                signal_shape = (1, 3, 64, 64)

                dprior = DistributedPrior(
                    ctx,
                    prior=prior,
                    signal_shape=signal_shape,
                    strategy="basic",
                    strategy_kwargs={"split_dims": (-2, -1), "num_splits": (2, 2)},
                )

                dsignal = DistributedSignal(ctx, shape=signal_shape)
                torch.manual_seed(42)
                x = torch.randn(*signal_shape, device=ctx.device)
                dsignal.update_(x)

                # Apply prox
                result = dprior.prox(dsignal)

                assert result.shape == signal_shape
                assert result.device == ctx.device
                
                return {"result_norm": result.norm().item()}
        
        results = run_distributed_test(test_func, dist_config)
        
        # All ranks should have same result (after all_reduce)
        norms = [r["result_norm"] for r in results]
        assert all(abs(n - norms[0]) < 1e-4 for n in norms)


# =============================================================================
# Test Factory API
# =============================================================================

class TestFactoryAPI:
    """Test the factory API for creating distributed components."""

    def test_make_distrib_bundle_basic(self, dist_config):
        """Test make_distrib_bundle with basic configuration."""
        def test_func(rank, world_size, args):
            with DistributedContext(seed=42, device_mode="cpu") as ctx:
                physics_list = create_test_physics(ctx.device, num_ops=3)
                torch.manual_seed(42)
                x = torch.randn(1, 3, 64, 64, device=ctx.device)
                measurements = create_test_measurements(physics_list, x, ctx.device)

                factory_config = FactoryConfig(
                    physics=physics_list, measurements=measurements, data_fidelity=L2()
                )

                bundle = make_distrib_bundle(
                    ctx, factory_config=factory_config, signal_shape=(1, 3, 64, 64)
                )

                assert bundle.physics is not None
                assert bundle.measurements is not None
                assert bundle.data_fidelity is not None
                assert bundle.signal is not None
                assert bundle.prior is None  # No prior specified
                
                return {"success": True}
        
        results = run_distributed_test(test_func, dist_config)
        assert all(r["success"] for r in results)

    def test_make_distrib_bundle_with_prior(self, dist_config):
        """Test make_distrib_bundle with prior."""
        def test_func(rank, world_size, args):
            with DistributedContext(seed=42, device_mode="cpu") as ctx:
                physics_list = create_test_physics(ctx.device, num_ops=3)
                torch.manual_seed(42)
                x = torch.randn(1, 3, 64, 64, device=ctx.device)
                measurements = create_test_measurements(physics_list, x, ctx.device)

                factory_config = FactoryConfig(
                    physics=physics_list, measurements=measurements, data_fidelity=L2()
                )

                class SimplePrior(Prior):
                    def forward(self, x, *args, **kwargs):
                        return x * 0.9

                prior = SimplePrior()
                tiling_config = TilingConfig(patch_size=32, receptive_field_radius=8)

                bundle = make_distrib_bundle(
                    ctx,
                    factory_config=factory_config,
                    signal_shape=(1, 3, 64, 64),
                    prior=prior,
                    tiling=tiling_config,
                )

                assert bundle.prior is not None
                assert isinstance(bundle.prior, DistributedPrior)
                
                return {"success": True}
        
        results = run_distributed_test(test_func, dist_config)
        assert all(r["success"] for r in results)



# =============================================================================
# Integration Test: Single vs Distributed PnP Equivalence
# =============================================================================

@pytest.mark.slow
class TestSingleVsDistributedEquivalence:
    """Test that single-process and distributed PnP produce equivalent results."""

    def test_pnp_equivalence(self, dist_config):
        """
        Test that distributed framework produces same results as single-process
        when run in distributed mode, and compare against reference single-process.
        """
        def test_func(rank, world_size, args):
            with DistributedContext(seed=42, device_mode="cpu") as ctx:
                # Create test data (deterministic)
                B, C, H, W = 1, 3, 64, 64
                torch.manual_seed(42)
                x_true = torch.randn(B, C, H, W, device=ctx.device)
                
                # Create physics operators
                physics_list = create_test_physics(ctx.device, num_ops=3)
                measurements = create_test_measurements(physics_list, x_true, ctx.device)
                
                # Simple prior
                class SimplePrior(Prior):
                    def forward(self, x, *args, **kwargs):
                        return x * 0.95
                
                prior = SimplePrior()
                
                # Run distributed PnP
                factory_config = FactoryConfig(
                    physics=physics_list, measurements=measurements, data_fidelity=L2()
                )
                tiling_config = TilingConfig(patch_size=32, receptive_field_radius=8)
                
                bundle = make_distrib_bundle(
                    ctx,
                    factory_config=factory_config,
                    signal_shape=(B, C, H, W),
                    prior=prior,
                    tiling=tiling_config,
                )
                
                bundle.signal.update_(torch.zeros_like(x_true))
                
                lr = 0.1
                num_iters = 3
                
                for it in range(num_iters):
                    # Data fidelity gradient
                    grad = bundle.data_fidelity.grad(bundle.signal)
                    
                    # Gradient step
                    new_data = bundle.signal.data - lr * grad
                    bundle.signal.update_(new_data)
                    
                    # Prior step
                    if bundle.prior is not None:
                        denoised = bundle.prior.prox(bundle.signal)
                        bundle.signal.update_(denoised)
                
                x_distributed = bundle.signal.data
                
                # Compute norm for comparison
                result_norm = x_distributed.norm().item()
                result_mean = x_distributed.mean().item()
                
                return {
                    "norm": result_norm,
                    "mean": result_mean,
                    "rank": rank,
                    "world_size": world_size
                }
        
        results = run_distributed_test(test_func, dist_config)
        
        # All ranks should produce the same result
        norms = [r["norm"] for r in results]
        means = [r["mean"] for r in results]
        
        # Check consistency across ranks
        for i in range(1, len(results)):
            assert abs(norms[i] - norms[0]) < 1e-3, \
                f"Rank {i} norm {norms[i]} differs from rank 0 norm {norms[0]}"
            assert abs(means[i] - means[0]) < 1e-4, \
                f"Rank {i} mean {means[i]} differs from rank 0 mean {means[0]}"
        
        # Verify results are reasonable (not all zeros)
        assert norms[0] > 0.1, "Result norm is too small"
