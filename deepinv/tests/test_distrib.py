"""
Tests for the simplified distributed API (distribute function).

This module tests the simplified distribute() API that makes it easy to convert
StackedPhysics, lists of physics, or factory functions into distributed physics operators.

Key test scenarios:
- distribute() with StackedPhysics
- distribute() with StackedLinearPhysics (automatic type detection)
- distribute() with list of Physics operators
- distribute() with list of LinearPhysics operators
- distribute() with callable factory
- Single-process and multi-process modes
- Forward operations (A)
- Adjoint operations (A_adjoint)
- A_adjoint_A (composition)
- All gather strategies (naive, concatenated, broadcast)
"""

from __future__ import annotations
import os
import pytest
import torch
import time
import torch.multiprocessing as mp

from deepinv.physics import Blur, GaussianNoise, LinearPhysics
from deepinv.physics.blur import gaussian_blur
from deepinv.physics.forward import StackedPhysics, StackedLinearPhysics
from deepinv.utils.tensorlist import TensorList

# Import distributed components
from deepinv.deepinv.distrib.distrib_framework import (
    DistributedContext,
    DistributedLinearPhysics,
)
from deepinv.distrib.distribute import distribute


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def num_ops():
    """Number of physics operators for tests."""
    return 3


@pytest.fixture
def test_input():
    """Test input tensor."""
    return torch.randn(1, 1, 16, 16)


@pytest.fixture(params=["naive", "concatenated", "broadcast"])
def gather_strategy(request):
    """Parameterized fixture for gather strategies."""
    return request.param


@pytest.fixture(params=["physics_list", "stacked_physics", "callable_factory"])
def physics_specification(request):
    """
    Parameterized fixture that returns different ways to specify physics operators.
    
    Returns a tuple: (spec_type, spec_object, num_ops_needed)
    - spec_type: "physics_list", "stacked_physics", or "callable_factory"
    - spec_object: the actual specification (list, StackedPhysics, or callable)
    - num_ops_needed: whether num_operators parameter is needed for distribute()
    """
    return request.param


def create_test_physics_list(device, num_ops=3):
    """Create simple test physics operators as a list."""
    physics_list = []
    for i in range(num_ops):
        # Create simple blur operators with different sigmas
        kernel = gaussian_blur(sigma=1.0 + i * 0.5, device=str(device))
        blur = Blur(filter=kernel, padding="circular", device=str(device))
        blur.noise_model = GaussianNoise(sigma=0.01)
        physics_list.append(blur)
    return physics_list


def create_physics_specification(spec_type, device, num_ops):
    """
    Create physics specification based on type.
    
    Args:
        spec_type: "physics_list", "stacked_physics", or "callable_factory"
        device: torch device
        num_ops: number of operators
        
    Returns:
        tuple: (physics_spec, needs_num_ops_param)
    """
    if spec_type == "physics_list":
        physics_list = create_test_physics_list(device, num_ops)
        return physics_list, False
    
    elif spec_type == "stacked_physics":
        physics_list = create_test_physics_list(device, num_ops)
        stacked = StackedLinearPhysics(physics_list)
        return stacked, False
    
    elif spec_type == "callable_factory":
        def physics_factory(idx, device, shared):
            kernel = gaussian_blur(sigma=1.0 + idx * 0.5, device=device)
            blur = Blur(filter=kernel, padding="circular", device=device)
            blur.noise_model = GaussianNoise(sigma=0.01)
            return blur
        return physics_factory, True
    
    else:
        raise ValueError(f"Unknown spec_type: {spec_type}")

# =============================================================================
# Helper Functions
# =============================================================================


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
    Run a test function across multiple ranks using torch.multiprocessing with 'spawn'.

    Args:
        test_func: picklable callable(rank: int, world_size: int, args: dict) -> Any
        dist_config: dict with at least {"world_size", "backend", "master_addr", "master_port"}
        test_args: optional dict passed to test_func on each rank

    Returns:
        List of per-rank results ordered by rank.

    Raises:
        RuntimeError if any rank fails, times out, or does not report a result.
    """

    world_size = int(dist_config["world_size"])
    args = test_args or {}

    # Single-process path: run directly without env-based init
    if world_size == 1:
        return [test_func(0, 1, args)]

    # Keep thread pools small to reduce fork/spawn overhead and flakiness
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")

    # Use 'spawn' for safety with PyTorch/NumPy
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    # Timeout policy
    per_rank_budget = 12.0  # seconds per rank
    timeout = max(20.0, per_rank_budget * world_size)

    processes = []
    start_time = time.monotonic()

    # Spawn worker processes
    for rank in range(world_size):
        p = ctx.Process(
            target=_worker,
            args=(rank, world_size, test_func, args, result_queue, dist_config),
            daemon=False,
        )
        p.start()
        processes.append(p)

    # Collect results/errors
    results_by_rank = {}
    errors_by_rank = {}
    received = 0

    try:
        # Poll the queue until we have results from all ranks or we time out
        while received < world_size:
            remaining = timeout - (time.monotonic() - start_time)
            if remaining <= 0:
                break

            # Use small blocking get to avoid busy waiting
            poll_window = min(0.25, remaining)
            try:
                rank, payload = result_queue.get(timeout=poll_window)
                received += 1
                if isinstance(payload, Exception):
                    errors_by_rank[rank] = payload
                else:
                    results_by_rank[rank] = payload
            except Exception:
                # queue.Empty or transient issues: just loop and re-check
                pass

    finally:
        # Wait for all processes to finish
        for p in processes:
            if p.is_alive():
                p.terminate()
            p.join(timeout=2.0)
            if p.is_alive():
                p.kill()
                p.join()

    # Check for errors
    if errors_by_rank:
        error_msgs = [f"{errors_by_rank[r]}" for r in sorted(errors_by_rank.keys())]
        raise RuntimeError(f"Distributed test failed:\n" + "\n".join(error_msgs))

    # Check we got all results
    if received < world_size:
        missing = [r for r in range(world_size) if r not in results_by_rank]
        raise RuntimeError(f"Timeout or missing results from ranks: {missing}")

    # Return results ordered by rank
    return [results_by_rank[r] for r in range(world_size)]


# =============================================================================
# Single-Process Tests
# =============================================================================


def test_distribute_single_process(physics_specification, gather_strategy, num_ops):
    """
    Test distribute() in single-process mode with different physics specifications
    and gather strategies.
    """
    with DistributedContext(device_mode="cpu") as ctx:
        # Create physics specification
        physics_spec, needs_num_ops = create_physics_specification(
            physics_specification, ctx.device, num_ops
        )
        
        # Distribute with gather strategy
        if needs_num_ops:
            distributed_physics = distribute(
                physics_spec, ctx,
                num_operators=num_ops,
                type_object="linear_physics",
                gather_strategy=gather_strategy
            )
        else:
            distributed_physics = distribute(
                physics_spec, ctx,
                gather_strategy=gather_strategy
            )
        
        # Check type
        assert isinstance(distributed_physics, DistributedLinearPhysics)
        assert distributed_physics.num_ops == num_ops
        assert distributed_physics.gather_strategy == gather_strategy
        
        # Create reference stacked physics for comparison
        reference_physics = StackedLinearPhysics(
            create_test_physics_list(ctx.device, num_ops)
        )
        
        # Test forward (A without noise)
        x = torch.randn(1, 1, 16, 16, device=ctx.device)
        y_distributed = distributed_physics.A(x)
        y_reference = reference_physics.A(x)
        
        assert len(y_distributed) == len(y_reference)
        for i in range(len(y_reference)):
            assert torch.allclose(y_distributed[i], y_reference[i], atol=1e-5), \
                f"Forward mismatch at index {i} with {physics_specification} and {gather_strategy}"
        
        # Test adjoint
        x_adj_distributed = distributed_physics.A_adjoint(y_distributed)
        x_adj_reference = reference_physics.A_adjoint(y_reference)
        
        assert torch.allclose(x_adj_distributed, x_adj_reference, atol=1e-5), \
            f"Adjoint mismatch with {physics_specification} and {gather_strategy}"
        
        # Test forward() method (with noise - just check structure)
        y_forward = distributed_physics.forward(x)
        assert len(y_forward) == num_ops
        for i in range(num_ops):
            assert y_forward[i].shape == y_reference[i].shape


# =============================================================================
# Multi-Process Tests
# =============================================================================


def _test_distributed_operation_worker(rank, world_size, args):
    """
    Generic worker function for multi-process tests.
    Tests both forward and adjoint operations.
    """
    with DistributedContext(device_mode="cpu") as ctx:
        # Create physics specification
        physics_spec, needs_num_ops = create_physics_specification(
            args["spec_type"], ctx.device, args["num_ops"]
        )
        
        # Distribute with gather strategy
        if needs_num_ops:
            distributed_physics = distribute(
                physics_spec, ctx,
                num_operators=args["num_ops"],
                type_object="linear_physics",
                gather_strategy=args["gather_strategy"]
            )
        else:
            distributed_physics = distribute(
                physics_spec, ctx,
                gather_strategy=args["gather_strategy"]
            )
        
        # Test forward operation
        x = args["x"].to(ctx.device)
        y_distributed = distributed_physics.A(x)
        
        # Each rank should get the full result
        assert len(y_distributed) == args["num_ops"]
        
        # Test adjoint operation if requested
        if args.get("test_adjoint", False):
            x_adj_distributed = distributed_physics.A_adjoint(y_distributed)
            
            # Verify on rank 0
            if rank == 0:
                # Create reference
                reference_physics = StackedLinearPhysics(
                    create_test_physics_list(ctx.device, args["num_ops"])
                )
                y_reference = reference_physics.A(x)
                x_adj_reference = reference_physics.A_adjoint(y_reference)
                
                max_diff = torch.max(torch.abs(x_adj_distributed - x_adj_reference)).item()
                assert max_diff < 1e-5, \
                    f"Adjoint mismatch with {args['spec_type']} and {args['gather_strategy']}: {max_diff}"
        
        # Verify forward on rank 0
        if rank == 0:
            reference_physics = StackedLinearPhysics(
                create_test_physics_list(ctx.device, args["num_ops"])
            )
            y_reference = reference_physics.A(x)
            
            for i in range(len(y_reference)):
                max_diff = torch.max(torch.abs(y_distributed[i] - y_reference[i])).item()
                assert max_diff < 1e-5, \
                    f"Forward mismatch at index {i} with {args['spec_type']} and {args['gather_strategy']}: {max_diff}"
        
        return "success"


def test_distribute_multiprocess_forward(dist_config, physics_specification, gather_strategy):
    """Test distribute() with multi-process forward operation."""
    # Prepare test data
    x = torch.randn(1, 1, 16, 16)
    test_args = {
        "num_ops": 4,
        "x": x,
        "spec_type": physics_specification,
        "gather_strategy": gather_strategy,
        "test_adjoint": False,
    }
    
    # Run distributed test
    results = run_distributed_test(_test_distributed_operation_worker, dist_config, test_args)
    
    # Check all ranks succeeded
    assert all(r == "success" for r in results)


def test_distribute_multiprocess_adjoint(dist_config, physics_specification, gather_strategy):
    """Test distribute() with multi-process adjoint operation."""
    # Prepare test data
    x = torch.randn(1, 1, 16, 16)
    test_args = {
        "num_ops": 4,
        "x": x,
        "spec_type": physics_specification,
        "gather_strategy": gather_strategy,
        "test_adjoint": True,
    }
    
    # Run distributed test
    results = run_distributed_test(_test_distributed_operation_worker, dist_config, test_args)
    
    # Check all ranks succeeded
    assert all(r == "success" for r in results)


# =============================================================================
# User Example Test
# =============================================================================


def _test_user_example_worker(rank, world_size, args):
    """Worker function testing the user's example code pattern."""
    import deepinv as dinv
    
    with DistributedContext(device_mode="cpu") as ctx:
        # User's example: create stacked physics
        physics1 = Blur(
            filter=gaussian_blur(sigma=1.0, device=str(ctx.device)),
            padding="circular",
            device=str(ctx.device)
        )
        physics2 = Blur(
            filter=gaussian_blur(sigma=1.5, device=str(ctx.device)),
            padding="circular",
            device=str(ctx.device)
        )
        physics3 = Blur(
            filter=gaussian_blur(sigma=2.0, device=str(ctx.device)),
            padding="circular",
            device=str(ctx.device)
        )
        
        # Stack physics (user's code)
        stacked_physics = StackedLinearPhysics([physics1, physics2, physics3])
        
        # Distribute with specified gather strategy (user's code)
        distributed_physics = distribute(
            stacked_physics, ctx,
            gather_strategy=args.get("gather_strategy", "concatenated")
        )
        
        # User's example: create input
        x = torch.ones(1, 1, 32, 32, device=ctx.device)
        
        # User's example: applies physics in parallel
        y = distributed_physics.A(x)
        
        # Check y is correct
        assert isinstance(y, TensorList)
        assert len(y) == 3
        
        # User's example: applies adjoint
        y_forward = distributed_physics.A(x)
        x2 = distributed_physics.A_adjoint(y_forward)
        
        # Verify correctness on rank 0
        if rank == 0:
            # Compare with stacked physics
            y_stacked = stacked_physics.A(x)
            x2_stacked = stacked_physics.A_adjoint(y_stacked)
            
            # Check forward
            for i in range(len(y)):
                max_diff = torch.max(torch.abs(y[i] - y_stacked[i])).item()
                assert max_diff < 1e-5, f"Forward mismatch at index {i}: {max_diff}"
            
            # Check adjoint
            max_diff = torch.max(torch.abs(x2 - x2_stacked)).item()
            assert max_diff < 1e-5, f"Adjoint mismatch: {max_diff}"
        
        return "success"


def test_user_example_code(dist_config, gather_strategy):
    """Test the user's example code pattern works in multi-process mode."""
    test_args = {
        "gather_strategy": gather_strategy
    }
    
    # Run distributed test
    results = run_distributed_test(_test_user_example_worker, dist_config, test_args)
    
    # Check all ranks succeeded
    assert all(r == "success" for r in results)


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_distribute_error_handling():
    """Test that distribute() properly handles errors."""
    with DistributedContext(device_mode="cpu") as ctx:
        # Test callable without num_operators
        def factory(idx, device, shared):
            return Blur(filter=torch.ones(1, 1, 3, 3, device=device), device=device)
        
        with pytest.raises(ValueError, match="num_operators"):
            distribute(factory, ctx, type_object="linear_physics")
        
        # Test callable with auto type (should error)
        with pytest.raises(ValueError, match="type_object"):
            distribute(factory, ctx, type_object="auto")


# =============================================================================
# Tests for Distributed Processors (Priors and Denoisers)
# =============================================================================


# Import prior and denoiser components
from deepinv.optim.prior import Prior
from deepinv.models.base import Denoiser


class SimplePrior(Prior):
    """Simple test prior that scales the input."""
    def __init__(self, scale=0.9):
        super().__init__()
        self.scale = scale
    
    def forward(self, x, *args, **kwargs):
        return x * self.scale


class SimpleDenoiser(Denoiser):
    """Simple test denoiser that scales the input."""
    def __init__(self, scale=0.95):
        super().__init__()
        self.scale = scale
    
    def forward(self, x, sigma=None, *args, **kwargs):
        # Simple scaling operation (not a real denoiser)
        return x * self.scale


@pytest.fixture(params=["basic", "smart_tiling"])
def tiling_strategy(request):
    """Parameterized fixture for tiling strategies."""
    return request.param


@pytest.fixture(params=["prior", "denoiser"])
def processor_type(request):
    """Parameterized fixture for processor types."""
    return request.param


def create_test_processor(processor_type, device):
    """Create a test processor (prior or denoiser)."""
    if processor_type == "prior":
        return SimplePrior(scale=0.9).to(device)
    elif processor_type == "denoiser":
        return SimpleDenoiser(scale=0.95).to(device)
    else:
        raise ValueError(f"Unknown processor type: {processor_type}")


# =============================================================================
# Single-Process Processor Tests
# =============================================================================


def test_distribute_processor_single_process(processor_type, tiling_strategy):
    """Test distribute() with processors in single-process mode."""
    with DistributedContext(device_mode="cpu") as ctx:
        # Create test processor
        processor = create_test_processor(processor_type, ctx.device)
        
        # Distribute the processor
        distributed_processor = distribute(
            processor,
            ctx,
            type_object=processor_type,
            tiling_strategy=tiling_strategy,
            patch_size=8,
            receptive_field_size=2,
        )
        
        # Test forward operation
        x = torch.randn(1, 3, 16, 16, device=ctx.device)
        result = distributed_processor(x)
        
        # Verify result has correct shape and device
        assert result.shape == x.shape
        assert result.device == ctx.device
        
        # Verify result is reasonable (not zeros, not same as input)
        assert not torch.allclose(result, torch.zeros_like(result))
        assert not torch.allclose(result, x)


def test_distribute_processor_different_patch_sizes():
    """Test processor distribution with different patch sizes."""
    with DistributedContext(device_mode="cpu") as ctx:
        processor = SimplePrior(scale=0.9).to(ctx.device)
        
        # Test with small patches
        distributed_small = distribute(
            processor,
            ctx,
            type_object="prior",
            tiling_strategy="smart_tiling",
            patch_size=8,
            receptive_field_size=2,
        )
        
        x = torch.randn(1, 3, 32, 32, device=ctx.device)
        result_small = distributed_small(x)
        assert result_small.shape == x.shape
        
        # Test with large patches (larger than image)
        distributed_large = distribute(
            processor,
            ctx,
            type_object="prior",
            tiling_strategy="smart_tiling",
            patch_size=64,
            receptive_field_size=8,
        )
        
        result_large = distributed_large(x)
        assert result_large.shape == x.shape


def test_distribute_processor_with_max_batch_size():
    """Test processor distribution with different max_batch_size settings."""
    with DistributedContext(device_mode="cpu") as ctx:
        processor = SimpleDenoiser(scale=0.95).to(ctx.device)
        
        x = torch.randn(1, 3, 32, 32, device=ctx.device)
        
        # Test with default batching (all patches at once)
        distributed_default = distribute(
            processor,
            ctx,
            type_object="denoiser",
            tiling_strategy="smart_tiling",
            patch_size=16,
            receptive_field_size=4,
            max_batch_size=None,
        )
        result_default = distributed_default(x, sigma=0.1)
        
        # Test with sequential processing (one patch at a time)
        distributed_seq = distribute(
            processor,
            ctx,
            type_object="denoiser",
            tiling_strategy="smart_tiling",
            patch_size=16,
            receptive_field_size=4,
            max_batch_size=1,
        )
        result_seq = distributed_seq(x, sigma=0.1)
        
        # Results should be identical
        assert torch.allclose(result_default, result_seq, atol=1e-5)


# =============================================================================
# Multi-Process Processor Tests
# =============================================================================


def _test_distributed_processor_worker(rank, world_size, args):
    """Worker function for multi-process processor tests."""
    with DistributedContext(device_mode="cpu") as ctx:
        # Create test processor
        processor_type = args["processor_type"]
        processor = create_test_processor(processor_type, ctx.device)
        
        # Distribute the processor
        distributed_processor = distribute(
            processor,
            ctx,
            type_object=processor_type,
            tiling_strategy=args.get("tiling_strategy", "smart_tiling"),
            patch_size=args.get("patch_size", 8),
            receptive_field_size=args.get("receptive_field_size", 2),
            max_batch_size=args.get("max_batch_size", None),
        )
        
        # Test forward operation with deterministic input
        torch.manual_seed(42)
        x = torch.randn(1, 3, 16, 16, device=ctx.device)
        
        # Apply processor
        if processor_type == "denoiser":
            result = distributed_processor(x, sigma=0.1)
        else:
            result = distributed_processor(x)
        
        # Verify result
        assert result.shape == x.shape
        assert result.device == ctx.device
        
        # All ranks should get the same result due to all_reduce
        return {"result_norm": result.norm().item(), "rank": rank}


def test_distribute_processor_multiprocess(dist_config, processor_type, tiling_strategy):
    """Test distribute() with processors in multi-process mode."""
    test_args = {
        "processor_type": processor_type,
        "tiling_strategy": tiling_strategy,
        "patch_size": 8,
        "receptive_field_size": 2,
    }
    
    # Run distributed test
    results = run_distributed_test(_test_distributed_processor_worker, dist_config, test_args)
    
    # All ranks should have the same result (due to all_reduce)
    norms = [r["result_norm"] for r in results]
    assert all(abs(n - norms[0]) < 1e-4 for n in norms), f"Norms differ: {norms}"


def test_distribute_processor_multiprocess_batching(dist_config):
    """Test processor distribution with different batching strategies in multi-process mode."""
    
    # Test with default batching
    test_args_default = {
        "processor_type": "denoiser",
        "tiling_strategy": "smart_tiling",
        "patch_size": 8,
        "receptive_field_size": 2,
        "max_batch_size": None,
    }
    results_default = run_distributed_test(_test_distributed_processor_worker, dist_config, test_args_default)
    norms_default = [r["result_norm"] for r in results_default]
    
    # Test with sequential processing
    test_args_seq = {
        "processor_type": "denoiser",
        "tiling_strategy": "smart_tiling",
        "patch_size": 8,
        "receptive_field_size": 2,
        "max_batch_size": 1,
    }
    results_seq = run_distributed_test(_test_distributed_processor_worker, dist_config, test_args_seq)
    norms_seq = [r["result_norm"] for r in results_seq]
    
    # Results should be the same regardless of batching strategy
    assert all(abs(n_default - n_seq) < 1e-4 for n_default, n_seq in zip(norms_default, norms_seq))


# =============================================================================
# 3D Processor Tests
# =============================================================================


def _test_distributed_processor_3d_worker(rank, world_size, args):
    """Worker function for 3D processor tests."""
    with DistributedContext(device_mode="cpu") as ctx:
        # Create simple 3D prior
        processor = SimplePrior(scale=0.9).to(ctx.device)
        
        # Distribute with 3D tiling strategy
        distributed_processor = distribute(
            processor,
            ctx,
            type_object="prior",
            tiling_strategy="smart_tiling_3d",
            patch_size=args.get("patch_size", 8),
            receptive_field_size=args.get("receptive_field_size", 2),
            max_batch_size=args.get("max_batch_size", 1),  # Use sequential for 3D
        )
        
        # Test with 3D input
        torch.manual_seed(42)
        x = torch.randn(1, 1, 16, 16, 16, device=ctx.device)
        
        result = distributed_processor(x)
        
        # Verify result
        assert result.shape == x.shape
        assert result.device == ctx.device
        
        return {"result_norm": result.norm().item(), "rank": rank}


def test_distribute_processor_3d(dist_config):
    """Test processor distribution with 3D volumes."""
    test_args = {
        "patch_size": 8,
        "receptive_field_size": 2,
        "max_batch_size": 1,
    }
    
    # Run distributed test
    results = run_distributed_test(_test_distributed_processor_3d_worker, dist_config, test_args)
    
    # All ranks should have the same result
    norms = [r["result_norm"] for r in results]
    assert all(abs(n - norms[0]) < 1e-4 for n in norms), f"Norms differ: {norms}"


# =============================================================================
# Integration Test: Verify Consistency Between Single and Distributed
# =============================================================================


def test_processor_single_vs_distributed():
    """Test that distributed processing gives same results as single-process for simple cases."""
    # Create a simple processor
    processor = SimplePrior(scale=0.9)
    
    # Small test input
    torch.manual_seed(42)
    x = torch.randn(1, 3, 16, 16)
    
    # Single-process result (direct application)
    result_direct = processor(x)
    
    # Distributed single-process result
    with DistributedContext(device_mode="cpu") as ctx:
        distributed_processor = distribute(
            processor.to(ctx.device),
            ctx,
            type_object="prior",
            tiling_strategy="basic",
            patch_size=16,  # Same as image size for exact match
            receptive_field_size=0,  # No overlap
        )
        
        x_ctx = x.to(ctx.device)
        result_distributed = distributed_processor(x_ctx)
    
    # Results should be very close (allowing for numerical precision)
    max_diff = torch.max(torch.abs(result_direct - result_distributed.cpu())).item()
    assert max_diff < 1e-5, f"Results differ by {max_diff}"
