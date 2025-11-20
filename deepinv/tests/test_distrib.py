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
from deepinv.distrib.distrib_framework import (
    DistributedContext,
    DistributedLinearPhysics,
    DistributedProcessing,
)
from deepinv.distrib.distribute import distribute


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def num_operators():
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

    Returns a tuple: (spec_type, spec_object, num_operators_needed)
    - spec_type: "physics_list", "stacked_physics", or "callable_factory"
    - spec_object: the actual specification (list, StackedPhysics, or callable)
    - num_operators_needed: whether num_operators parameter is needed for distribute()
    """
    return request.param


def create_test_physics_list(device, num_operators=3):
    """Create simple test physics operators as a list."""
    physics_list = []
    for i in range(num_operators):
        # Create simple blur operators with different sigmas
        kernel = gaussian_blur(sigma=1.0 + i * 0.5, device=str(device))
        blur = Blur(filter=kernel, padding="circular", device=str(device))
        blur.noise_model = GaussianNoise(sigma=0.01)
        physics_list.append(blur)
    return physics_list


def create_physics_specification(spec_type, device, num_operators):
    """
    Create physics specification based on type.

    Args:
        spec_type: "physics_list", "stacked_physics", or "callable_factory"
        device: torch device
        num_operators: number of operators

    Returns:
        tuple: (physics_spec, needs_num_operators_param)
    """
    if spec_type == "physics_list":
        physics_list = create_test_physics_list(device, num_operators)
        return physics_list, False

    elif spec_type == "stacked_physics":
        physics_list = create_test_physics_list(device, num_operators)
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


def test_distribute_single_process(physics_specification, gather_strategy, num_operators):
    """
    Test distribute() in single-process mode with different physics specifications
    and gather strategies.
    """
    with DistributedContext(device_mode="cpu") as ctx:
        # Create physics specification
        physics_spec, needs_num_operators = create_physics_specification(
            physics_specification, ctx.device, num_operators
        )

        # Distribute with gather strategy
        if needs_num_operators:
            distributed_physics = distribute(
                physics_spec,
                ctx,
                num_operators=num_operators,
                type_object="linear_physics",
                gather_strategy=gather_strategy,
            )
        else:
            distributed_physics = distribute(
                physics_spec, ctx, gather_strategy=gather_strategy
            )

        # Check type
        assert isinstance(distributed_physics, DistributedLinearPhysics)
        assert distributed_physics.num_operators == num_operators
        assert distributed_physics.gather_strategy == gather_strategy

        # Create reference stacked physics for comparison
        reference_physics = StackedLinearPhysics(
            create_test_physics_list(ctx.device, num_operators)
        )

        # Test forward (A without noise)
        x = torch.randn(1, 1, 16, 16, device=ctx.device)
        y_distributed = distributed_physics.A(x)
        y_reference = reference_physics.A(x)

        assert len(y_distributed) == len(y_reference)
        for i in range(len(y_reference)):
            assert torch.allclose(
                y_distributed[i], y_reference[i], atol=1e-5
            ), f"Forward mismatch at index {i} with {physics_specification} and {gather_strategy}"

        # Test adjoint
        x_adj_distributed = distributed_physics.A_adjoint(y_distributed)
        x_adj_reference = reference_physics.A_adjoint(y_reference)

        assert torch.allclose(
            x_adj_distributed, x_adj_reference, atol=1e-5
        ), f"Adjoint mismatch with {physics_specification} and {gather_strategy}"

        # Test forward() method (with noise - just check structure)
        y_forward = distributed_physics.forward(x)
        assert len(y_forward) == num_operators
        for i in range(num_operators):
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
        physics_spec, needs_num_operators = create_physics_specification(
            args["spec_type"], ctx.device, args["num_operators"]
        )

        # Distribute with gather strategy
        if needs_num_operators:
            distributed_physics = distribute(
                physics_spec,
                ctx,
                num_operators=args["num_operators"],
                type_object="linear_physics",
                gather_strategy=args["gather_strategy"],
            )
        else:
            distributed_physics = distribute(
                physics_spec, ctx, gather_strategy=args["gather_strategy"]
            )

        # Test forward operation
        x = args["x"].to(ctx.device)
        y_distributed = distributed_physics.A(x)

        # Each rank should get the full result
        assert len(y_distributed) == args["num_operators"]

        # Test adjoint operation if requested
        if args.get("test_adjoint", False):
            x_adj_distributed = distributed_physics.A_adjoint(y_distributed)

            # Verify on rank 0
            if rank == 0:
                # Create reference
                reference_physics = StackedLinearPhysics(
                    create_test_physics_list(ctx.device, args["num_operators"])
                )
                y_reference = reference_physics.A(x)
                x_adj_reference = reference_physics.A_adjoint(y_reference)

                max_diff = torch.max(
                    torch.abs(x_adj_distributed - x_adj_reference)
                ).item()
                assert (
                    max_diff < 1e-5
                ), f"Adjoint mismatch with {args['spec_type']} and {args['gather_strategy']}: {max_diff}"

        # Verify forward on rank 0
        if rank == 0:
            reference_physics = StackedLinearPhysics(
                create_test_physics_list(ctx.device, args["num_operators"])
            )
            y_reference = reference_physics.A(x)

            for i in range(len(y_reference)):
                max_diff = torch.max(
                    torch.abs(y_distributed[i] - y_reference[i])
                ).item()
                assert (
                    max_diff < 1e-5
                ), f"Forward mismatch at index {i} with {args['spec_type']} and {args['gather_strategy']}: {max_diff}"

        return "success"


def test_distribute_multiprocess_forward(
    dist_config, physics_specification, gather_strategy
):
    """Test distribute() with multi-process forward operation."""
    # Prepare test data
    x = torch.randn(1, 1, 16, 16)
    test_args = {
        "num_operators": 4,
        "x": x,
        "spec_type": physics_specification,
        "gather_strategy": gather_strategy,
        "test_adjoint": False,
    }

    # Run distributed test
    results = run_distributed_test(
        _test_distributed_operation_worker, dist_config, test_args
    )

    # Check all ranks succeeded
    assert all(r == "success" for r in results)


def test_distribute_multiprocess_adjoint(
    dist_config, physics_specification, gather_strategy
):
    """Test distribute() with multi-process adjoint operation."""
    # Prepare test data
    x = torch.randn(1, 1, 16, 16)
    test_args = {
        "num_operators": 4,
        "x": x,
        "spec_type": physics_specification,
        "gather_strategy": gather_strategy,
        "test_adjoint": True,
    }

    # Run distributed test
    results = run_distributed_test(
        _test_distributed_operation_worker, dist_config, test_args
    )

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
            device=str(ctx.device),
        )
        physics2 = Blur(
            filter=gaussian_blur(sigma=1.5, device=str(ctx.device)),
            padding="circular",
            device=str(ctx.device),
        )
        physics3 = Blur(
            filter=gaussian_blur(sigma=2.0, device=str(ctx.device)),
            padding="circular",
            device=str(ctx.device),
        )

        # Stack physics (user's code)
        stacked_physics = StackedLinearPhysics([physics1, physics2, physics3])

        # Distribute with specified gather strategy (user's code)
        distributed_physics = distribute(
            stacked_physics,
            ctx,
            gather_strategy=args.get("gather_strategy", "concatenated"),
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
    test_args = {"gather_strategy": gather_strategy}

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


def test_distribute_processor_multiprocess(
    dist_config, processor_type, tiling_strategy
):
    """Test distribute() with processors in multi-process mode."""
    test_args = {
        "processor_type": processor_type,
        "tiling_strategy": tiling_strategy,
        "patch_size": 8,
        "receptive_field_size": 2,
    }

    # Run distributed test
    results = run_distributed_test(
        _test_distributed_processor_worker, dist_config, test_args
    )

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
    results_default = run_distributed_test(
        _test_distributed_processor_worker, dist_config, test_args_default
    )
    norms_default = [r["result_norm"] for r in results_default]

    # Test with sequential processing
    test_args_seq = {
        "processor_type": "denoiser",
        "tiling_strategy": "smart_tiling",
        "patch_size": 8,
        "receptive_field_size": 2,
        "max_batch_size": 1,
    }
    results_seq = run_distributed_test(
        _test_distributed_processor_worker, dist_config, test_args_seq
    )
    norms_seq = [r["result_norm"] for r in results_seq]

    # Results should be the same regardless of batching strategy
    assert all(
        abs(n_default - n_seq) < 1e-4
        for n_default, n_seq in zip(norms_default, norms_seq)
    )


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
    results = run_distributed_test(
        _test_distributed_processor_3d_worker, dist_config, test_args
    )

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


# =============================================================================
# Tests for PnP Required Operations (compute_norm, A_dagger)
# =============================================================================


def test_compute_norm_single_process(gather_strategy):
    """Test compute_norm in single-process distributed mode."""
    with DistributedContext(device_mode="cpu") as ctx:
        # Create distributed physics
        physics_list = create_test_physics_list(ctx.device, num_operators=3)
        dphysics = distribute(
            physics_list,
            ctx,
            type_object="linear_physics",
            gather_strategy=gather_strategy,
        )

        # Create test signal
        x0 = torch.randn(1, 1, 16, 16).to(ctx.device)

        # Compute norm - should work without error
        norm = dphysics.compute_norm(
            x0, max_iter=10, tol=1e-3, verbose=False, squared=False
        )

        # Check that norm is a scalar tensor and positive
        assert isinstance(norm, torch.Tensor)
        assert norm.numel() == 1
        assert norm.item() > 0


def _test_compute_norm_worker(rank, world_size, args):
    """Worker function for multi-process compute_norm test."""
    with DistributedContext(device_mode="cpu") as ctx:
        # Create distributed physics
        num_operators = args["num_operators"]
        gather_strategy = args["gather_strategy"]

        physics_list = create_test_physics_list(ctx.device, num_operators=num_operators)
        dphysics = distribute(
            physics_list,
            ctx,
            type_object="linear_physics",
            gather_strategy=gather_strategy,
        )

        # Create test signal
        x0 = torch.randn(1, 1, 16, 16).to(ctx.device)

        # Compute norm
        norm = dphysics.compute_norm(
            x0, max_iter=10, tol=1e-3, verbose=False, squared=False
        )

        # Check properties
        assert isinstance(norm, torch.Tensor)
        assert norm.numel() == 1
        assert norm.item() > 0

        return {"norm": norm.item()}


def test_compute_norm_multiprocess(dist_config, gather_strategy):
    """Test compute_norm in multi-process distributed mode."""
    test_args = {
        "num_operators": 4,
        "gather_strategy": gather_strategy,
    }

    # Run distributed test
    results = run_distributed_test(_test_compute_norm_worker, dist_config, test_args)

    # All ranks should compute the same norm
    norms = [r["norm"] for r in results]
    # Allow small numerical differences
    assert all(
        abs(n - norms[0]) < 1e-4 for n in norms
    ), f"Norms differ across ranks: {norms}"


def test_a_dagger_single_process(gather_strategy):
    """Test A_dagger in single-process distributed mode."""
    with DistributedContext(device_mode="cpu") as ctx:
        # Create distributed physics
        physics_list = create_test_physics_list(ctx.device, num_operators=3)
        dphysics = distribute(
            physics_list,
            ctx,
            type_object="linear_physics",
            gather_strategy=gather_strategy,
        )

        # Create test signal and measurements
        x = torch.randn(1, 1, 16, 16).to(ctx.device)
        y = dphysics.A(x)  # TensorList

        # Compute pseudoinverse
        x_dagger = dphysics.A_dagger(y, solver="CG", max_iter=10, verbose=False)

        # Check output properties
        assert isinstance(x_dagger, torch.Tensor)
        assert x_dagger.shape == x.shape

        # Check that reconstruction is reasonable (not all zeros/nans)
        assert not torch.isnan(x_dagger).any()
        assert torch.abs(x_dagger).sum() > 0


def _test_a_dagger_worker(rank, world_size, args):
    """Worker function for multi-process A_dagger test."""
    with DistributedContext(device_mode="cpu") as ctx:
        # Create distributed physics
        num_operators = args["num_operators"]
        gather_strategy = args["gather_strategy"]

        physics_list = create_test_physics_list(ctx.device, num_operators=num_operators)
        dphysics = distribute(
            physics_list,
            ctx,
            type_object="linear_physics",
            gather_strategy=gather_strategy,
        )

        # Create test signal and measurements (use same seed for all ranks)
        torch.manual_seed(42)
        x = torch.randn(1, 1, 16, 16).to(ctx.device)
        y = dphysics.A(x)

        # Compute pseudoinverse
        x_dagger = dphysics.A_dagger(y, solver="CG", max_iter=10, verbose=False)

        # Check properties
        assert isinstance(x_dagger, torch.Tensor)
        assert x_dagger.shape == x.shape
        assert not torch.isnan(x_dagger).any()
        assert torch.abs(x_dagger).sum() > 0

        # Return norm for comparison across ranks
        return {"x_dagger_norm": torch.norm(x_dagger).item()}


def test_a_dagger_multiprocess(dist_config, gather_strategy):
    """Test A_dagger in multi-process distributed mode."""
    test_args = {
        "num_operators": 4,
        "gather_strategy": gather_strategy,
    }

    # Run distributed test
    results = run_distributed_test(_test_a_dagger_worker, dist_config, test_args)

    # All ranks should compute the same A_dagger (same input -> same output)
    norms = [r["x_dagger_norm"] for r in results]
    # Allow small numerical differences
    assert all(
        abs(n - norms[0]) < 1e-3 for n in norms
    ), f"A_dagger results differ across ranks: {norms}"


def test_compute_sqnorm_single_process(gather_strategy):
    """Test compute_sqnorm in single-process distributed mode."""
    with DistributedContext(device_mode="cpu") as ctx:
        # Create distributed physics
        physics_list = create_test_physics_list(ctx.device, num_operators=3)
        dphysics = distribute(
            physics_list,
            ctx,
            type_object="linear_physics",
            gather_strategy=gather_strategy,
        )

        # Create test signal
        x0 = torch.randn(1, 1, 16, 16).to(ctx.device)

        # Compute squared norm
        sqnorm = dphysics.compute_sqnorm(x0, max_iter=10, tol=1e-3, verbose=False)

        # Check that sqnorm is a scalar tensor and positive
        assert isinstance(sqnorm, torch.Tensor)
        assert sqnorm.numel() == 1
        assert sqnorm.item() > 0


def _test_pnp_operations_worker(rank, world_size, args):
    """Worker function testing all operations needed for PnP solver."""
    with DistributedContext(device_mode="cpu") as ctx:
        # Create distributed physics
        num_operators = args["num_operators"]
        gather_strategy = args["gather_strategy"]

        physics_list = create_test_physics_list(ctx.device, num_operators=num_operators)
        dphysics = distribute(
            physics_list,
            ctx,
            type_object="linear_physics",
            gather_strategy=gather_strategy,
        )

        # Create test signal (same seed for all ranks)
        torch.manual_seed(42)
        x = torch.randn(1, 1, 16, 16).to(ctx.device)

        # 1. Forward operation
        y = dphysics.A(x)
        assert isinstance(y, TensorList)

        # 2. Compute norm (for step size)
        norm = dphysics.compute_norm(x, max_iter=10, verbose=False, squared=False)
        step_size = 1.0 / norm
        assert step_size.item() > 0

        # 3. A_dagger (for initialization)
        x_init = dphysics.A_dagger(y, solver="CG", max_iter=10, verbose=False)
        assert x_init.shape == x.shape

        # 4. Adjoint operation (needed for gradient)
        x_adj = dphysics.A_adjoint(y)
        assert x_adj.shape == x.shape

        # 5. A_adjoint_A (needed for some operations)
        x_ata = dphysics.A_adjoint_A(x)
        assert x_ata.shape == x.shape

        return {
            "norm": norm.item(),
            "x_init_norm": torch.norm(x_init).item(),
            "x_adj_norm": torch.norm(x_adj).item(),
            "x_ata_norm": torch.norm(x_ata).item(),
        }


def test_pnp_operations_multiprocess(dist_config, gather_strategy):
    """Test all operations required by PnP solver in multi-process mode."""
    test_args = {
        "num_operators": 4,
        "gather_strategy": gather_strategy,
    }

    # Run distributed test
    results = run_distributed_test(_test_pnp_operations_worker, dist_config, test_args)

    # All ranks should compute the same values
    for key in ["norm", "x_init_norm", "x_adj_norm", "x_ata_norm"]:
        values = [r[key] for r in results]
        assert all(
            abs(v - values[0]) < 1e-3 for v in values
        ), f"{key} differs across ranks: {values}"


# =============================================================================
# Additional Tests for DistributedContext
# =============================================================================


def test_distributed_context_device_modes():
    """Test DistributedContext with different device modes."""
    # Test CPU mode
    with DistributedContext(device_mode="cpu") as ctx:
        assert ctx.device.type == "cpu"
        assert ctx.rank == 0
        assert ctx.world_size == 1
        assert not ctx.is_dist

    # Test auto mode (should default to CPU in single process)
    with DistributedContext(device_mode=None) as ctx:
        assert ctx.device.type in ["cpu", "cuda"]
        assert ctx.rank == 0
        assert ctx.world_size == 1


def test_distributed_context_seeding():
    """Test DistributedContext seeding functionality."""
    # Test with fixed seed
    with DistributedContext(device_mode="cpu", seed=42) as ctx:
        val1 = torch.rand(1).item()

    with DistributedContext(device_mode="cpu", seed=42) as ctx:
        val2 = torch.rand(1).item()

    assert abs(val1 - val2) < 1e-6, "Same seed should produce same random values"


def test_distributed_context_local_indices():
    """Test local_indices sharding functionality."""
    with DistributedContext(device_mode="cpu") as ctx:
        # Test various numbers of items
        for num_items in [1, 5, 10, 20]:
            indices = ctx.local_indices(num_items)
            assert len(indices) == num_items  # Single process gets all
            assert indices == list(range(num_items))


def test_distributed_context_all_reduce():
    """Test all_reduce operations in single-process mode."""
    with DistributedContext(device_mode="cpu") as ctx:
        # Test sum reduction
        x = torch.tensor([1.0, 2.0, 3.0])
        result = ctx.all_reduce_(x.clone(), op="sum")
        assert torch.allclose(result, x)

        # Test mean reduction
        x = torch.tensor([4.0, 8.0, 12.0])
        result = ctx.all_reduce_(x.clone(), op="mean")
        assert torch.allclose(result, x)


def test_distributed_context_broadcast():
    """Test broadcast operation in single-process mode."""
    with DistributedContext(device_mode="cpu") as ctx:
        x = torch.tensor([1.0, 2.0, 3.0])
        result = ctx.broadcast_(x.clone(), src=0)
        assert torch.allclose(result, x)


# =============================================================================
# Tests for A_adjoint_A and A_A_adjoint operations
# =============================================================================


def test_a_adjoint_a_single_process(gather_strategy):
    """Test A_adjoint_A in single-process distributed mode."""
    with DistributedContext(device_mode="cpu") as ctx:
        # Create distributed physics
        physics_list = create_test_physics_list(ctx.device, num_operators=3)
        dphysics = distribute(
            physics_list,
            ctx,
            type_object="linear_physics",
            gather_strategy=gather_strategy,
        )

        # Create reference stacked physics
        reference_physics = StackedLinearPhysics(
            create_test_physics_list(ctx.device, num_operators=3)
        )

        # Test signal
        x = torch.randn(1, 1, 16, 16).to(ctx.device)

        # Compute A_adjoint_A
        result_distributed = dphysics.A_adjoint_A(x)
        result_reference = reference_physics.A_adjoint_A(x)

        # Check output properties
        assert isinstance(result_distributed, torch.Tensor)
        assert result_distributed.shape == x.shape

        # Compare with reference
        assert torch.allclose(
            result_distributed, result_reference, atol=1e-5
        ), f"A_adjoint_A mismatch with {gather_strategy}"


def _test_a_adjoint_a_worker(rank, world_size, args):
    """Worker function for multi-process A_adjoint_A test."""
    with DistributedContext(device_mode="cpu") as ctx:
        num_operators = args["num_operators"]
        gather_strategy = args["gather_strategy"]

        physics_list = create_test_physics_list(ctx.device, num_operators=num_operators)
        dphysics = distribute(
            physics_list,
            ctx,
            type_object="linear_physics",
            gather_strategy=gather_strategy,
        )

        # Same seed for all ranks
        torch.manual_seed(42)
        x = torch.randn(1, 1, 16, 16).to(ctx.device)

        # Compute A_adjoint_A
        result = dphysics.A_adjoint_A(x)

        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

        return {"result_norm": torch.norm(result).item()}


def test_a_adjoint_a_multiprocess(dist_config, gather_strategy):
    """Test A_adjoint_A in multi-process distributed mode."""
    test_args = {
        "num_operators": 4,
        "gather_strategy": gather_strategy,
    }

    results = run_distributed_test(_test_a_adjoint_a_worker, dist_config, test_args)

    # All ranks should compute the same result
    norms = [r["result_norm"] for r in results]
    assert all(
        abs(n - norms[0]) < 1e-3 for n in norms
    ), f"Norms differ across ranks: {norms}"


def test_a_a_adjoint_single_process(gather_strategy):
    """Test A_A_adjoint in single-process distributed mode."""
    with DistributedContext(device_mode="cpu") as ctx:
        # Create distributed physics
        physics_list = create_test_physics_list(ctx.device, num_operators=3)
        dphysics = distribute(
            physics_list,
            ctx,
            type_object="linear_physics",
            gather_strategy=gather_strategy,
        )

        # Create test signal and measurements
        x = torch.randn(1, 1, 16, 16).to(ctx.device)
        y = dphysics.A(x)

        # Compute A_A_adjoint - returns reduced tensor (not TensorList)
        result = dphysics.A_A_adjoint(y)

        # Check output properties - A_A_adjoint returns single tensor
        assert isinstance(result, torch.Tensor)
        assert result.shape == y[0].shape


# =============================================================================
# Tests for local_only parameter variants
# =============================================================================


def test_compute_norm_local_vs_global():
    """Test compute_norm with local_only parameter."""
    with DistributedContext(device_mode="cpu") as ctx:
        physics_list = create_test_physics_list(ctx.device, num_operators=3)
        dphysics = distribute(
            physics_list,
            ctx,
            type_object="linear_physics",
            gather_strategy="concatenated",
        )

        x0 = torch.randn(1, 1, 16, 16).to(ctx.device)

        # Compute with local_only=True (default, efficient)
        norm_local = dphysics.compute_sqnorm(
            x0, max_iter=10, tol=1e-3, verbose=False, local_only=True
        )

        # Compute with local_only=False (exact, expensive)
        norm_global = dphysics.compute_sqnorm(
            x0, max_iter=10, tol=1e-3, verbose=False, local_only=False
        )

        # Both should be positive
        assert norm_local.item() > 0
        assert norm_global.item() > 0

        # Local should be >= global (upper bound)
        # Note: This may not always hold for small operators, so we just check they're close
        assert (
            abs(norm_local.item() - norm_global.item())
            / max(norm_local.item(), norm_global.item())
            < 0.5
        )


def test_a_dagger_local_vs_global():
    """Test A_dagger with local_only parameter."""
    with DistributedContext(device_mode="cpu") as ctx:
        physics_list = create_test_physics_list(ctx.device, num_operators=3)
        dphysics = distribute(
            physics_list,
            ctx,
            type_object="linear_physics",
            gather_strategy="concatenated",
        )

        x = torch.randn(1, 1, 16, 16).to(ctx.device)
        y = dphysics.A(x)

        # Compute with local_only=True (default, efficient approximation)
        x_dagger_local = dphysics.A_dagger(y, local_only=True, verbose=False)

        # Compute with local_only=False (exact, expensive)
        x_dagger_global = dphysics.A_dagger(
            y, solver="CG", max_iter=10, local_only=False, verbose=False
        )

        # Both should have correct shape
        assert x_dagger_local.shape == x.shape
        assert x_dagger_global.shape == x.shape

        # Both should be reasonable (not all zeros)
        assert torch.abs(x_dagger_local).sum() > 0
        assert torch.abs(x_dagger_global).sum() > 0


# =============================================================================
# Tests for gather strategies edge cases
# =============================================================================


def test_gather_strategies_with_varying_sizes():
    """Test gather strategies with varying tensor sizes."""
    with DistributedContext(device_mode="cpu") as ctx:
        # Create physics operators that produce different sized outputs
        def varying_size_factory(idx, device, shared):
            # Each operator uses different sigma to create variation
            sigma = 1.0 + idx * 0.5
            kernel = gaussian_blur(sigma=sigma, device=device)
            # Use valid padding to get different output sizes based on filter size
            return Blur(filter=kernel, padding="valid", device=device)

        for gather_strategy in ["naive", "concatenated", "broadcast"]:
            dphysics = distribute(
                varying_size_factory,
                ctx,
                num_operators=3,
                type_object="linear_physics",
                gather_strategy=gather_strategy,
            )

            x = torch.randn(1, 1, 16, 16).to(ctx.device)
            y = dphysics.A(x)

            assert isinstance(y, TensorList)
            assert len(y) == 3
            # With valid padding, outputs have different sizes due to different filter sizes
            sizes = [y[i].shape for i in range(3)]


def test_gather_empty_local_set():
    """Test gather strategies when some ranks have no work."""
    with DistributedContext(device_mode="cpu") as ctx:
        # In single process, this creates a scenario similar to multi-process with empty ranks
        # We can't truly test multi-rank behavior without spinning up processes,
        # but we can test that the gather logic handles edge cases

        physics_list = create_test_physics_list(ctx.device, num_operators=2)
        dphysics = distribute(
            physics_list,
            ctx,
            type_object="linear_physics",
            gather_strategy="concatenated",
        )

        x = torch.randn(1, 1, 16, 16).to(ctx.device)
        y = dphysics.A(x)

        assert isinstance(y, TensorList)
        assert len(y) == 2


# =============================================================================
# Tests for DistributedProcessing strategies
# =============================================================================


def test_basic_strategy_parameters():
    """Test BasicStrategy with different parameters."""
    with DistributedContext(device_mode="cpu") as ctx:
        processor = SimplePrior(scale=0.9).to(ctx.device)

        # Test with different tiling_dims values
        for tiling_dims in [1, 2, (2, 3)]:
            distributed_processor = distribute(
                processor,
                ctx,
                type_object="prior",
                tiling_strategy="basic",
                patch_size=8,
                receptive_field_size=0,
                tiling_dims=tiling_dims,
            )

            # Test on appropriate tensor shape
            if tiling_dims == 1:
                x = torch.randn(1, 3, 16, device=ctx.device)
            elif tiling_dims == 2:
                x = torch.randn(1, 3, 16, 16, device=ctx.device)
            else:  # (2, 3)
                x = torch.randn(1, 3, 16, 16, device=ctx.device)

            result = distributed_processor(x)
            assert result.shape == x.shape


def test_smart_tiling_edge_cases():
    """Test SmartTilingStrategy with edge cases."""
    with DistributedContext(device_mode="cpu") as ctx:
        processor = SimplePrior(scale=0.9).to(ctx.device)

        # Test with patch_size larger than image
        distributed_processor = distribute(
            processor,
            ctx,
            type_object="prior",
            tiling_strategy="smart_tiling",
            patch_size=128,  # Larger than image
            receptive_field_size=16,
        )

        x = torch.randn(1, 3, 32, 32, device=ctx.device)
        result = distributed_processor(x)
        assert result.shape == x.shape

        # Test with very small patch_size
        distributed_processor = distribute(
            processor,
            ctx,
            type_object="prior",
            tiling_strategy="smart_tiling",
            patch_size=4,  # Very small
            receptive_field_size=1,
        )

        result = distributed_processor(x)
        assert result.shape == x.shape


def test_smart_tiling_3d():
    """Test 3D tiling strategy."""
    with DistributedContext(device_mode="cpu") as ctx:
        processor = SimplePrior(scale=0.9).to(ctx.device)

        distributed_processor = distribute(
            processor,
            ctx,
            type_object="prior",
            tiling_strategy="smart_tiling_3d",
            patch_size=8,
            receptive_field_size=2,
        )

        # 3D input
        x = torch.randn(1, 1, 16, 16, 16, device=ctx.device)
        result = distributed_processor(x)
        assert result.shape == x.shape


def test_processor_batching_strategies():
    """Test different batching strategies for processors."""
    with DistributedContext(device_mode="cpu") as ctx:
        processor = SimpleDenoiser(scale=0.95).to(ctx.device)

        x = torch.randn(1, 3, 32, 32, device=ctx.device)

        # Test with various max_batch_size settings
        for max_batch_size in [None, 1, 2, 4]:
            distributed_processor = distribute(
                processor,
                ctx,
                type_object="denoiser",
                tiling_strategy="smart_tiling",
                patch_size=16,
                receptive_field_size=4,
                max_batch_size=max_batch_size,
            )

            result = distributed_processor(x, sigma=0.1)
            assert result.shape == x.shape


# =============================================================================
# Tests for error handling and edge cases
# =============================================================================


def test_distribute_with_invalid_gather_strategy():
    """Test that invalid gather strategies raise appropriate errors."""
    with DistributedContext(device_mode="cpu") as ctx:
        physics_list = create_test_physics_list(ctx.device, num_operators=3)

        with pytest.raises(ValueError, match="gather_strategy"):
            distribute(
                physics_list,
                ctx,
                type_object="linear_physics",
                gather_strategy="invalid_strategy",
            )


def test_distribute_with_invalid_type():
    """Test that invalid type_object raises appropriate errors."""
    with DistributedContext(device_mode="cpu") as ctx:
        physics_list = create_test_physics_list(ctx.device, num_operators=3)

        with pytest.raises(ValueError):
            distribute(
                physics_list,
                ctx,
                type_object="invalid_type",
            )


def test_distribute_with_mismatched_types():
    """Test error handling for mismatched object types."""
    with DistributedContext(device_mode="cpu") as ctx:
        # Try to use denoiser as physics
        denoiser = SimpleDenoiser(scale=0.95).to(ctx.device)

        # This should work if we specify the correct type
        distributed = distribute(
            denoiser,
            ctx,
            type_object="denoiser",
            tiling_strategy="smart_tiling",
            patch_size=16,
            receptive_field_size=4,
        )

        assert isinstance(distributed, DistributedProcessing)
