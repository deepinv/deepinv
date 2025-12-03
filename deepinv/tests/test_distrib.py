"""
Tests for the distributed framework (distribute function and distributed components).

This module tests the distributed API that enables converting Physics, DataFidelity,
and Denoisers into distributed versions that work across multiple processes and GPUs.

Test Organization:
- Test helpers and fixtures (CPU/GPU support)
- Core distributed physics tests
- Distributed processor (denoiser) tests
- Distributed data fidelity tests
- Advanced operations (A_dagger, compute_norm, reduce=False parameter)
- Integration tests

All tests support both CPU-only and multi-GPU configurations automatically.
"""

from __future__ import annotations
import os
import pytest
import torch
import time
import socket
import torch.multiprocessing as mp
from typing import Callable, Any

from deepinv.physics import Blur, GaussianNoise, LinearPhysics
from deepinv.physics.blur import gaussian_blur
from deepinv.physics.forward import StackedPhysics, StackedLinearPhysics
from deepinv.utils.tensorlist import TensorList
from deepinv.models.base import Denoiser
from deepinv.optim import L2, L1
from deepinv.optim.data_fidelity import StackedPhysicsDataFidelity

from deepinv.distrib.distrib_framework import (
    DistributedContext,
    DistributedLinearPhysics,
    DistributedProcessing,
    DistributedDataFidelity,
)
from deepinv.distrib.distribute import distribute


# =============================================================================
# Test Infrastructure: Multi-GPU and CPU Support
# =============================================================================


def _get_free_port():
    """Get a free port for distributed communication."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get_gpu_count():
    """Get the number of available GPUs."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


@pytest.fixture(params=["cpu", "gpu"])
def device_config(request):
    """
    Parameterized fixture for device configuration.

    Returns:
        dict with keys:
            - device_mode: "cpu" or "gpu"
            - world_size: number of processes to spawn
            - skip_reason: reason to skip if device not available
    """
    mode = request.param

    if mode == "cpu":
        return {
            "device_mode": "cpu",
            "world_size": 2,
            "skip_reason": None,
        }
    elif mode == "gpu":
        gpu_count = _get_gpu_count()
        if gpu_count == 0:
            return {
                "device_mode": "gpu",
                "world_size": 0,
                "skip_reason": "No GPUs available",
            }
        elif gpu_count == 1:
            # Single GPU: can't test multi-process NCCL
            return {
                "device_mode": "gpu",
                "world_size": 1,
                "skip_reason": None,
            }
        else:
            # Multi-GPU: use 2 GPUs for testing
            return {
                "device_mode": "gpu",
                "world_size": min(2, gpu_count),
                "skip_reason": None,
            }


def _worker(rank, world_size, test_func, test_args, result_queue, dist_config):
    """
    Worker function that runs in each process - must be at module level for pickling.

    Supports both CPU (Gloo) and GPU (NCCL) backends automatically.
    """
    # Set environment variables for this rank
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = dist_config["master_addr"]
    os.environ["MASTER_PORT"] = dist_config["master_port"]

    # For GPU mode, set CUDA_VISIBLE_DEVICES to isolate GPUs
    if dist_config["device_mode"] == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

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
        for key in [
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
            "CUDA_VISIBLE_DEVICES",
        ]:
            os.environ.pop(key, None)


def run_distributed_test(
    test_func: Callable,
    device_config: dict,
    test_args: dict = None,
    timeout_per_rank: float = 12.0,
) -> list[Any]:
    """
    Run a test function across multiple ranks using torch.multiprocessing.

    Automatically handles both CPU (Gloo) and GPU (NCCL) configurations.

    Args:
        test_func: picklable callable(rank: int, world_size: int, args: dict) -> Any
        device_config: device configuration from fixture
        test_args: optional dict passed to test_func on each rank
        timeout_per_rank: timeout budget per rank in seconds

    Returns:
        List of per-rank results ordered by rank.

    Raises:
        RuntimeError if any rank fails, times out, or does not report a result.
    """
    # Check if we should skip this test
    if device_config.get("skip_reason"):
        pytest.skip(device_config["skip_reason"])

    world_size = device_config["world_size"]
    device_mode = device_config["device_mode"]
    args = test_args or {}

    # Single-process path: run directly
    if world_size == 1:
        return [test_func(0, 1, args)]

    # Multi-process configuration
    dist_config = {
        "world_size": world_size,
        "backend": "nccl" if device_mode == "gpu" else "gloo",
        "master_addr": "127.0.0.1",
        "master_port": str(_get_free_port()),
        "device_mode": device_mode,
    }

    # Keep thread pools small for spawn safety
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if device_mode == "cpu":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")

    # Use 'spawn' for safety with PyTorch/NumPy
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    # Timeout policy
    timeout = max(20.0, timeout_per_rank * world_size)

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
# Test Helpers: Physics and Processor Creation
# =============================================================================


class SimpleDenoiser(Denoiser):
    """Simple test denoiser that scales the input."""

    def __init__(self, scale=0.95):
        super().__init__()
        self.scale = scale

    def forward(self, x, sigma=None, *args, **kwargs):
        return x * self.scale


def create_test_physics_list(device, num_operators=3):
    """Create simple test physics operators as a list."""
    physics_list = []
    for i in range(num_operators):
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
# Core Tests: Distributed Physics Operations
# =============================================================================


@pytest.mark.parametrize("gather_strategy", ["naive", "concatenated", "broadcast"])
@pytest.mark.parametrize(
    "physics_spec", ["physics_list", "stacked_physics", "callable_factory"]
)
def test_distribute_physics_forward(device_config, gather_strategy, physics_spec):
    """Test distributing physics operators and forward operation."""
    # Skip GPU tests if not available
    if device_config.get("skip_reason"):
        pytest.skip(device_config["skip_reason"])

    # Single process test
    with DistributedContext(device_mode=device_config["device_mode"]) as ctx:
        num_operators = 3
        physics, needs_num_ops = create_physics_specification(
            physics_spec, ctx.device, num_operators
        )

        # Distribute
        if needs_num_ops:
            distributed_physics = distribute(
                physics,
                ctx,
                type_object="physics",
                num_operators=num_operators,
                gather_strategy=gather_strategy,
            )
        else:
            distributed_physics = distribute(
                physics, ctx, gather_strategy=gather_strategy
            )

        # Test forward
        x = torch.randn(1, 1, 16, 16, device=ctx.device)
        y = distributed_physics.A(x)

        # Verify structure
        assert len(y) == num_operators
        assert all(yi.device == ctx.device for yi in y)

        # Compare with reference
        reference = StackedLinearPhysics(
            create_test_physics_list(ctx.device, num_operators)
        )
        y_ref = reference.A(x)

        for i in range(num_operators):
            assert torch.allclose(y[i], y_ref[i], atol=1e-5)


def _test_multiprocess_physics_worker(rank, world_size, args):
    """Worker for multi-process physics tests."""
    with DistributedContext(device_mode=args["device_mode"]) as ctx:
        physics, needs_num_ops = create_physics_specification(
            args["spec_type"], ctx.device, args["num_operators"]
        )

        if needs_num_ops:
            distributed_physics = distribute(
                physics,
                ctx,
                type_object="physics",
                num_operators=args["num_operators"],
                gather_strategy=args["gather_strategy"],
            )
        else:
            distributed_physics = distribute(
                physics, ctx, gather_strategy=args["gather_strategy"]
            )

        # Test operation
        x = args["x"].to(ctx.device)

        if args.get("test_adjoint"):
            y = distributed_physics.A(x)
            result = distributed_physics.A_adjoint(y)
        else:
            result = distributed_physics.A(x)

        # Return success indicator
        return "success"


@pytest.mark.parametrize("gather_strategy", ["naive", "concatenated"])
@pytest.mark.parametrize("physics_spec", ["physics_list", "stacked_physics"])
def test_distribute_physics_multiprocess(device_config, gather_strategy, physics_spec):
    """Test physics distribution in multi-process mode."""
    x = torch.randn(1, 1, 16, 16)
    test_args = {
        "num_operators": 4,
        "x": x,
        "spec_type": physics_spec,
        "gather_strategy": gather_strategy,
        "device_mode": device_config["device_mode"],
        "test_adjoint": False,
    }

    results = run_distributed_test(
        _test_multiprocess_physics_worker, device_config, test_args
    )

    assert all(r == "success" for r in results)


# =============================================================================
# Distributed Processor Tests
# =============================================================================


@pytest.mark.parametrize("tiling_strategy", ["basic", "smart_tiling"])
def test_distribute_processor_single(device_config, tiling_strategy):
    """Test processor distribution in single-process mode."""
    if device_config.get("skip_reason"):
        pytest.skip(device_config["skip_reason"])

    with DistributedContext(device_mode=device_config["device_mode"]) as ctx:
        processor = SimpleDenoiser(scale=0.9).to(ctx.device)

        distributed_processor = distribute(
            processor,
            ctx,
            type_object="denoiser",
            tiling_strategy=tiling_strategy,
            patch_size=8,
            receptive_field_size=2,
        )

        x = torch.randn(1, 3, 16, 16, device=ctx.device)
        result = distributed_processor(x)

        assert result.shape == x.shape
        assert result.device == ctx.device


def _test_multiprocess_processor_worker(rank, world_size, args):
    """Worker for multi-process processor tests."""
    with DistributedContext(device_mode=args["device_mode"]) as ctx:
        processor = SimpleDenoiser(scale=0.9).to(ctx.device)

        distributed_processor = distribute(
            processor,
            ctx,
            type_object="denoiser",
            tiling_strategy=args["tiling_strategy"],
            patch_size=args["patch_size"],
            receptive_field_size=args["receptive_field_size"],
        )

        torch.manual_seed(42)
        x = torch.randn(1, 3, 16, 16, device=ctx.device)
        result = distributed_processor(x)

        return {"result_norm": result.norm().item(), "rank": rank}


@pytest.mark.parametrize("tiling_strategy", ["smart_tiling"])
def test_distribute_processor_multiprocess(device_config, tiling_strategy):
    """Test processor distribution in multi-process mode."""
    test_args = {
        "device_mode": device_config["device_mode"],
        "tiling_strategy": tiling_strategy,
        "patch_size": 8,
        "receptive_field_size": 2,
    }

    results = run_distributed_test(
        _test_multiprocess_processor_worker, device_config, test_args
    )

    # All ranks should have the same result (due to all_reduce)
    norms = [r["result_norm"] for r in results]
    assert all(abs(n - norms[0]) < 1e-4 for n in norms), f"Norms differ: {norms}"


# =============================================================================
# Distributed Data Fidelity Tests
# =============================================================================


def test_distribute_data_fidelity_single(device_config):
    """Test data fidelity distribution in single-process mode."""
    if device_config.get("skip_reason"):
        pytest.skip(device_config["skip_reason"])

    with DistributedContext(device_mode=device_config["device_mode"]) as ctx:
        num_operators = 3
        physics_list = create_test_physics_list(ctx.device, num_operators)
        distributed_physics = distribute(physics_list, ctx=ctx)

        def fidelity_factory(idx, device, shared):
            return L2()

        distributed_fidelity = DistributedDataFidelity(
            ctx, fidelity_factory, num_operators=num_operators
        )

        x = torch.randn(1, 1, 16, 16, device=ctx.device)
        y = distributed_physics.A(x)

        # Test fn() and grad()
        fid = distributed_fidelity.fn(x, y, distributed_physics)
        grad = distributed_fidelity.grad(x, y, distributed_physics)

        # fid should be a scalar or tensor with minimal dimensions
        assert torch.is_tensor(fid)
        assert grad.shape == x.shape


# =============================================================================
# Advanced Operations Tests (A_dagger, compute_norm, etc.)
# =============================================================================


@pytest.mark.parametrize("gather_strategy", ["concatenated"])
def test_compute_norm_single(device_config, gather_strategy):
    """Test compute_norm in single-process mode."""
    if device_config.get("skip_reason"):
        pytest.skip(device_config["skip_reason"])

    with DistributedContext(device_mode=device_config["device_mode"]) as ctx:
        physics_list = create_test_physics_list(ctx.device, 3)
        distributed_physics = distribute(
            physics_list, ctx=ctx, gather_strategy=gather_strategy
        )

        x0 = torch.randn(1, 1, 16, 16, device=ctx.device)
        norm = distributed_physics.compute_sqnorm(x0, max_iter=10, verbose=False)

        assert norm.ndim == 0
        assert norm.item() > 0


def test_a_dagger_single(device_config):
    """Test A_dagger in single-process mode."""
    if device_config.get("skip_reason"):
        pytest.skip(device_config["skip_reason"])

    with DistributedContext(device_mode=device_config["device_mode"]) as ctx:
        physics_list = create_test_physics_list(ctx.device, 3)
        distributed_physics = distribute(physics_list, ctx=ctx)

        x = torch.randn(1, 1, 16, 16, device=ctx.device)
        y = distributed_physics.A(x)

        x_dagger = distributed_physics.A_dagger(y)

        assert x_dagger.shape == x.shape


# =============================================================================
# Reduce Parameter Tests
# =============================================================================


def test_reduce_false_operations(device_config):
    """Test operations with reduce=False parameter."""
    if device_config.get("skip_reason"):
        pytest.skip(device_config["skip_reason"])

    with DistributedContext(device_mode=device_config["device_mode"]) as ctx:
        physics_list = create_test_physics_list(ctx.device, 3)
        distributed_physics = distribute(physics_list, ctx=ctx)

        x = torch.randn(1, 1, 16, 16, device=ctx.device)

        # Test A with reduce=False
        y_local = distributed_physics.A(x, reduce=False)
        assert isinstance(y_local, list)

        # Test A_adjoint with reduce=False
        y_full = distributed_physics.A(x, reduce=True)
        x_adj_local = distributed_physics.A_adjoint(y_full, reduce=False)
        assert torch.is_tensor(x_adj_local)


# =============================================================================
# Integration and Consistency Tests
# =============================================================================


def _test_consistency_worker(rank, world_size, args):
    """Worker for consistency test between single and multiprocess."""
    with DistributedContext(device_mode="cpu") as ctx:
        physics_list = create_test_physics_list(ctx.device, 4)
        distributed_physics = distribute(physics_list, ctx=ctx)

        x = args["x"].to(ctx.device)
        y = distributed_physics.A(x)
        x_adj = distributed_physics.A_adjoint(y)

        return {
            "y_norms": [yi.norm().item() for yi in y],
            "x_adj_norm": x_adj.norm().item(),
        }


def test_consistency_single_vs_multiprocess(device_config):
    """Verify that single-process and multi-process give same results."""
    # This test only makes sense for CPU where we can compare easily
    if device_config["device_mode"] != "cpu" or device_config["world_size"] < 2:
        pytest.skip("Test requires CPU multi-process configuration")

    # Single process result
    with DistributedContext(device_mode="cpu") as ctx:
        physics_list = create_test_physics_list(ctx.device, 4)
        distributed_physics = distribute(physics_list, ctx=ctx)

        torch.manual_seed(42)
        x = torch.randn(1, 1, 16, 16, device=ctx.device)
        y_single = distributed_physics.A(x)
        x_adj_single = distributed_physics.A_adjoint(y_single)

    # Multi-process result
    results = run_distributed_test(_test_consistency_worker, device_config, {"x": x})

    # All ranks should produce same result
    y_norms_ref = [yi.norm().item() for yi in y_single]
    for r in results:
        assert all(abs(a - b) < 1e-4 for a, b in zip(r["y_norms"], y_norms_ref))
        assert abs(r["x_adj_norm"] - x_adj_single.norm().item()) < 1e-4


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


def test_empty_local_set(device_config):
    """Test handling when some ranks have no operators to process."""
    if device_config.get("skip_reason"):
        pytest.skip(device_config["skip_reason"])

    with DistributedContext(device_mode=device_config["device_mode"]) as ctx:
        # Create fewer operators than potential ranks to test empty sets
        num_operators = 1
        physics_list = create_test_physics_list(ctx.device, num_operators)
        distributed_physics = distribute(physics_list, ctx=ctx)

        x = torch.randn(1, 1, 16, 16, device=ctx.device)

        # Should handle empty local sets gracefully
        y = distributed_physics.A(x)
        x_adj = distributed_physics.A_adjoint(y)

        assert len(y) == num_operators
        assert x_adj.shape == x.shape


def test_gather_strategy_validation():
    """Test that invalid gather strategies are rejected."""
    with DistributedContext(device_mode="cpu") as ctx:
        physics_list = create_test_physics_list(ctx.device, 3)

        with pytest.raises(ValueError, match="gather_strategy"):
            distribute(physics_list, ctx=ctx, gather_strategy="invalid_strategy")


def test_distribute_auto_type_detection():
    """Test automatic type detection in distribute()."""
    with DistributedContext(device_mode="cpu") as ctx:
        # Physics list
        physics_list = create_test_physics_list(ctx.device, 3)
        dist_phys = distribute(physics_list, ctx=ctx, type_object="auto")
        assert isinstance(dist_phys, DistributedLinearPhysics)

        # Denoiser
        denoiser = SimpleDenoiser()
        dist_den = distribute(denoiser, ctx=ctx, type_object="auto")
        assert isinstance(dist_den, DistributedProcessing)


# =============================================================================
# Adjoint Operations Tests
# =============================================================================


@pytest.mark.parametrize("gather_strategy", ["concatenated"])
def test_adjoint_operations(device_config, gather_strategy):
    """Test A_adjoint, A_vjp, A_adjoint_A, A_A_adjoint operations."""
    if device_config.get("skip_reason"):
        pytest.skip(device_config["skip_reason"])

    with DistributedContext(device_mode=device_config["device_mode"]) as ctx:
        physics_list = create_test_physics_list(ctx.device, 3)
        distributed_physics = distribute(
            physics_list, ctx=ctx, gather_strategy=gather_strategy
        )

        x = torch.randn(1, 1, 16, 16, device=ctx.device)

        # Test A and A_adjoint
        y = distributed_physics.A(x)
        x_adj = distributed_physics.A_adjoint(y)
        assert x_adj.shape == x.shape

        # Test A_vjp
        v = y  # Use y as cotangent vector
        x_vjp = distributed_physics.A_vjp(x, v)
        assert x_vjp.shape == x.shape
        # A_vjp should equal A_adjoint for LinearPhysics
        assert torch.allclose(x_vjp, x_adj, atol=1e-5)

        # Test A_adjoint_A
        x_ata = distributed_physics.A_adjoint_A(x)
        assert x_ata.shape == x.shape

        # Test A_A_adjoint
        y_aat = distributed_physics.A_A_adjoint(y)
        # A_A_adjoint may return a tensor (not TensorList) depending on operator
        assert torch.is_tensor(y_aat)


# =============================================================================
# Advanced Solver Tests (local_only parameter)
# =============================================================================


@pytest.mark.parametrize("local_only", [True, False])
def test_compute_norm_local_vs_global(device_config, local_only):
    """Test compute_norm with local_only parameter."""
    if device_config.get("skip_reason"):
        pytest.skip(device_config["skip_reason"])

    # Skip expensive global computation in CI
    if not local_only and device_config["world_size"] > 1:
        pytest.skip(
            "Global norm computation is expensive, testing local_only=True only"
        )

    with DistributedContext(device_mode=device_config["device_mode"]) as ctx:
        physics_list = create_test_physics_list(ctx.device, 3)
        distributed_physics = distribute(physics_list, ctx=ctx)

        x0 = torch.randn(1, 1, 16, 16, device=ctx.device)
        norm = distributed_physics.compute_sqnorm(
            x0, max_iter=10, verbose=False, local_only=local_only
        )

        assert norm.ndim == 0
        assert norm.item() > 0


@pytest.mark.parametrize("local_only", [True, False])
def test_a_dagger_local_vs_global(device_config, local_only):
    """Test A_dagger with local_only parameter."""
    if device_config.get("skip_reason"):
        pytest.skip(device_config["skip_reason"])

    # Skip expensive global computation in CI
    if not local_only:
        pytest.skip(
            "Global A_dagger computation is expensive, testing local_only=True only"
        )

    with DistributedContext(device_mode=device_config["device_mode"]) as ctx:
        physics_list = create_test_physics_list(ctx.device, 3)
        distributed_physics = distribute(physics_list, ctx=ctx)

        x = torch.randn(1, 1, 16, 16, device=ctx.device)
        y = distributed_physics.A(x)

        x_dagger = distributed_physics.A_dagger(y, local_only=local_only)

        assert x_dagger.shape == x.shape


# =============================================================================
# DistributedContext Tests
# =============================================================================


def test_distributed_context_device_modes():
    """Test DistributedContext with different device modes."""
    # Test CPU mode
    with DistributedContext(device_mode="cpu") as ctx:
        assert ctx.device.type == "cpu"

    # Test auto mode (should select available device)
    with DistributedContext(device_mode=None) as ctx:
        assert ctx.device is not None

    # Test GPU mode only if available
    if torch.cuda.is_available():
        with DistributedContext(device_mode="gpu") as ctx:
            assert ctx.device.type == "cuda"


def test_distributed_context_seeding():
    """Test that seeding works correctly across ranks."""
    with DistributedContext(device_mode="cpu", seed=42) as ctx:
        val1 = torch.rand(1).item()

    with DistributedContext(device_mode="cpu", seed=42) as ctx:
        val2 = torch.rand(1).item()

    assert abs(val1 - val2) < 1e-7


def test_distributed_context_local_indices():
    """Test local_indices sharding."""
    with DistributedContext(device_mode="cpu") as ctx:
        # Single process should get all indices
        indices = ctx.local_indices(10)
        assert len(indices) == 10
        assert indices == list(range(10))


def test_distributed_context_collectives():
    """Test collective operations (all_reduce, broadcast, barrier)."""
    with DistributedContext(device_mode="cpu") as ctx:
        # Test all_reduce_
        t = torch.tensor([1.0, 2.0, 3.0])
        result = ctx.all_reduce_(t.clone(), op="sum")
        assert torch.allclose(result, t)  # Single process: no change

        result_mean = ctx.all_reduce_(t.clone(), op="mean")
        assert torch.allclose(result_mean, t)

        # Test broadcast_
        t_bcast = torch.tensor([1.0, 2.0, 3.0])
        result_bcast = ctx.broadcast_(t_bcast.clone(), src=0)
        assert torch.allclose(result_bcast, t_bcast)

        # Test barrier (should not hang)
        ctx.barrier()


# =============================================================================
# Processor Batch Size and Tiling Tests
# =============================================================================


def test_processor_different_patch_sizes(device_config):
    """Test processor distribution with varying patch sizes."""
    if device_config.get("skip_reason"):
        pytest.skip(device_config["skip_reason"])

    with DistributedContext(device_mode=device_config["device_mode"]) as ctx:
        processor = SimpleDenoiser(scale=0.9).to(ctx.device)

        for patch_size in [4, 8, 16]:
            distributed_processor = distribute(
                processor,
                ctx,
                type_object="denoiser",
                tiling_strategy="smart_tiling",
                patch_size=patch_size,
                receptive_field_size=2,
            )

            x = torch.randn(1, 3, 16, 16, device=ctx.device)
            result = distributed_processor(x)
            assert result.shape == x.shape


def test_processor_max_batch_size(device_config):
    """Test processor with different max_batch_size settings."""
    if device_config.get("skip_reason"):
        pytest.skip(device_config["skip_reason"])

    with DistributedContext(device_mode=device_config["device_mode"]) as ctx:
        processor = SimpleDenoiser(scale=0.9).to(ctx.device)

        x = torch.randn(1, 3, 16, 16, device=ctx.device)

        # Test with different batch sizes
        results = []
        for max_batch in [None, 1, 4]:
            distributed_processor = distribute(
                processor,
                ctx,
                type_object="denoiser",
                tiling_strategy="smart_tiling",
                patch_size=8,
                receptive_field_size=2,
                max_batch_size=max_batch,
            )
            result = distributed_processor(x)
            results.append(result)

        # All should give same result
        for r in results[1:]:
            assert torch.allclose(results[0], r, atol=1e-5)


def test_processor_3d(device_config):
    """Test processor with 3D volumes."""
    if device_config.get("skip_reason"):
        pytest.skip(device_config["skip_reason"])

    with DistributedContext(device_mode=device_config["device_mode"]) as ctx:
        processor = SimpleDenoiser(scale=0.9).to(ctx.device)

        distributed_processor = distribute(
            processor,
            ctx,
            type_object="denoiser",
            tiling_strategy="smart_tiling",
            patch_size=8,
            receptive_field_size=2,
        )

        # Test with 3D input
        x_3d = torch.randn(1, 1, 16, 16, 16, device=ctx.device)
        result_3d = distributed_processor(x_3d)

        assert result_3d.shape == x_3d.shape


# =============================================================================
# Data Fidelity Comprehensive Tests
# =============================================================================


def test_data_fidelity_vs_stacked(device_config):
    """Compare DistributedDataFidelity with StackedPhysicsDataFidelity."""
    if device_config.get("skip_reason"):
        pytest.skip(device_config["skip_reason"])

    with DistributedContext(device_mode=device_config["device_mode"]) as ctx:
        num_operators = 3
        physics_list = create_test_physics_list(ctx.device, num_operators)

        # Create distributed version
        distributed_physics = distribute(physics_list, ctx=ctx)

        def fidelity_factory(idx, device, shared):
            return L2()

        distributed_fidelity = DistributedDataFidelity(
            ctx, fidelity_factory, num_operators=num_operators
        )

        # Create stacked version for comparison
        stacked_physics = StackedLinearPhysics(physics_list)
        stacked_fidelity = StackedPhysicsDataFidelity(
            [L2() for _ in range(num_operators)]
        )

        # Test
        x = torch.randn(1, 1, 16, 16, device=ctx.device)
        y_dist = distributed_physics.A(x)
        y_stack = stacked_physics.A(x)

        # Compare fidelity values
        fid_dist = distributed_fidelity.fn(x, y_dist, distributed_physics)
        fid_stack = stacked_fidelity.fn(x, y_stack, stacked_physics)

        assert torch.allclose(fid_dist, fid_stack, atol=1e-5)

        # Compare gradients
        grad_dist = distributed_fidelity.grad(x, y_dist, distributed_physics)
        grad_stack = stacked_fidelity.grad(x, y_stack, stacked_physics)

        assert torch.allclose(grad_dist, grad_stack, atol=1e-5)


def test_data_fidelity_different_fidelities(device_config):
    """Test with different data fidelity types (L1, L2)."""
    if device_config.get("skip_reason"):
        pytest.skip(device_config["skip_reason"])

    with DistributedContext(device_mode=device_config["device_mode"]) as ctx:
        num_operators = 3
        physics_list = create_test_physics_list(ctx.device, num_operators)
        distributed_physics = distribute(physics_list, ctx=ctx)

        # Test L1 and L2
        for FidelityClass in [L1, L2]:

            def fidelity_factory(idx, device, shared):
                return FidelityClass()

            distributed_fidelity = DistributedDataFidelity(
                ctx, fidelity_factory, num_operators=num_operators
            )

            x = torch.randn(1, 1, 16, 16, device=ctx.device)
            y = distributed_physics.A(x)

            fid = distributed_fidelity.fn(x, y, distributed_physics)
            grad = distributed_fidelity.grad(x, y, distributed_physics)

            assert torch.is_tensor(fid)
            assert grad.shape == x.shape


# =============================================================================
# Reduce=False Comprehensive Tests
# =============================================================================


@pytest.mark.parametrize(
    "operation", ["A", "forward", "A_adjoint", "A_vjp", "A_adjoint_A"]
)
def test_reduce_false_physics_operations(device_config, operation):
    """Test all physics operations with reduce=False."""
    if device_config.get("skip_reason"):
        pytest.skip(device_config["skip_reason"])

    with DistributedContext(device_mode=device_config["device_mode"]) as ctx:
        physics_list = create_test_physics_list(ctx.device, 3)
        distributed_physics = distribute(physics_list, ctx=ctx)

        x = torch.randn(1, 1, 16, 16, device=ctx.device)

        if operation == "A":
            result_local = distributed_physics.A(x, reduce=False)
            result_global = distributed_physics.A(x, reduce=True)
            assert isinstance(result_local, list)
            assert isinstance(result_global, TensorList)

        elif operation == "forward":
            result_local = distributed_physics.forward(x, reduce=False)
            result_global = distributed_physics.forward(x, reduce=True)
            assert isinstance(result_local, list)
            assert isinstance(result_global, TensorList)

        elif operation == "A_adjoint":
            y = distributed_physics.A(x)
            result_local = distributed_physics.A_adjoint(y, reduce=False)
            result_global = distributed_physics.A_adjoint(y, reduce=True)
            assert torch.is_tensor(result_local)
            assert torch.is_tensor(result_global)

        elif operation == "A_vjp":
            y = distributed_physics.A(x)
            result_local = distributed_physics.A_vjp(x, y, reduce=False)
            result_global = distributed_physics.A_vjp(x, y, reduce=True)
            assert torch.is_tensor(result_local)
            assert torch.is_tensor(result_global)

        elif operation == "A_adjoint_A":
            result_local = distributed_physics.A_adjoint_A(x, reduce=False)
            result_global = distributed_physics.A_adjoint_A(x, reduce=True)
            assert torch.is_tensor(result_local)
            assert torch.is_tensor(result_global)


def test_reduce_false_processor(device_config):
    """Test DistributedProcessing with reduce=False."""
    if device_config.get("skip_reason"):
        pytest.skip(device_config["skip_reason"])

    with DistributedContext(device_mode=device_config["device_mode"]) as ctx:
        processor = SimpleDenoiser(scale=0.9).to(ctx.device)
        distributed_processor = distribute(
            processor,
            ctx,
            type_object="denoiser",
            tiling_strategy="smart_tiling",
            patch_size=8,
            receptive_field_size=2,
        )

        x = torch.randn(1, 3, 16, 16, device=ctx.device)

        result_local = distributed_processor(x, reduce=False)
        result_global = distributed_processor(x, reduce=True)

        assert torch.is_tensor(result_local)
        assert torch.is_tensor(result_global)


def test_reduce_false_data_fidelity(device_config):
    """Test DistributedDataFidelity with reduce=False."""
    if device_config.get("skip_reason"):
        pytest.skip(device_config["skip_reason"])

    with DistributedContext(device_mode=device_config["device_mode"]) as ctx:
        physics_list = create_test_physics_list(ctx.device, 3)
        distributed_physics = distribute(physics_list, ctx=ctx)

        def fidelity_factory(idx, device, shared):
            return L2()

        distributed_fidelity = DistributedDataFidelity(
            ctx, fidelity_factory, num_operators=3
        )

        x = torch.randn(1, 1, 16, 16, device=ctx.device)
        y = distributed_physics.A(x)

        # Test fn with reduce=False
        fid_local = distributed_fidelity.fn(x, y, distributed_physics, reduce=False)
        fid_global = distributed_fidelity.fn(x, y, distributed_physics, reduce=True)
        assert torch.is_tensor(fid_local)
        assert torch.is_tensor(fid_global)

        # Test grad with reduce=False
        grad_local = distributed_fidelity.grad(x, y, distributed_physics, reduce=False)
        grad_global = distributed_fidelity.grad(x, y, distributed_physics, reduce=True)
        assert torch.is_tensor(grad_local)
        assert torch.is_tensor(grad_global)
