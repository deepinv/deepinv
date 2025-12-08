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
from deepinv.models.drunet import DRUNet
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


@pytest.fixture(params=["naive", "concatenated", "broadcast"])
def gather_strategy(request):
    """Parameterized fixture for gather strategy."""
    return request.param


@pytest.fixture(params=["cpu_single", "cpu_multi", "gpu_single", "gpu_multi"])
def device_config(request):
    """
    Parameterized fixture for device configuration.

    Returns:
        dict with keys:
            - device_mode: "cpu_single", "cpu_multi", "gpu_single", or "gpu_multi"
            - world_size: number of processes to spawn
            - skip_reason: reason to skip if device not available
    """
    mode = request.param

    if mode == "cpu_single":
        return {
            "device_mode": "cpu",
            "world_size": 1,
            "skip_reason": None,
        }
    elif mode == "cpu_multi":
        return {
            "device_mode": "cpu",
            "world_size": 2,
            "skip_reason": None,
        }
    elif mode == "gpu_single":
        gpu_count = _get_gpu_count()
        if gpu_count == 0:
            return {
                "device_mode": "gpu",
                "world_size": 0,
                "skip_reason": "No GPUs available",
            }
        elif gpu_count > 0:
            return {
                "device_mode": "gpu",
                "world_size": 1,
                "skip_reason": None,
            }
    elif mode == "gpu_multi":
        gpu_count = _get_gpu_count()
        if gpu_count < 2:
            return {
                "device_mode": "gpu",
                "world_size": 0,
                "skip_reason": "Less than 2 GPUs available",
            }
        else:
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

    # For GPU mode, set the right device for each rank
    if dist_config["device_mode"] == "gpu":
        torch.cuda.set_device(rank)

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
        "master_addr": "127.0.0.1",
        "master_port": str(_get_free_port()),
        "device_mode": device_mode,
    }

    # Keep thread pools small for spawn safety
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

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

    def __init__(self, scale=0.9):
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


def create_drunet_denoiser(
    num_channels=1, device="cpu", dtype=torch.float32
) -> Denoiser:
    """Create a DRUNet denoiser appropriate for the given ground truth shape.

    Automatically detects whether to use:
    - Grayscale (1 channel) or color (3 channels) based on channel dimension
    - 2D or 3D based on number of spatial dimensions

    Parameters
    ----------
    num_channels : int
    device : str or torch.device, optional
        Device to load the model on. Default: 'cpu'.
    dtype : torch.dtype, optional
        Data type for the model. Default: torch.float32.

    Returns
    -------
    DRUNet
        Configured DRUNet denoiser model.
    """

    # Determine if grayscale or color
    if num_channels == 1:
        # Grayscale: use single-channel DRUNet
        model = DRUNet(in_channels=1, out_channels=1, device=device)
    elif num_channels == 3:
        # Color: use default RGB DRUNet
        model = DRUNet(device=device)
    else:
        raise ValueError(
            f"Unsupported number of channels: {num_channels}. Expected 1 (grayscale) or 3 (color)."
        )

    # Move to device and dtype
    return model.to(dtype)


def create_denoiser(spec_type, device, num_channels=1) -> Denoiser:
    """
    Create denoiser based on specification type.

    Args:
        spec_type: "simple" or "callable_factory"
        device: torch device

    Returns:
        denoiser or denoiser factory
    """
    if spec_type == "simple":
        return SimpleDenoiser(scale=0.9).to(device)

    elif spec_type == "drunet":
        return create_drunet_denoiser(num_channels=num_channels, device=device)

    else:
        raise ValueError(f"Unknown denoiser spec_type: {spec_type}")


# =============================================================================
# Core Tests: Distributed Physics Operations
# =============================================================================


def _test_multiprocess_physics_worker(rank, world_size, args):
    """Worker for multi-process physics tests."""
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
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
            assert result.shape == x.shape
        else:
            result = distributed_physics.A(x)

            # Verify structure
            assert len(result) == args["num_operators"]
            assert all(yi.device == ctx.device for yi in result)

            # Compare with reference
            reference = StackedLinearPhysics(
                create_test_physics_list(ctx.device, args["num_operators"])
            )
            y_ref = reference.A(x)

            for i in range(args["num_operators"]):
                assert torch.allclose(result[i], y_ref[i], atol=1e-5)

        # Return success indicator
        return "success"


@pytest.mark.parametrize(
    "physics_spec", ["physics_list", "stacked_physics", "callable_factory"]
)
def test_distribute_physics(device_config, gather_strategy, physics_spec):
    """Test physics distribution in multi-process mode."""
    # Skip naive strategy with multi-GPU (NCCL doesn't support all_gather_object)
    if gather_strategy == "naive" and device_config["device_mode"] == "gpu":
        pytest.skip("Naive gather strategy not supported with NCCL backend")

    torch.manual_seed(42)
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


def _test_multiprocess_processor_worker(rank, world_size, args):
    """Worker for multi-process processor tests."""
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        processor = create_denoiser(args["denoiser_spec"], ctx.device, num_channels=3)

        distributed_processor = distribute(
            processor,
            ctx,
            type_object="denoiser",
            tiling_strategy=args["tiling_strategy"],
            patch_size=args["patch_size"],
            receptive_field_size=args["receptive_field_size"],
        )

        x = torch.randn(1, 3, 16, 16, device=ctx.device)
        with torch.no_grad():
            result = distributed_processor(x, sigma=0.1)

        assert result.shape == x.shape
        assert result.device == ctx.device

        return {"result_norm": result.norm().item(), "rank": rank}


@pytest.mark.parametrize("tiling_strategy", ["basic", "smart_tiling"])
@pytest.mark.parametrize("denoiser_spec", ["simple", "drunet"])
def test_distribute_processor(device_config, tiling_strategy, denoiser_spec):
    """Test processor distribution in multi-process mode."""
    test_args = {
        "device_mode": device_config["device_mode"],
        "tiling_strategy": tiling_strategy,
        "patch_size": 8,
        "receptive_field_size": 2,
        "denoiser_spec": denoiser_spec,
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


def _test_data_fidelity_worker(rank, world_size, args):
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        num_operators = 3
        physics_list = create_test_physics_list(ctx.device, num_operators)
        distributed_physics = distribute(physics_list, ctx=ctx)

        # 1. Test factory method
        def fidelity_factory(idx, device, shared):
            return L2()

        distributed_fidelity_factory = distribute(
            fidelity_factory,
            ctx=ctx,
            num_operators=num_operators,
            type_object="data_fidelity",
        )

        assert isinstance(distributed_fidelity_factory, DistributedDataFidelity)

        x = torch.randn(1, 1, 16, 16, device=ctx.device)
        y = distributed_physics.A(x)

        # Test fn() and grad()
        fid_factory = distributed_fidelity_factory.fn(x, y, distributed_physics)
        grad_factory = distributed_fidelity_factory.grad(x, y, distributed_physics)

        # fid should be a scalar or tensor with minimal dimensions
        assert torch.is_tensor(fid_factory)
        assert grad_factory.shape == x.shape

        # 2. Test single object method
        single_fidelity = L2()
        distributed_fidelity_single = distribute(single_fidelity, ctx=ctx)

        fid_single = distributed_fidelity_single.fn(x, y, distributed_physics)
        grad_single = distributed_fidelity_single.grad(x, y, distributed_physics)

        assert torch.is_tensor(fid_single)
        assert grad_single.shape == x.shape
        assert torch.allclose(fid_factory, fid_single)
        assert torch.allclose(grad_factory, grad_single)

        return "success"


def test_distribute_data_fidelity(device_config):
    """Test data fidelity distribution."""
    test_args = {
        "device_mode": device_config["device_mode"],
    }
    results = run_distributed_test(_test_data_fidelity_worker, device_config, test_args)
    assert all(r == "success" for r in results)


# =============================================================================
# Advanced Operations Tests (A_dagger, compute_norm, etc.)
# =============================================================================


def _test_compute_norm_worker(rank, world_size, args):
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        physics_list = create_test_physics_list(ctx.device, 3)
        distributed_physics = distribute(
            physics_list, ctx=ctx, gather_strategy=args["gather_strategy"]
        )

        x0 = torch.randn(1, 1, 16, 16, device=ctx.device)
        norm = distributed_physics.compute_sqnorm(x0, max_iter=10, verbose=False)

        assert norm.ndim == 0
        assert norm.item() > 0
        return "success"


def test_compute_norm(device_config, gather_strategy):
    """Test compute_norm."""
    # Skip naive strategy with multi-GPU (NCCL doesn't support all_gather_object)
    if gather_strategy == "naive" and device_config["device_mode"] == "gpu_multi":
        pytest.skip("Naive gather strategy not supported with NCCL backend (gpu_multi)")

    test_args = {
        "device_mode": device_config["device_mode"],
        "gather_strategy": gather_strategy,
    }
    results = run_distributed_test(_test_compute_norm_worker, device_config, test_args)
    assert all(r == "success" for r in results)


def _test_a_dagger_worker(rank, world_size, args):
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        physics_list = create_test_physics_list(ctx.device, 3)
        distributed_physics = distribute(physics_list, ctx=ctx)

        x = torch.randn(1, 1, 16, 16, device=ctx.device)
        y = distributed_physics.A(x)

        x_dagger = distributed_physics.A_dagger(y)

        assert x_dagger.shape == x.shape
        return "success"


def test_a_dagger(device_config):
    """Test A_dagger."""
    test_args = {
        "device_mode": device_config["device_mode"],
    }
    results = run_distributed_test(_test_a_dagger_worker, device_config, test_args)
    assert all(r == "success" for r in results)


# =============================================================================
# Reduce Parameter Tests
# =============================================================================


def _test_reduce_false_worker(rank, world_size, args):
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
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
        return "success"


def test_reduce_false_operations(device_config):
    """Test operations with reduce=False parameter."""
    test_args = {"device_mode": device_config["device_mode"]}
    results = run_distributed_test(_test_reduce_false_worker, device_config, test_args)
    assert all(r == "success" for r in results)


# =============================================================================
# Integration and Consistency Tests
# =============================================================================


def _test_consistency_worker(rank, world_size, args):
    """Worker for consistency test between single and multiprocess."""
    with DistributedContext(device_mode="cpu") as ctx:
        physics_list = create_test_physics_list(ctx.device, 4)
        distributed_physics = distribute(physics_list, ctx=ctx)

        torch.manual_seed(42)
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
    with DistributedContext(device_mode="cpu", seed=42) as ctx:
        physics_list = create_test_physics_list(ctx.device, 4)
        distributed_physics = distribute(physics_list, ctx=ctx)

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


def _test_empty_local_set_worker(rank, world_size, args):
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        # Test with fewer operators than ranks - this creates the edge case
        # where rank 1 (or higher ranks) will have 0 operators
        num_operators = 1
        physics_list = create_test_physics_list(ctx.device, num_operators)
        distributed_physics = distribute(physics_list, ctx=ctx)

        x = torch.randn(1, 1, 16, 16, device=ctx.device)

        # Should handle empty local sets gracefully
        # This tests the critical edge case where some ranks have no operators
        y = distributed_physics.A(x)
        x_adj = distributed_physics.A_adjoint(y)

        assert len(y) == num_operators
        assert x_adj.shape == x.shape
        return "success"


def test_empty_local_set(device_config):
    """Test handling when some ranks have no operators to process."""
    test_args = {"device_mode": device_config["device_mode"]}
    results = run_distributed_test(
        _test_empty_local_set_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


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


def _test_adjoint_operations_worker(rank, world_size, args):
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        physics_list = create_test_physics_list(ctx.device, 3)
        distributed_physics = distribute(
            physics_list, ctx=ctx, gather_strategy=args["gather_strategy"]
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
        return "success"


def test_adjoint_operations(device_config, gather_strategy):
    """Test A_adjoint, A_vjp, A_adjoint_A, A_A_adjoint operations."""
    # Skip naive strategy with multi-GPU (NCCL doesn't support all_gather_object)
    if gather_strategy == "naive" and device_config["device_mode"] == "gpu":
        pytest.skip("Naive gather strategy not supported with NCCL backend (gpu_multi)")

    test_args = {
        "device_mode": device_config["device_mode"],
        "gather_strategy": gather_strategy,
    }
    results = run_distributed_test(
        _test_adjoint_operations_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


# =============================================================================
# Advanced Solver Tests (local_only parameter)
# =============================================================================


def _test_compute_norm_local_vs_global_worker(rank, world_size, args):
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        physics_list = create_test_physics_list(ctx.device, 3)
        distributed_physics = distribute(physics_list, ctx=ctx)

        x0 = torch.randn(1, 1, 16, 16, device=ctx.device)
        norm = distributed_physics.compute_sqnorm(
            x0, max_iter=10, verbose=False, local_only=args["local_only"]
        )

        assert norm.ndim == 0
        assert norm.item() > 0
        return "success"


@pytest.mark.parametrize("local_only", [True, False])
def test_compute_norm_local_vs_global(device_config, local_only):
    """Test compute_norm with local_only parameter."""

    test_args = {
        "device_mode": device_config["device_mode"],
        "local_only": local_only,
    }
    results = run_distributed_test(
        _test_compute_norm_local_vs_global_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


def _test_a_dagger_local_vs_global_worker(rank, world_size, args):
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        physics_list = create_test_physics_list(ctx.device, 3)
        distributed_physics = distribute(physics_list, ctx=ctx)

        x = torch.randn(1, 1, 16, 16, device=ctx.device)
        y = distributed_physics.A(x)

        x_dagger = distributed_physics.A_dagger(y, local_only=args["local_only"])

        assert x_dagger.shape == x.shape
        return "success"


@pytest.mark.parametrize("local_only", [True, False])
def test_a_dagger_local_vs_global(device_config, local_only):
    """Test A_dagger with local_only parameter."""

    test_args = {
        "device_mode": device_config["device_mode"],
        "local_only": local_only,
    }
    results = run_distributed_test(
        _test_a_dagger_local_vs_global_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


# =============================================================================
# DistributedContext Tests
# =============================================================================


def _test_distributed_context_device_modes_worker(rank, world_size, args):
    """Worker for testing DistributedContext with different device modes."""
    device_mode = args["device_mode"]
    with DistributedContext(device_mode=device_mode) as ctx:
        if device_mode == "cpu":
            assert ctx.device.type == "cpu"
        elif device_mode == "gpu":
            assert ctx.device.type == "cuda"
        elif device_mode is None:
            assert ctx.device is not None
    return "success"


def test_distributed_context_device_modes(device_config):
    """Test DistributedContext with different device modes."""
    test_args = {"device_mode": device_config["device_mode"]}
    results = run_distributed_test(
        _test_distributed_context_device_modes_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


def _test_distributed_context_local_indices_worker(rank, world_size, args):
    """Worker for testing local_indices sharding."""
    with DistributedContext(device_mode=args["device_mode"]) as ctx:
        total_items = 10
        indices = ctx.local_indices(total_items)
        # Each rank gets a subset of indices
        assert len(indices) <= total_items
        # Indices should be in range
        assert all(0 <= i < total_items for i in indices)
    return {"rank": rank, "indices": indices, "num_indices": len(indices)}


def test_distributed_context_local_indices(device_config):
    """Test local_indices sharding across ranks."""
    test_args = {"device_mode": device_config["device_mode"]}
    results = run_distributed_test(
        _test_distributed_context_local_indices_worker, device_config, test_args
    )

    # Check that all indices are covered and non-overlapping
    all_indices = []
    for r in results:
        all_indices.extend(r["indices"])

    # Should cover all items from 0 to 9
    if device_config["world_size"] == 1:
        assert len(all_indices) == 10
    else:
        # In multi-process, all indices should be covered
        assert sorted(all_indices) == list(range(10))


def _test_distributed_context_collectives_worker(rank, world_size, args):
    """Worker for testing collective operations."""
    with DistributedContext(device_mode=args["device_mode"]) as ctx:
        # Test all_reduce_ with sum
        t = torch.tensor([float(rank + 1), 2.0, 3.0], device=ctx.device)
        result_sum = ctx.all_reduce_(t.clone(), op="sum")
        # Sum across all ranks: rank 0 contributes 1, rank 1 contributes 2, etc.
        expected_sum = torch.tensor(
            [sum(range(1, world_size + 1)), 2.0 * world_size, 3.0 * world_size],
            device=ctx.device,
        )
        assert torch.allclose(
            result_sum, expected_sum
        ), f"Rank {rank}: {result_sum} vs {expected_sum}"

        # Test all_reduce_ with mean
        result_mean = ctx.all_reduce_(t.clone(), op="mean")
        expected_mean = expected_sum / world_size
        assert torch.allclose(
            result_mean, expected_mean
        ), f"Rank {rank}: {result_mean} vs {expected_mean}"

        # Test broadcast_
        if rank == 0:
            t_bcast = torch.tensor([10.0, 20.0, 30.0], device=ctx.device)
        else:
            t_bcast = torch.tensor([0.0, 0.0, 0.0], device=ctx.device)

        result_bcast = ctx.broadcast_(t_bcast.clone(), src=0)
        expected_bcast = torch.tensor([10.0, 20.0, 30.0], device=ctx.device)
        assert torch.allclose(result_bcast, expected_bcast)

    return "success"


def test_distributed_context_collectives(device_config):
    """Test collective operations (all_reduce, broadcast, barrier) across ranks."""
    test_args = {"device_mode": device_config["device_mode"]}
    results = run_distributed_test(
        _test_distributed_context_collectives_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


# =============================================================================
# Processor Batch Size and Tiling Tests
# =============================================================================


def _test_processor_different_patch_sizes_worker(rank, world_size, args):
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        processor = create_denoiser(args["denoiser_spec"], ctx.device, num_channels=3)

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
            result = distributed_processor(x, sigma=0.1)
            assert result.shape == x.shape
        return "success"


@pytest.mark.parametrize("denoiser_spec", ["simple"])
def test_processor_different_patch_sizes(device_config, denoiser_spec):
    """Test processor distribution with varying patch sizes."""
    test_args = {
        "device_mode": device_config["device_mode"],
        "denoiser_spec": denoiser_spec,
    }
    results = run_distributed_test(
        _test_processor_different_patch_sizes_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


def _test_processor_max_batch_size_worker(rank, world_size, args):
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        processor = create_denoiser(args["denoiser_spec"], ctx.device, num_channels=3)

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
            result = distributed_processor(x, sigma=0.1)
            results.append(result)

        # All should give same result
        for r in results[1:]:
            assert torch.allclose(results[0], r, atol=1e-5)
        return "success"


@pytest.mark.parametrize("denoiser_spec", ["simple"])
def test_processor_max_batch_size(device_config, denoiser_spec):
    """Test processor with different max_batch_size settings."""
    test_args = {
        "device_mode": device_config["device_mode"],
        "denoiser_spec": denoiser_spec,
    }
    results = run_distributed_test(
        _test_processor_max_batch_size_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


def _test_processor_3d_worker(rank, world_size, args):
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        processor = create_denoiser(args["denoiser_spec"], ctx.device, num_channels=1)

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
        result_3d = distributed_processor(x_3d, sigma=0.1)

        assert result_3d.shape == x_3d.shape
        return "success"


@pytest.mark.parametrize("denoiser_spec", ["simple"])
def test_processor_3d(device_config, denoiser_spec):
    """Test processor with 3D volumes."""
    test_args = {
        "device_mode": device_config["device_mode"],
        "denoiser_spec": denoiser_spec,
    }
    results = run_distributed_test(_test_processor_3d_worker, device_config, test_args)
    assert all(r == "success" for r in results)


# =============================================================================
# Data Fidelity Comprehensive Tests
# =============================================================================


def _test_data_fidelity_vs_stacked_worker(rank, world_size, args):
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        num_operators = 3
        physics_list = create_test_physics_list(ctx.device, num_operators)

        # Create distributed version
        distributed_physics = distribute(physics_list, ctx=ctx)

        distributed_fidelity = distribute(L2(), ctx=ctx, num_operators=num_operators)

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
        return "success"


def test_data_fidelity_vs_stacked(device_config):
    """Compare DistributedDataFidelity with StackedPhysicsDataFidelity."""
    test_args = {"device_mode": device_config["device_mode"]}
    results = run_distributed_test(
        _test_data_fidelity_vs_stacked_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


def _test_data_fidelity_different_fidelities_worker(rank, world_size, args):
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        num_operators = 3
        physics_list = create_test_physics_list(ctx.device, num_operators)
        distributed_physics = distribute(physics_list, ctx=ctx)

        # Test L1 and L2
        for FidelityClass in [L1, L2]:

            distributed_fidelity = distribute(
                FidelityClass(), ctx=ctx, num_operators=num_operators
            )

            x = torch.randn(1, 1, 16, 16, device=ctx.device)
            y = distributed_physics.A(x)

            fid = distributed_fidelity.fn(x, y, distributed_physics)
            grad = distributed_fidelity.grad(x, y, distributed_physics)

            assert torch.is_tensor(fid)
            assert grad.shape == x.shape
    return "success"


def test_data_fidelity_different_fidelities(device_config):
    """Test with different data fidelity types (L1, L2)."""
    test_args = {"device_mode": device_config["device_mode"]}
    results = run_distributed_test(
        _test_data_fidelity_different_fidelities_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


# =============================================================================
# Reduce=False Comprehensive Tests
# =============================================================================


def _test_reduce_false_physics_operations_worker(rank, world_size, args):
    operation = args["operation"]
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
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
            # A_adjoint reduces by default, so we test both modes
            result_global = distributed_physics.A_adjoint(y)
            assert torch.is_tensor(result_global)
            # Local version returns unreduced result (one per local operator)
            result_local = distributed_physics.A_adjoint(y, reduce=False)
            assert isinstance(result_local, list) or torch.is_tensor(result_local)

        elif operation == "A_vjp":
            y = distributed_physics.A(x)
            # A_vjp reduces by default
            result_global = distributed_physics.A_vjp(x, y)
            assert torch.is_tensor(result_global)
            # Local version
            result_local = distributed_physics.A_vjp(x, y, reduce=False)
            assert isinstance(result_local, list) or torch.is_tensor(result_local)

        elif operation == "A_adjoint_A":
            # A_adjoint_A reduces by default
            result_global = distributed_physics.A_adjoint_A(x)
            assert torch.is_tensor(result_global)
            # Local version
            result_local = distributed_physics.A_adjoint_A(x, reduce=False)
            assert isinstance(result_local, list) or torch.is_tensor(result_local)
    return "success"


@pytest.mark.parametrize(
    "operation", ["A", "forward", "A_adjoint", "A_vjp", "A_adjoint_A"]
)
def test_reduce_false_physics_operations(device_config, operation):
    """Test all physics operations with reduce=False."""
    test_args = {"device_mode": device_config["device_mode"], "operation": operation}
    results = run_distributed_test(
        _test_reduce_false_physics_operations_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


def _test_reduce_false_processor_worker(rank, world_size, args):
    denoiser_spec = args["denoiser_spec"]
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        processor = create_denoiser(denoiser_spec, ctx.device, num_channels=3)
        distributed_processor = distribute(
            processor,
            ctx,
            type_object="denoiser",
            tiling_strategy="smart_tiling",
            patch_size=8,
            receptive_field_size=2,
        )

        x = torch.randn(1, 3, 16, 16, device=ctx.device)

        # Test with reduce=True (default)
        result_global = distributed_processor(x, sigma=0.1)
        assert torch.is_tensor(result_global)

        # Test with reduce=False if supported
        result_local = distributed_processor(x, reduce=False, sigma=0.1)
        assert isinstance(result_local, list) or torch.is_tensor(result_local)

    return "success"


@pytest.mark.parametrize("denoiser_spec", ["simple", "drunet"])
def test_reduce_false_processor(device_config, denoiser_spec):
    """Test DistributedProcessing with reduce=False."""
    test_args = {
        "device_mode": device_config["device_mode"],
        "denoiser_spec": denoiser_spec,
    }
    results = run_distributed_test(
        _test_reduce_false_processor_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


def _test_reduce_false_data_fidelity_worker(rank, world_size, args):
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        physics_list = create_test_physics_list(ctx.device, 3)
        distributed_physics = distribute(physics_list, ctx=ctx)

        distributed_fidelity = distribute(L2(), ctx=ctx, num_operators=3)

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
    return "success"


def test_reduce_false_data_fidelity(device_config):
    """Test DistributedDataFidelity with reduce=False."""
    test_args = {"device_mode": device_config["device_mode"]}
    results = run_distributed_test(
        _test_reduce_false_data_fidelity_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


def _test_distribute_helper_data_fidelity_worker(rank, world_size, args):
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        num_operators = 3
        physics_list = create_test_physics_list(ctx.device, num_operators)
        distributed_physics = distribute(physics_list, ctx=ctx)

        fidelity = L2()
        # No num_operators needed
        distributed_fidelity = distribute(fidelity, ctx=ctx)

        x = torch.randn(1, 1, 16, 16, device=ctx.device)
        y = distributed_physics.A(x)

        fid = distributed_fidelity.fn(x, y, distributed_physics)
        assert torch.is_tensor(fid)
    return "success"


def test_distribute_helper_data_fidelity(device_config):
    """Test distribute_data_fidelity helper with single object."""
    test_args = {"device_mode": device_config["device_mode"]}
    results = run_distributed_test(
        _test_distribute_helper_data_fidelity_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)
