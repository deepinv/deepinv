"""
Tests for the distributed framework (distribute function and distributed components).

This module tests the distributed API that enables converting Physics, DataFidelity,
and Denoisers into distributed versions that work across multiple processes and GPUs.

Test Organization:
- Test helpers and fixtures (CPU/GPU support)
- Core distributed physics tests
- Distributed processor (denoiser) tests
- Distributed data fidelity tests
- Advanced operations (A_dagger, compute_norm, gather=False parameter)
- Integration tests

All tests support both CPU-only and multi-GPU configurations automatically.
"""

from __future__ import annotations
import os
import pytest
import torch
import torch.distributed as dist
import time
import socket
import platform
import torch.multiprocessing as mp
from typing import Callable, Any

from deepinv.physics import Blur, GaussianNoise
from deepinv.physics.blur import gaussian_blur
from deepinv.physics.forward import StackedLinearPhysics
from deepinv.utils.tensorlist import TensorList
from deepinv.models.base import Denoiser
from deepinv.models.drunet import DRUNet
from deepinv.optim import L2, L1, PGD
from deepinv.optim.data_fidelity import StackedPhysicsDataFidelity
from deepinv.optim.prior import PnP

from deepinv.distributed.distrib_framework import (
    DistributedContext,
    DistributedStackedLinearPhysics,
    DistributedProcessing,
    DistributedDataFidelity,
)
from deepinv.distributed.distribute import (
    distribute,
    _distribute_base_optim,
)

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


@pytest.fixture(params=[1,2,3])
def num_operators(request):
    """Parameterized fixture for number of operators. Tests edges cases when load is imbalanced across ranks, i.e., when num_operators < world_size and num_operators > world_size """
    return request.param

@pytest.fixture(params=["naive", "concatenated", "broadcast"])
def gather_strategy(request):
    """Parameterized fixture for gather strategy."""
    return request.param


@pytest.fixture(params=["cpu_single", "cpu_multi", "gpu_single", "gpu_multi"])
def device_config(request):
    """
    Parameterized fixture for device configuration.

    :param request: pytest request object.
    :return: Dictionary with keys:
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
        skip_reason = (
            "Gloo backend not supported on Windows for multi-process tests"
            if platform.system() == "Windows"
            else None
        )
        return {
            "device_mode": "cpu",
            "world_size": 2,
            "skip_reason": skip_reason,
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
            skip_reason = (
                "NCCL backend not supported on Windows for multi-GPU tests"
                if platform.system() == "Windows"
                else None
            )
            return {
                "device_mode": "gpu",
                "world_size": min(2, gpu_count),
                "skip_reason": skip_reason,
            }


def _worker(rank, world_size, test_func, test_args, result_queue, dist_config):
    """
    Worker function that runs in each process - must be at module level for pickling.

    Supports both CPU (Gloo) and GPU (NCCL) backends automatically.

    :param int rank: Rank of the current process.
    :param int world_size: Total number of processes.
    :param Callable test_func: Test function to execute.
    :param dict test_args: Arguments to pass to the test function.
    :param Queue result_queue: Queue to store results.
    :param dict dist_config: Distributed configuration dictionary.
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

    :param Callable test_func: Picklable callable with signature (rank: int, world_size: int, args: dict) -> Any.
    :param dict device_config: Device configuration from fixture.
    :param dict | None test_args: Optional dict passed to test_func on each rank. Default is `None`.
    :param float timeout_per_rank: Timeout budget per rank in seconds. Default is `12.0`.
    :return: List of per-rank results ordered by rank.
    :raises RuntimeError: If any rank fails, times out, or does not report a result.
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
    """
    Simple test denoiser that scales the input.

    :param float scale: Scaling factor to apply to input. Default is `0.9`.
    """

    def __init__(self, scale=0.9):
        super().__init__()
        self.scale = scale

    def forward(self, x, sigma=None, *args, **kwargs):
        return x * self.scale


def create_test_physics_list(device, num_operators=3):
    """
    Create simple test physics operators as a list.

    :param torch.device device: Device to create operators on.
    :param int num_operators: Number of operators to create. Default is `3`.
    :return: List of physics operators.
    """
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

    :param str spec_type: Type of specification ("physics_list", "stacked_physics", or "callable_factory").
    :param torch.device device: Device to create operators on.
    :param int num_operators: Number of operators to create.
    :return: Tuple of (physics_spec, needs_num_operators_param).
    """
    if spec_type == "physics_list":
        physics_list = create_test_physics_list(device, num_operators)
        return physics_list, False

    elif spec_type == "stacked_physics":
        physics_list = create_test_physics_list(device, num_operators)
        stacked = StackedLinearPhysics(physics_list)
        return stacked, False

    elif spec_type == "callable_factory":

        def physics_factory(idx, device, factory_kwargs):
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

    :param int num_channels: Number of input channels (1 for grayscale, 3 for color). Default is `1`.
    :param str | torch.device device: Device to load the model on. Default is `'cpu'`.
    :param torch.dtype dtype: Data type for the model. Default is `torch.float32`.
    :return: Configured DRUNet denoiser model.
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

    :param str spec_type: Type of denoiser ("simple" or "drunet").
    :param torch.device device: Device to create denoiser on.
    :param int num_channels: Number of input channels. Default is `1`.
    :return: Denoiser instance or denoiser factory.
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
    """
    Test physics distribution in multi-process mode.

    :param dict device_config: Device configuration from fixture.
    :param str gather_strategy: Gather strategy to use.
    :param str physics_spec: Physics specification type.
    """
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
    }

    results = run_distributed_test(
        _test_multiprocess_physics_worker, device_config, test_args
    )

    assert all(r == "success" for r in results)


# =============================================================================
# Distributed Processor Tests
# =============================================================================


def _test_multiprocess_processor_worker(rank, world_size, args):
    """
    Worker for multi-process processor tests.

    :param int rank: Rank of the current process.
    :param int world_size: Total number of processes.
    :param dict args: Test arguments dictionary.
    :return: Test result tensor.
    """
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        processor = create_denoiser(args["denoiser_spec"], ctx.device, num_channels=3)

        distributed_processor = distribute(
            processor,
            ctx,
            type_object="denoiser",
            tiling_strategy=args["tiling_strategy"],
            patch_size=args["patch_size"],
            overlap=args["overlap"],
        )

        x = torch.randn(1, 3, 16, 16, device=ctx.device)
        with torch.no_grad():
            result = distributed_processor(x, sigma=0.1)

        assert result.shape == x.shape
        assert result.device == ctx.device

        return {"result_norm": result.norm().item(), "rank": rank}


@pytest.mark.parametrize("tiling_strategy", ["basic", "overlap_tiling"])
@pytest.mark.parametrize("denoiser_spec", ["simple", "drunet"])
def test_distribute_processor(device_config, tiling_strategy, denoiser_spec):
    """Test processor distribution in multi-process mode."""
    test_args = {
        "device_mode": device_config["device_mode"],
        "tiling_strategy": tiling_strategy,
        "patch_size": 8,
        "overlap": 2,
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
    """
    Worker for data fidelity tests.

    :param int rank: Rank of the current process.
    :param int world_size: Total number of processes.
    :param dict args: Test arguments dictionary.
    :return: Test result tensor.
    """
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        num_operators = 3
        physics_list = create_test_physics_list(ctx.device, num_operators)
        distributed_physics = distribute(physics_list, ctx=ctx)

        # 1. Test factory method
        def fidelity_factory(idx, device, factory_kwargs):
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
    """
    Test data fidelity distribution.

    :param dict device_config: Device configuration from fixture.
    """
    test_args = {
        "device_mode": device_config["device_mode"],
    }
    results = run_distributed_test(_test_data_fidelity_worker, device_config, test_args)
    assert all(r == "success" for r in results)


# =============================================================================
# Advanced Operations Tests (A_dagger, compute_norm, etc.)
# =============================================================================


def _test_compute_norm_worker(rank, world_size, args):
    """
    Worker for compute_norm tests.

    :param int rank: Rank of the current process.
    :param int world_size: Total number of processes.
    :param dict args: Test arguments dictionary.
    :return: Success status string.
    """
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
    """
    Test compute_norm operation in distributed setting.

    :param dict device_config: Device configuration from fixture.
    :param str gather_strategy: Gather strategy to use.
    """
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
    """
    Worker for A_dagger tests.

    :param int rank: Rank of the current process.
    :param int world_size: Total number of processes.
    :param dict args: Test arguments dictionary.
    :return: Success status string.
    """
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        physics_list = create_test_physics_list(ctx.device, 3)
        distributed_physics = distribute(physics_list, ctx=ctx)

        x = torch.randn(1, 1, 16, 16, device=ctx.device)
        y = distributed_physics.A(x)

        x_dagger = distributed_physics.A_dagger(y)

        assert x_dagger.shape == x.shape
        return "success"


def test_a_dagger(device_config):
    """
    Test A_dagger (pseudo-inverse) operation in distributed setting.

    :param dict device_config: Device configuration from fixture.
    """
    test_args = {
        "device_mode": device_config["device_mode"],
    }
    results = run_distributed_test(_test_a_dagger_worker, device_config, test_args)
    assert all(r == "success" for r in results)


# =============================================================================
# Reduce Parameter Tests
# =============================================================================


def _test_gather_false_worker(rank, world_size, args):
    """
    Worker for testing operations with gather=False parameter.

    :param int rank: Rank of the current process.
    :param int world_size: Total number of processes.
    :param dict args: Test arguments dictionary.
    :return: Success status string.
    """
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        physics_list = create_test_physics_list(ctx.device, 3)
        distributed_physics = distribute(physics_list, ctx=ctx)

        x = torch.randn(1, 1, 16, 16, device=ctx.device)

        # Test A with gather=False
        y_local = distributed_physics.A(x, gather=False)
        assert isinstance(y_local, list)

        # Test A_adjoint with gather=False
        y_full = distributed_physics.A(x, gather=True)
        x_adj_local = distributed_physics.A_adjoint(y_full, gather=False)
        assert torch.is_tensor(x_adj_local)
        return "success"


def test_gather_false_operations(device_config):
    """
    Test operations with gather=False parameter.

    Verifies that operations return local results without global reduction.

    :param dict device_config: Device configuration from fixture.
    """
    test_args = {"device_mode": device_config["device_mode"]}
    results = run_distributed_test(_test_gather_false_worker, device_config, test_args)
    assert all(r == "success" for r in results)


# =============================================================================
# Integration and Consistency Tests
# =============================================================================


def _test_consistency_worker(rank, world_size, args):
    """
    Worker for consistency test between single and multiprocess.

    :param int rank: Rank of the current process.
    :param int world_size: Total number of processes.
    :param dict args: Test arguments dictionary.
    :return: Test result tensor.
    """
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
    """
    Verify that single-process and multi-process give same results.

    :param dict device_config: Device configuration from fixture.
    """
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
    """
    Worker for testing empty local set handling.

    Tests the edge case where some ranks have no operators to process.

    :param int rank: Rank of the current process.
    :param int world_size: Total number of processes.
    :param dict args: Test arguments dictionary.
    :return: Success status string.
    """
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
    """
    Test handling when some ranks have no operators to process.

    :param dict device_config: Device configuration from fixture.
    """
    test_args = {"device_mode": device_config["device_mode"]}
    results = run_distributed_test(
        _test_empty_local_set_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


def test_gather_strategy_validation():
    """
    Test that invalid gather strategies are rejected.
    """
    with DistributedContext(device_mode="cpu") as ctx:
        physics_list = create_test_physics_list(ctx.device, 3)

        with pytest.raises(ValueError, match="gather_strategy"):
            distribute(physics_list, ctx=ctx, gather_strategy="invalid_strategy")


def test_distribute_auto_type_detection():
    """
    Test automatic type detection in distribute().
    """
    with DistributedContext(device_mode="cpu") as ctx:
        # Physics list
        physics_list = create_test_physics_list(ctx.device, 3)
        dist_phys = distribute(physics_list, ctx=ctx, type_object="auto")
        assert isinstance(dist_phys, DistributedStackedLinearPhysics)

        # Denoiser
        denoiser = SimpleDenoiser()
        dist_den = distribute(denoiser, ctx=ctx, type_object="auto")
        assert isinstance(dist_den, DistributedProcessing)


# =============================================================================
# Adjoint Operations Tests
# =============================================================================


def _test_adjoint_operations_worker(rank, world_size, args):
    """
    Worker for adjoint operations tests.

    Tests A_adjoint, A_vjp, A_adjoint_A, and A_A_adjoint operations.

    :param int rank: Rank of the current process.
    :param int world_size: Total number of processes.
    :param dict args: Test arguments dictionary.
    :return: Success status string.
    """
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        physics_list = create_test_physics_list(ctx.device, 3)
        distributed_physics = distribute(
            physics_list,
            ctx=ctx,
            gather_strategy=args["gather_strategy"],
            reduction=args["reduction"],
        )
        stacked_physics = StackedLinearPhysics(
            physics_list, reduction=args["reduction"]
        )

        x = args["x"].to(ctx.device)

        # Test A and A_adjoint
        y = distributed_physics.A(x)
        x_adj = distributed_physics.A_adjoint(y)
        x_adj_ref = stacked_physics.A_adjoint(y)
        assert x_adj.shape == x.shape
        assert torch.allclose(x_adj, x_adj_ref, atol=1e-5)

        # Test A_vjp
        # Use y as cotangent vector
        x_vjp = distributed_physics.A_vjp(x, y)
        x_vjp_ref = stacked_physics.A_vjp(x, y)
        assert x_vjp.shape == x.shape
        # A_vjp should equal A_adjoint for LinearPhysics
        assert torch.allclose(x_vjp, x_vjp_ref, atol=1e-5)

        # Test A_adjoint_A
        x_ata = distributed_physics.A_adjoint_A(x)
        x_ata_ref = stacked_physics.A_adjoint_A(x)
        assert x_ata.shape == x.shape
        assert torch.allclose(x_ata, x_ata_ref, atol=1e-5)

        # Test A_A_adjoint - should return TensorList like StackedLinearPhysics
        y_aat = distributed_physics.A_A_adjoint(y)
        y_aat_stacked = stacked_physics.A_A_adjoint(y)
        assert isinstance(y_aat, TensorList), "A_A_adjoint should return TensorList"
        assert len(y_aat) == len(
            physics_list
        ), "TensorList should have one entry per operator"
        for i in range(len(physics_list)):
            assert torch.allclose(
                y_aat[i], y_aat_stacked[i], atol=1e-5
            ), f"Mismatch at operator {i}"
        return "success"


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_adjoint_operations(device_config, gather_strategy, reduction):
    """
    Test A_adjoint, A_vjp, A_adjoint_A, A_A_adjoint operations.

    :param dict device_config: Device configuration from fixture.
    :param str gather_strategy: Gather strategy to use.
    :param str reduction: Reduction mode ('sum' or 'mean').
    """
    # Skip naive strategy with multi-GPU (NCCL doesn't support all_gather_object)
    if gather_strategy == "naive" and device_config["device_mode"] == "gpu":
        pytest.skip("Naive gather strategy not supported with NCCL backend (gpu_multi)")

    torch.manual_seed(42)
    x = torch.randn(1, 1, 16, 16)
    test_args = {
        "device_mode": device_config["device_mode"],
        "gather_strategy": gather_strategy,
        "reduction": reduction,
        "x": x,
    }
    results = run_distributed_test(
        _test_adjoint_operations_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


# =============================================================================
# Advanced Solver Tests (local_only parameter)
# =============================================================================


def _test_compute_norm_local_vs_global_worker(rank, world_size, args):
    """
    Worker for compute_norm tests with local_only parameter.

    :param int rank: Rank of the current process.
    :param int world_size: Total number of processes.
    :param dict args: Test arguments dictionary.
    :return: Success status string.
    """
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
    """
    Test compute_norm with local_only parameter.

    :param dict device_config: Device configuration from fixture.
    :param bool local_only: Whether to compute norm locally only.
    """

    test_args = {
        "device_mode": device_config["device_mode"],
        "local_only": local_only,
    }
    results = run_distributed_test(
        _test_compute_norm_local_vs_global_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


def _test_a_dagger_local_vs_global_worker(rank, world_size, args):
    """
    Worker for A_dagger tests with local_only parameter.

    :param int rank: Rank of the current process.
    :param int world_size: Total number of processes.
    :param dict args: Test arguments dictionary.
    :return: Success status string.
    """
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
    """
    Test A_dagger with local_only parameter.

    :param dict device_config: Device configuration from fixture.
    :param bool local_only: Whether to compute A_dagger locally only.
    """

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
    """
    Worker for testing DistributedContext with different device modes.

    :param int rank: Rank of the current process.
    :param int world_size: Total number of processes.
    :param dict args: Test arguments dictionary.
    :return: Success status string.
    """
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
    """
    Test DistributedContext with different device modes.

    :param dict device_config: Device configuration from fixture.
    """
    test_args = {"device_mode": device_config["device_mode"]}
    results = run_distributed_test(
        _test_distributed_context_device_modes_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


def _test_distributed_context_local_indices_worker(rank, world_size, args):
    """
    Worker for testing local_indices sharding.

    :param int rank: Rank of the current process.
    :param int world_size: Total number of processes.
    :param dict args: Test arguments dictionary.
    :return: Dictionary containing rank, indices, and number of indices.
    """
    with DistributedContext(device_mode=args["device_mode"]) as ctx:
        total_items = 10
        indices = ctx.local_indices(total_items)
        # Each rank gets a subset of indices
        assert len(indices) <= total_items
        # Indices should be in range
        assert all(0 <= i < total_items for i in indices)
    return {"rank": rank, "indices": indices, "num_indices": len(indices)}


def test_distributed_context_local_indices(device_config):
    """
    Test local_indices sharding across ranks.

    :param dict device_config: Device configuration from fixture.
    """
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
    """
    Worker for testing collective operations.

    :param int rank: Rank of the current process.
    :param int world_size: Total number of processes.
    :param dict args: Test arguments dictionary.
    :return: Success status string.
    """
    with DistributedContext(device_mode=args["device_mode"]) as ctx:
        # Test all_reduce with sum
        t = torch.tensor([float(rank + 1), 2.0, 3.0], device=ctx.device)
        result_sum = t.clone()
        ctx.all_reduce(result_sum, op=dist.ReduceOp.SUM)
        # Sum across all ranks: rank 0 contributes 1, rank 1 contributes 2, etc.
        expected_sum = torch.tensor(
            [sum(range(1, world_size + 1)), 2.0 * world_size, 3.0 * world_size],
            device=ctx.device,
        )
        assert torch.allclose(
            result_sum, expected_sum
        ), f"Rank {rank}: {result_sum} vs {expected_sum}"

        # Test all_reduce with mean
        result_mean = t.clone()
        ctx.all_reduce(result_mean, op=dist.ReduceOp.AVG)
        expected_mean = expected_sum / world_size
        assert torch.allclose(
            result_mean, expected_mean
        ), f"Rank {rank}: {result_mean} vs {expected_mean}"

        # Test broadcast
        if rank == 0:
            t_bcast = torch.tensor([10.0, 20.0, 30.0], device=ctx.device)
        else:
            t_bcast = torch.tensor([0.0, 0.0, 0.0], device=ctx.device)

        result_bcast = t_bcast.clone()
        ctx.broadcast(result_bcast, src=0)
        expected_bcast = torch.tensor([10.0, 20.0, 30.0], device=ctx.device)
        assert torch.allclose(result_bcast, expected_bcast)

    return "success"


def test_distributed_context_collectives(device_config):
    """
    Test collective operations (all_reduce, broadcast, barrier) across ranks.

    :param dict device_config: Device configuration from fixture.
    """
    test_args = {"device_mode": device_config["device_mode"]}
    results = run_distributed_test(
        _test_distributed_context_collectives_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


# =============================================================================
# Processor Batch Size and Tiling Tests
# =============================================================================


def _test_processor_different_patch_sizes_worker(rank, world_size, args):
    """
    Worker for testing processor with different patch sizes.

    :param int rank: Rank of the current process.
    :param int world_size: Total number of processes.
    :param dict args: Test arguments dictionary.
    :return: Success status string.
    """
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        processor = create_denoiser(args["denoiser_spec"], ctx.device, num_channels=3)

        for patch_size in [4, 8, 16]:
            distributed_processor = distribute(
                processor,
                ctx,
                type_object="denoiser",
                tiling_strategy="overlap_tiling",
                patch_size=patch_size,
                overlap=2,
            )

            x = torch.randn(1, 3, 16, 16, device=ctx.device)

            if patch_size == 16:
                with pytest.raises(ValueError):
                    result = distributed_processor(x, sigma=0.1)
            else:
                result = distributed_processor(x, sigma=0.1)
                assert result.shape == x.shape
        return "success"


@pytest.mark.parametrize("denoiser_spec", ["simple"])
def test_processor_different_patch_sizes(device_config, denoiser_spec):
    """
    Test processor distribution with varying patch sizes.

    :param dict device_config: Device configuration from fixture.
    :param str denoiser_spec: Denoiser specification type.
    """
    test_args = {
        "device_mode": device_config["device_mode"],
        "denoiser_spec": denoiser_spec,
    }
    results = run_distributed_test(
        _test_processor_different_patch_sizes_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


def _test_processor_max_batch_size_worker(rank, world_size, args):
    """
    Worker for testing processor with different max batch sizes.

    :param int rank: Rank of the current process.
    :param int world_size: Total number of processes.
    :param dict args: Test arguments dictionary.
    :return: Success status string.
    """
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
                tiling_strategy="overlap_tiling",
                patch_size=8,
                overlap=2,
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
    """
    Test processor with different max_batch_size settings.

    :param dict device_config: Device configuration from fixture.
    :param str denoiser_spec: Denoiser specification type.
    """
    test_args = {
        "device_mode": device_config["device_mode"],
        "denoiser_spec": denoiser_spec,
    }
    results = run_distributed_test(
        _test_processor_max_batch_size_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


def _test_processor_3d_worker(rank, world_size, args):
    """
    Worker for testing processor with 3D volumes.

    :param int rank: Rank of the current process.
    :param int world_size: Total number of processes.
    :param dict args: Test arguments dictionary.
    :return: Success status string.
    """
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        processor = create_denoiser(args["denoiser_spec"], ctx.device, num_channels=1)

        distributed_processor = distribute(
            processor,
            ctx,
            type_object="denoiser",
            tiling_strategy="overlap_tiling",
            patch_size=8,
            overlap=2,
        )

        # Test with 3D input
        x_3d = torch.randn(1, 1, 16, 16, 16, device=ctx.device)
        result_3d = distributed_processor(x_3d, sigma=0.1)

        assert result_3d.shape == x_3d.shape
        return "success"


@pytest.mark.parametrize("denoiser_spec", ["simple"])
def test_processor_3d(device_config, denoiser_spec):
    """
    Test processor with 3D volumes.

    :param dict device_config: Device configuration from fixture.
    :param str denoiser_spec: Denoiser specification type.
    """
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
    """
    Worker for comparing distributed data fidelity with stacked version.

    :param int rank: Rank of the current process.
    :param int world_size: Total number of processes.
    :param dict args: Test arguments dictionary.
    :return: Success status string.
    """
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
    """
    Compare DistributedDataFidelity with StackedPhysicsDataFidelity.

    :param dict device_config: Device configuration from fixture.
    """
    test_args = {"device_mode": device_config["device_mode"]}
    results = run_distributed_test(
        _test_data_fidelity_vs_stacked_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


def _test_data_fidelity_different_fidelities_worker(rank, world_size, args):
    """
    Worker for testing different data fidelity types.

    :param int rank: Rank of the current process.
    :param int world_size: Total number of processes.
    :param dict args: Test arguments dictionary.
    :return: Success status string.
    """
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
    """
    Test with different data fidelity types (L1, L2).

    :param dict device_config: Device configuration from fixture.
    """
    test_args = {"device_mode": device_config["device_mode"]}
    results = run_distributed_test(
        _test_data_fidelity_different_fidelities_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


# =============================================================================
# Gather=False Comprehensive Tests
# =============================================================================


def _test_gather_false_physics_operations_worker(rank, world_size, args):
    operation = args["operation"]
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        physics_list = create_test_physics_list(ctx.device, 3)
        distributed_physics = distribute(physics_list, ctx=ctx)

        x = torch.randn(1, 1, 16, 16, device=ctx.device)

        if operation == "A":
            result_local = distributed_physics.A(x, gather=False)
            result_global = distributed_physics.A(x, gather=True)
            assert isinstance(result_local, list)
            assert isinstance(result_global, TensorList)

        elif operation == "forward":
            result_local = distributed_physics.forward(x, gather=False)
            result_global = distributed_physics.forward(x, gather=True)
            assert isinstance(result_local, list)
            assert isinstance(result_global, TensorList)

        elif operation == "A_adjoint":
            y = distributed_physics.A(x)
            # A_adjoint reduces by default, so we test both modes
            result_global = distributed_physics.A_adjoint(y)
            assert torch.is_tensor(result_global)
            # Local version returns unreduced result (one per local operator)
            result_local = distributed_physics.A_adjoint(y, gather=False)
            assert isinstance(result_local, list) or torch.is_tensor(result_local)

        elif operation == "A_vjp":
            y = distributed_physics.A(x)
            # A_vjp reduces by default
            result_global = distributed_physics.A_vjp(x, y)
            assert torch.is_tensor(result_global)
            # Local version
            result_local = distributed_physics.A_vjp(x, y, gather=False)
            assert isinstance(result_local, list) or torch.is_tensor(result_local)

        elif operation == "A_adjoint_A":
            # A_adjoint_A reduces by default
            result_global = distributed_physics.A_adjoint_A(x)
            assert torch.is_tensor(result_global)
            # Local version
            result_local = distributed_physics.A_adjoint_A(x, gather=False)
            assert isinstance(result_local, list) or torch.is_tensor(result_local)

        elif operation == "A_A_adjoint":
            y = distributed_physics.A(x)
            # A_A_adjoint should return TensorList when gather=True
            result_global = distributed_physics.A_A_adjoint(y)
            assert isinstance(result_global, TensorList)
            # Local version returns list of tensors (one per local operator)
            result_local = distributed_physics.A_A_adjoint(y, gather=False)
            assert isinstance(result_local, list)
    return "success"


@pytest.mark.parametrize(
    "operation", ["A", "forward", "A_adjoint", "A_vjp", "A_adjoint_A", "A_A_adjoint"]
)
def test_gather_false_physics_operations(device_config, operation):
    """Test all physics operations with gather=False."""
    test_args = {"device_mode": device_config["device_mode"], "operation": operation}
    results = run_distributed_test(
        _test_gather_false_physics_operations_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


def _test_gather_false_processor_worker(rank, world_size, args):
    denoiser_spec = args["denoiser_spec"]
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        processor = create_denoiser(denoiser_spec, ctx.device, num_channels=3)
        distributed_processor = distribute(
            processor,
            ctx,
            type_object="denoiser",
            tiling_strategy="overlap_tiling",
            patch_size=8,
            overlap=2,
        )

        x = torch.randn(1, 3, 16, 16, device=ctx.device)

        # Test with gather=True (default)
        result_global = distributed_processor(x, sigma=0.1)
        assert torch.is_tensor(result_global)

        # Test with gather=False if supported
        result_local = distributed_processor(x, gather=False, sigma=0.1)
        assert torch.is_tensor(
            result_local
        )  # Same tensor but a lot of zeros (patches not processed locally)

    return "success"


@pytest.mark.parametrize("denoiser_spec", ["simple", "drunet"])
def test_gather_false_processor(device_config, denoiser_spec):
    """Test DistributedProcessing with gather=False."""
    test_args = {
        "device_mode": device_config["device_mode"],
        "denoiser_spec": denoiser_spec,
    }
    results = run_distributed_test(
        _test_gather_false_processor_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


def _test_gather_false_data_fidelity_worker(rank, world_size, args):
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        physics_list = create_test_physics_list(ctx.device, 3)
        distributed_physics = distribute(physics_list, ctx=ctx)

        distributed_fidelity = distribute(L2(), ctx=ctx, num_operators=3)

        x = torch.randn(1, 1, 16, 16, device=ctx.device)
        y = distributed_physics.A(x)

        # Test fn with gather=False
        fid_local = distributed_fidelity.fn(x, y, distributed_physics, gather=False)
        fid_global = distributed_fidelity.fn(x, y, distributed_physics, gather=True)
        assert torch.is_tensor(fid_local)
        assert torch.is_tensor(fid_global)

        # Test grad with gather=False
        grad_local = distributed_fidelity.grad(x, y, distributed_physics, gather=False)
        grad_global = distributed_fidelity.grad(x, y, distributed_physics, gather=True)
        assert torch.is_tensor(grad_local)
        assert torch.is_tensor(grad_global)
    return "success"


def test_gather_false_data_fidelity(device_config):
    """Test DistributedDataFidelity with gather=False."""
    test_args = {"device_mode": device_config["device_mode"]}
    results = run_distributed_test(
        _test_gather_false_data_fidelity_worker, device_config, test_args
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


# =============================================================================
# Backward / Gradient Tests
# =============================================================================


class TrainableDenoiser(Denoiser):
    """
    Simple trainable denoiser for testing gradients.
    """

    def __init__(self, channels=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # Initialize with known weights for reproducibility
        with torch.no_grad():
            self.conv.weight.fill_(0.1)
            self.conv.bias.fill_(0.01)

    def forward(self, x, sigma=None, **kwargs):
        return self.conv(x)


def _test_physics_backward_worker(rank, world_size, args):
    """Worker for testing physics backward pass."""
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        num_operators = args["num_operators"]
        # Create physics
        # Note: Physics usually don't have learnable params, so we check grad w.r.t input x
        physics_list = create_test_physics_list(ctx.device, num_operators)

        # We need a physics that supports backward on its operations.
        # Blur/GaussianNoise are standard diffable ops.

        # Distributed
        if args["gather_strategy"] == "naive" and ctx.get_backend() == "nccl":
            # Naive not supported on NCCL
            return "skipped"

        distributed_physics = distribute(
            physics_list, ctx, gather_strategy=args["gather_strategy"]
        )

        # Reference
        stacked_physics = StackedLinearPhysics(physics_list)

        # Input
        # Ensure identical initialization across ranks for inputs
        rng_state = torch.get_rng_state()
        torch.manual_seed(1234)  # Shared seed (overrides DistributedContext diversity)
        x = torch.randn(1, 1, 16, 16, device=ctx.device, requires_grad=True)
        torch.set_rng_state(rng_state)

        # We need a clone for reference to ensure clean gradients
        x_ref = x.clone().detach().requires_grad_(True)

        # 1. Distributed Forward
        y_dist = distributed_physics.A(x)

        # Loss
        # We compute the global loss on EACH rank.
        # Since dist_nn functions backpropagate gradients from all ranks to the source,
        # computing the loss on all ranks effectively multiplies the gradient by world_size.
        # We must normalize to match the single-process reference.
        loss_dist = sum([yi.sum() for yi in y_dist])

        # Backward
        # For broadcast strategy, we have multiple independent broadcast operations in the graph.
        # Autograd backward order is not guaranteed for independent nodes, which can cause deadlocks
        # with distributed collectives (Rank 0 does Bcast2 then Bcast1, Rank 1 does Bcast1 then Bcast2).
        # We enforce deterministic backward order by backpropagating sequentially.
        if args.get("gather_strategy") == "broadcast":
            grad_scaler = 1.0
            # Backward in reverse order (standard convention, though any agreed order works)
            # We use retain_graph=True for all but the last one (or just all and let iter finish)
            # But x.grad accumulates, so it's fine.
            # actually we don't need retain_graph for x, but we need it for intermediate graph if shared?
            # independent branches only join at x.
            for i in range(len(y_dist) - 1, -1, -1):
                (y_dist[i].sum() * grad_scaler).backward()
        else:
            loss_dist = sum([yi.sum() for yi in y_dist])
            loss_dist.backward()

        # Aggregate gradients from all ranks IS NOW AUTOMATIC via DistributedGradientSync!
        # if ctx.use_dist:
        #      dist.all_reduce(x.grad, op=dist.ReduceOp.SUM)

        grad_dist = x.grad.clone()

        # 2. Reference Forward
        y_ref = stacked_physics.A(x_ref)
        loss_ref = sum([yi.sum() for yi in y_ref])
        loss_ref.backward()
        grad_ref = x_ref.grad.clone()

        # Compare Gradients
        assert torch.allclose(grad_dist, grad_ref, atol=1e-5)

        return "success"


@pytest.mark.parametrize("gather_strategy", ["concatenated", "broadcast"])
def test_distributed_physics_backward(device_config, gather_strategy, num_operators):
    """
    Test that gradients flow correctly through DistributedStackedPhysics.
    """
    if gather_strategy == "broadcast" and device_config["world_size"] > 1:
        pytest.skip(
            "Skipping broadcast backward due to known deadlock issue with dist_nn.broadcast backward on some backends"
        )

    test_args = {
        "num_operators": num_operators,
        "device_mode": device_config["device_mode"],
        "gather_strategy": gather_strategy,
    }
    results = run_distributed_test(
        _test_physics_backward_worker, device_config, test_args
    )

    # filter skipped
    results = [r for r in results if r != "skipped"]
    if not results:
        pytest.skip("All ranks skipped (likely due to naive+NCCL)")

    if test_args["gather_strategy"] == "broadcast" and results == []:
        # If results empty/timeout for broadcast, it might be the deadlock issue
        pass

    assert all(r == "success" for r in results)


def _test_data_fidelity_backward_worker(rank, world_size, args):
    """Worker for testing backward-through-grad with DistributedDataFidelity."""
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        num_operators = args["num_operators"]
        physics_list = create_test_physics_list(ctx.device, num_operators)

        distributed_physics = distribute(physics_list, ctx)
        distributed_fidelity = distribute(L2(), ctx=ctx, num_operators=num_operators)

        stacked_physics = StackedLinearPhysics(physics_list)
        stacked_fidelity = StackedPhysicsDataFidelity(
            [L2() for _ in range(num_operators)]
        )

        # Fixed measurement from a detached clean input (measurement should be treated as constant).
        x_true = torch.randn(1, 1, 16, 16, device=ctx.device)
        y_dist = distributed_physics.A(x_true)

        # Reuse the exact same measurements for the non-distributed reference.
        y_ref = [yi.detach().clone() for yi in y_dist]

        # Dist path: first-order differentiation through data_fidelity.grad wrt x.
        x = torch.randn(1, 1, 16, 16, device=ctx.device, requires_grad=True)
        g_dist = distributed_fidelity.grad(x, y_dist, distributed_physics)
        loss_dist = g_dist.sum()
        loss_dist.backward()
        grad_dist = x.grad.clone()

        # Reference path.
        x_ref = x.detach().clone().requires_grad_(True)
        g_ref = stacked_fidelity.grad(x_ref, y_ref, stacked_physics)
        loss_ref = g_ref.sum()
        loss_ref.backward()
        grad_ref = x_ref.grad.clone()

        assert torch.allclose(grad_dist, grad_ref, atol=1e-5)
        return "success"

def test_distributed_data_fidelity_backward(device_config, num_operators):
    """
    Test gradients through DistributedDataFidelity.grad wrt x.
    """
    test_args = {"device_mode": device_config["device_mode"], "num_operators": num_operators}
    results = run_distributed_test(
        _test_data_fidelity_backward_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


def _test_distributed_gradient_sync_higher_order_worker(rank, world_size, args):
    """Test higher-order gradients through DistributedGradientSync."""
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        x = torch.randn(1, 1, 8, 8, device=ctx.device, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)

        physics_list = create_test_physics_list(ctx.device, 3)
        for p in physics_list:
            p.noise_model = GaussianNoise(sigma=0.0)
        distributed_physics = distribute(physics_list, ctx)
        stacked_physics = StackedLinearPhysics(physics_list)

        y_dist = distributed_physics.A(x)
        loss_dist = sum((yi.square().sum() for yi in y_dist)) / ctx.world_size
        grad1_dist = torch.autograd.grad(loss_dist, x, create_graph=True)[0]
        loss2_dist = grad1_dist.square().sum() / ctx.world_size
        grad2_dist = torch.autograd.grad(loss2_dist, x)[0]

        y_ref = stacked_physics.A(x_ref)
        loss_ref = sum((yi.square().sum() for yi in y_ref))
        grad1_ref = torch.autograd.grad(loss_ref, x_ref, create_graph=True)[0]
        loss2_ref = grad1_ref.square().sum()
        grad2_ref = torch.autograd.grad(loss2_ref, x_ref)[0]

        assert torch.allclose(grad2_dist, grad2_ref, atol=1e-5)
        return "success"


def test_distributed_gradient_sync_higher_order(device_config):
    """Test second-order gradient consistency for distributed physics input sync."""
    if device_config["world_size"] > 1:
        pytest.skip(
            "Higher-order exact equivalence is only enforced in single-process mode."
        )
    test_args = {"device_mode": device_config["device_mode"]}
    results = run_distributed_test(
        _test_distributed_gradient_sync_higher_order_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


def _test_distributed_parameter_sync_higher_order_worker(rank, world_size, args):
    """Test higher-order gradients through DistributedParameterSync."""
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        denoiser = TrainableDenoiser(channels=3).to(ctx.device)
        distributed_processor = distribute(
            denoiser,
            ctx,
            type_object="denoiser",
            tiling_strategy="overlap_tiling",
            patch_size=8,
            overlap=2,
        )

        denoiser_ref = TrainableDenoiser(channels=3).to(ctx.device)
        denoiser_ref.load_state_dict(denoiser.state_dict())

        x = torch.randn(1, 3, 16, 16, device=ctx.device, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)

        # Distributed path
        out_dist = distributed_processor(x)
        loss_dist = out_dist.sum() / ctx.world_size
        loss_dist.backward(create_graph=True)
        meta_dist = 0.0
        for p in denoiser.parameters():
            assert p.grad is not None
            meta_dist = meta_dist + p.grad.square().sum()
        grad2_dist = torch.autograd.grad(meta_dist, x)[0]

        # Reference path (single-process equivalent on same rank)
        from deepinv.distributed.strategies import create_strategy

        strategy = create_strategy(
            "overlap_tiling", x_ref.shape, patch_size=8, overlap=2
        )
        all_indices = list(range(strategy.get_num_patches()))
        patch_pairs = strategy.get_local_patches(x_ref, all_indices)
        all_patches = [p for _, p in patch_pairs]
        processed = [denoiser_ref(p) for p in all_patches]
        out_ref = torch.zeros_like(x_ref)
        strategy.reduce_patches(out_ref, list(zip(all_indices, processed)))

        loss_ref = out_ref.sum()
        loss_ref.backward(create_graph=True)
        meta_ref = 0.0
        for p in denoiser_ref.parameters():
            assert p.grad is not None
            meta_ref = meta_ref + p.grad.square().sum()
        grad2_ref = torch.autograd.grad(meta_ref, x_ref)[0]

        assert torch.allclose(grad2_dist, grad2_ref, atol=1e-5)
        return "success"

def test_distributed_parameter_sync_higher_order(device_config, num_operators):
    """Test second-order gradient consistency for distributed parameter sync."""
    if device_config["world_size"] > 1:
        pytest.skip(
            "Higher-order exact equivalence is only enforced in single-process mode."
        )
    test_args = {"device_mode": device_config["device_mode"], "num_operators": num_operators}
    results = run_distributed_test(
        _test_distributed_parameter_sync_higher_order_worker,
        device_config,
        test_args,
    )
    assert all(r == "success" for r in results)


def _test_unrolled_backward_worker(rank, world_size, args):
    """
    Worker for unrolled backward consistency:
    compares distributed vs reference forward and gradients.
    """
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        num_operators = args["num_operators"]
        n_unroll = 3
        sigma_denoiser = 0.02
        patch_size = 8
        overlap = 2
        max_batch_size = args.get("max_batch_size", None)
        checkpoint_batches = args.get("checkpoint_batches", "auto")

        physics_list = create_test_physics_list(ctx.device, num_operators)
        for p in physics_list:
            p.noise_model = GaussianNoise(sigma=0.0)

        distributed_physics = distribute(physics_list, ctx)
        distributed_fidelity = distribute(L2(), ctx=ctx, num_operators=num_operators)

        stacked_physics = StackedLinearPhysics(physics_list)
        stacked_fidelity = StackedPhysicsDataFidelity(
            [L2() for _ in range(num_operators)]
        )

        # Build same DRUNet on all ranks and sync weights from rank 0.
        denoiser = create_drunet_denoiser(num_channels=1, device=ctx.device)
        if ctx.use_dist:
            with torch.no_grad():
                for p in denoiser.parameters():
                    dist.broadcast(p.data, src=0)
                for b in denoiser.buffers():
                    dist.broadcast(b.data, src=0)

        distributed_denoiser = distribute(
            denoiser,
            ctx,
            type_object="denoiser",
            tiling_strategy="overlap_tiling",
            patch_size=patch_size,
            overlap=overlap,
            max_batch_size=max_batch_size,
            checkpoint_batches=checkpoint_batches,
        )

        denoiser_ref = create_drunet_denoiser(num_channels=1, device=ctx.device)
        denoiser_ref.load_state_dict(denoiser.state_dict())
        from deepinv.distributed.strategies import create_strategy

        strategy = create_strategy(
            "overlap_tiling",
            (1, 1, 16, 16),
            patch_size=patch_size,
            overlap=overlap,
        )
        num_patches = strategy.get_num_patches()
        all_indices = list(range(num_patches))
        if max_batch_size is not None and max_batch_size > 0:
            assert (
                max_batch_size < num_patches
            ), f"Expected max_batch_size < num_patches, got {max_batch_size} >= {num_patches}"

        def _reference_tiled_denoise(x_in):
            patch_pairs = strategy.get_local_patches(x_in, all_indices)
            all_patches = [p for _, p in patch_pairs]
            batched_patches = strategy.apply_batching(
                all_patches, max_batch_size=max_batch_size
            )
            processed_batches = [
                denoiser_ref(batch, sigma=sigma_denoiser) for batch in batched_patches
            ]
            processed = strategy.unpack_batched_results(
                processed_batches, len(all_patches)
            )
            x_out = torch.zeros_like(x_in)
            strategy.reduce_patches(x_out, list(zip(all_indices, processed)))
            return x_out

        # Fixed measurement (constant in the graph).
        rng_state = torch.get_rng_state()
        torch.manual_seed(1234)
        x_true = torch.randn(1, 1, 16, 16, device=ctx.device)
        x0 = torch.randn(1, 1, 16, 16, device=ctx.device)
        torch.set_rng_state(rng_state)

        y_dist = distributed_physics.A(x_true)
        y_ref = [yi.detach().clone() for yi in y_dist]

        steps = distribute(
            torch.nn.Parameter(torch.tensor([0.4, 0.3, 0.2], device=ctx.device)),
            ctx,
        )
        x = x0.detach().clone().requires_grad_(True)

        # Distributed unrolled forward
        x_dist = x
        for k in range(n_unroll):
            grad_k = distributed_fidelity.grad(x_dist, y_dist, distributed_physics)
            x_dist = x_dist - steps[k] * grad_k
            x_dist = distributed_denoiser(x_dist, sigma=sigma_denoiser)

        # Reference unrolled forward
        steps_ref = steps.detach().clone().requires_grad_(True)
        x_ref = x0.detach().clone().requires_grad_(True)
        x_ref_out = x_ref
        for k in range(n_unroll):
            grad_k_ref = stacked_fidelity.grad(x_ref_out, y_ref, stacked_physics)
            x_ref_out = x_ref_out - steps_ref[k] * grad_k_ref
            x_ref_out = _reference_tiled_denoise(x_ref_out)

        # 1) Forward closeness check
        diff_out = (x_dist - x_ref_out).abs()
        assert torch.allclose(x_dist, x_ref_out, atol=1e-4), (
            "Unrolled forward mismatch: "
            f"mean={diff_out.mean().item():.3e}, max={diff_out.max().item():.3e}"
        )

        # 2) Backward checks (x, steps, denoiser params)
        loss_dist = x_dist.square().sum()
        loss_ref = x_ref_out.square().sum()
        loss_dist.backward()
        loss_ref.backward()

        if ctx.use_dist and x.grad is not None:
            dist.all_reduce(x.grad, op=dist.ReduceOp.SUM)
            x.grad = x.grad / float(ctx.world_size)

        issues = []

        assert x.grad is not None and x_ref.grad is not None
        diff_x = (x.grad - x_ref.grad).abs()
        x_norm = x.grad.norm().item()
        x_ref_norm = x_ref.grad.norm().item()
        norm_ratio = x_norm / (x_ref_norm + 1e-12)
        cos_sim = torch.nn.functional.cosine_similarity(
            x.grad.reshape(1, -1), x_ref.grad.reshape(1, -1), dim=1
        ).item()
        x_similar = torch.allclose(x.grad, x_ref.grad, atol=5e-1, rtol=5e-2) or (
            0.9 <= norm_ratio <= 1.1 and cos_sim > 0.99
        )
        if not x_similar:
            issues.append(
                "Input gradient mismatch: "
                f"mean={diff_x.mean().item():.3e}, max={diff_x.max().item():.3e}, "
                f"norm_dist={x_norm:.3e}, norm_ref={x_ref_norm:.3e}, "
                f"ratio={norm_ratio:.3e}, cosine={cos_sim:.6f}"
            )

        assert steps.grad is not None and steps_ref.grad is not None
        diff_steps = (steps.grad - steps_ref.grad).abs()
        if not torch.allclose(steps.grad, steps_ref.grad, atol=1e-3, rtol=5e-4):
            issues.append(
                "Steps gradient mismatch: "
                f"dist={steps.grad.detach().cpu().tolist()}, ref={steps_ref.grad.detach().cpu().tolist()}, "
                f"max={diff_steps.max().item():.3e}"
            )

        for i, (p_dist, p_ref) in enumerate(
            zip(denoiser.parameters(), denoiser_ref.parameters())
        ):
            assert p_dist.grad is not None and p_ref.grad is not None
            d = (p_dist.grad - p_ref.grad).abs()
            if not torch.allclose(p_dist.grad, p_ref.grad, atol=5e-3, rtol=1e-4):
                issues.append(
                    f"Param grad mismatch at index {i}: "
                    f"mean={d.mean().item():.3e}, max={d.max().item():.3e}, "
                    f"norm_dist={p_dist.grad.norm().item():.3e}, norm_ref={p_ref.grad.norm().item():.3e}"
                )

        assert not issues, "\n".join(issues)

        return "success"


def test_unrolled_backward(device_config, num_operators):
    """
    End-to-end unrolled backward consistency test with DRUNet and trainable step sizes.
    """
    test_args = {"device_mode": device_config["device_mode"], "num_operators": num_operators}
    results = run_distributed_test(
        _test_unrolled_backward_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


def test_unrolled_backward_checkpointed_batches(device_config, num_operators):
    """
    End-to-end unrolled backward consistency with checkpointed patch-batches.

    Uses max_batch_size smaller than the total patch count and checks that
    distributed gradients match the non-distributed reference.
    """
    test_args = {
        "device_mode": device_config["device_mode"],
        "num_operators": num_operators,
        "max_batch_size": 1,
        "checkpoint_batches": "always",
    }
    results = run_distributed_test(
        _test_unrolled_backward_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


def _test_distributed_parameter_autodetect_worker(rank, world_size, args):
    """Auto-detected distribute(Parameter) should sync/average gradients."""
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        step = distribute(torch.nn.Parameter(torch.tensor(0.7, device=ctx.device)), ctx)

        x_local = torch.linspace(-1.0, 1.0, steps=16, device=ctx.device).reshape(4, 4)
        x_local = x_local + 0.2 * ctx.rank
        target = torch.zeros_like(x_local)

        loss_dist = ((step * x_local - target) ** 2).mean()
        loss_dist.backward()
        grad_dist = step.grad.detach().clone()

        step_ref = torch.nn.Parameter(torch.tensor(0.7, device=ctx.device))
        loss_ref = 0.0
        for r in range(ctx.world_size):
            x_r = torch.linspace(-1.0, 1.0, steps=16, device=ctx.device).reshape(4, 4)
            x_r = x_r + 0.2 * r
            loss_ref = loss_ref + ((step_ref * x_r - target) ** 2).mean()
        loss_ref = loss_ref / float(ctx.world_size)
        loss_ref.backward()

        assert torch.allclose(grad_dist, step_ref.grad, atol=1e-6), (
            f"Parameter autodetect grad mismatch: dist={grad_dist.item():.6e}, "
            f"ref={step_ref.grad.item():.6e}"
        )
        return "success"


def test_distributed_parameter_autodetect(device_config):
    test_args = {"device_mode": device_config["device_mode"]}
    results = run_distributed_test(
        _test_distributed_parameter_autodetect_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


def _make_test_unfolded_model(device: torch.device, unfold: bool = True) -> PGD:
    return PGD(
        stepsize=[0.8, 0.7],
        sigma_denoiser=0.05,
        trainable_params=["stepsize"],
        data_fidelity=L2(),
        prior=PnP(TrainableDenoiser(channels=1).to(device)),
        max_iter=2,
        unfold=unfold,
    )


def _test_distribute_base_optim_worker(rank, world_size, args):
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        model = _make_test_unfolded_model(ctx.device, unfold=True)
        model_out = _distribute_base_optim(
            model, ctx, patch_size=8, overlap=2, max_batch_size=1
        )
        assert model_out is model

        assert isinstance(model.data_fidelity, torch.nn.ModuleList)
        assert all(
            isinstance(df, DistributedDataFidelity) for df in model.data_fidelity
        )
        assert isinstance(model.prior[0].denoiser, DistributedProcessing)
        assert hasattr(model, "_deepinv_dist_sync")

        physics_list = create_test_physics_list(ctx.device, args["num_operators"])
        for p in physics_list:
            p.noise_model = GaussianNoise(sigma=0.0)
        distributed_physics = distribute(physics_list, ctx=ctx)

        x_true = torch.randn(1, 1, 16, 16, device=ctx.device)
        y = distributed_physics.A(x_true)
        out = model(y, distributed_physics)
        out.square().mean().backward()

        assert all(p.grad is not None for p in model.params_algo["stepsize"])
        denoiser_params = [
            p for p in model.prior[0].denoiser.processor.parameters() if p.requires_grad
        ]
        assert denoiser_params and all(p.grad is not None for p in denoiser_params)
        return "success"


def test_distribute_base_optim(device_config, num_operators):
    test_args = {"device_mode": device_config["device_mode"], "num_operators": num_operators}
    results = run_distributed_test(
        _test_distribute_base_optim_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


def _test_distribute_module_type_baseoptim_worker(rank, world_size, args):
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        model = _make_test_unfolded_model(ctx.device, unfold=True)
        out = distribute(
            model,
            ctx,
            type_object="module",
            patch_size=8,
            overlap=2,
            max_batch_size=1,
        )
        assert out is model
        assert isinstance(model.prior[0].denoiser, DistributedProcessing)
        assert all(
            isinstance(df, DistributedDataFidelity) for df in model.data_fidelity
        )
        return "success"


def test_distribute_module_type_baseoptim(device_config):
    test_args = {"device_mode": device_config["device_mode"]}
    results = run_distributed_test(
        _test_distribute_module_type_baseoptim_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


def _test_distribute_module_type_rejects_non_unfold_worker(rank, world_size, args):
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        model = _make_test_unfolded_model(ctx.device, unfold=False)
        with pytest.raises(TypeError, match="unfold=False"):
            distribute(model, ctx, type_object="module")
        return "success"


def test_distribute_module_type_rejects_non_unfold(device_config):
    test_args = {"device_mode": device_config["device_mode"]}
    results = run_distributed_test(
        _test_distribute_module_type_rejects_non_unfold_worker,
        device_config,
        test_args,
    )
    assert all(r == "success" for r in results)


def _test_distribute_module_autodetect_rejects_generic_worker(rank, world_size, args):
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        module = torch.nn.Linear(4, 1, bias=True).to(ctx.device)
        with pytest.raises(
            ValueError, match="Cannot auto-detect generic torch.nn.Module"
        ):
            distribute(module, ctx)
        with pytest.raises(TypeError, match="type_object='module' only supports"):
            distribute(module, ctx, type_object="module")
        return "success"


def test_distribute_module_autodetect_rejects_generic(device_config):
    test_args = {"device_mode": device_config["device_mode"]}
    results = run_distributed_test(
        _test_distribute_module_autodetect_rejects_generic_worker,
        device_config,
        test_args,
    )
    assert all(r == "success" for r in results)


def _test_processor_backward_worker(rank, world_size, args):
    """Worker for testing processor backward pass (gradients on parameters)."""
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        # Create Trainable Denoiser
        denoiser = TrainableDenoiser(channels=3).to(ctx.device)

        # We need to share the SAME initial weights across ranks to ensure determinism
        # Since we use seed=42 in context, and TrainableDenoiser inits from fixed values,
        # let's double check simple syncing or just rely on manual init in TrainableDenoiser
        # TrainableDenoiser uses fill_, so it's deterministic.

        distributed_processor = distribute(
            denoiser,
            ctx,
            type_object="denoiser",
            tiling_strategy="overlap_tiling",
            patch_size=8,
            overlap=2,
        )

        # Reference
        denoiser_ref = TrainableDenoiser(channels=3).to(ctx.device)
        # Ensure weights allow perfect comparison (though they should be identical by init)
        denoiser_ref.load_state_dict(denoiser.state_dict())

        # Ensure identical initialization across ranks while preserving outer RNG state.
        fork_devices = [ctx.device.index] if ctx.device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(
                1234
            )  # Shared seed (overrides DistributedContext diversity)
            x = torch.randn(1, 3, 16, 16, device=ctx.device, requires_grad=True)

        x_ref = x.clone().detach().requires_grad_(True)

        # 1. Distributed Forward & Backward
        # Test standard optimizer integration
        optimizer = torch.optim.SGD(denoiser.parameters(), lr=1e-3)
        optimizer.zero_grad()

        out_dist = distributed_processor(x)
        # Normalize sum by world_size because we are computing loss on all ranks (Replicated Loss).
        # Since dist_nn.all_reduce sums gradients from all ranks, we need to scale down to match Single-GPU expectation.
        loss_dist = out_dist.sum()
        loss_dist.backward()

        # Optimizer step should work now as gradients are synced
        optimizer.step()

        # Get gradients of parameters
        grads_dist = []
        for param in denoiser.parameters():
            if param.grad is not None:
                grads_dist.append(param.grad.clone())

        # Also check input gradient
        x_grad_dist = x.grad.clone()

        if len(grads_dist) == 0:
            return "failure: no param gradients"

        if x_grad_dist is None:
            return "failure: no input gradients"

        # Check against reference simulated LOCALLY
        local_weight_grad = denoiser.conv.weight.grad
        local_bias_grad = denoiser.conv.bias.grad

        # Gradient reduction is now AUTOMATIC via DistributedParameterSync!
        # if ctx.use_dist:
        #    # Reduce gradients to see if they match the global run
        #    dist.all_reduce(local_weight_grad, op=dist.ReduceOp.SUM)
        #    dist.all_reduce(local_bias_grad, op=dist.ReduceOp.SUM)

        # Now local_weight_grad contains the SUM of gradients from all ranks.
        # This should match the gradient if we ran the whole thing on one GPU.

        # Let's compute the Single-GPU reference locally
        from deepinv.distributed.strategies import create_strategy

        strategy = create_strategy(
            "overlap_tiling", x_ref.shape, patch_size=8, overlap=2
        )
        # All patches
        num_patches = strategy.get_num_patches()
        all_indices = list(range(num_patches))

        # Extract
        patch_pairs = strategy.get_local_patches(x_ref, all_indices)
        all_patches = [p for _, p in patch_pairs]

        # Process using the same batching path as DistributedProcessing to avoid
        # backend-dependent numeric drift between batched and per-patch conv calls.
        batched_patches = strategy.apply_batching(
            all_patches, max_batch_size=distributed_processor.max_batch_size
        )
        processed_batches = [denoiser_ref(batch) for batch in batched_patches]
        processed = strategy.unpack_batched_results(processed_batches, len(all_patches))

        # Reconstruct
        out_ref = torch.zeros_like(x_ref)
        strategy.reduce_patches(out_ref, list(zip(all_indices, processed)))

        # Loss
        loss_ref = out_ref.sum()
        loss_ref.backward()

        # Compare reduced distributed grads with reference grads
        assert torch.allclose(
            local_weight_grad, denoiser_ref.conv.weight.grad, atol=1e-5
        ), f"Weight grad mismatch! Dist: {local_weight_grad.norm()}, Ref: {denoiser_ref.conv.weight.grad.norm()}"

        assert torch.allclose(local_bias_grad, denoiser_ref.conv.bias.grad, atol=1e-5)

        return "success"


def _test_processor_backward_multiple_calls_worker(rank, world_size, args):
    """Worker ensuring one param sync per backward graph with multiple processor calls."""
    with DistributedContext(device_mode=args["device_mode"], seed=42) as ctx:
        denoiser = TrainableDenoiser(channels=3).to(ctx.device)
        distributed_processor = distribute(
            denoiser,
            ctx,
            type_object="denoiser",
            tiling_strategy="overlap_tiling",
            patch_size=8,
            overlap=2,
        )

        denoiser_ref = TrainableDenoiser(channels=3).to(ctx.device)
        denoiser_ref.load_state_dict(denoiser.state_dict())

        fork_devices = [ctx.device.index] if ctx.device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(1234)
            x = torch.randn(1, 3, 16, 16, device=ctx.device, requires_grad=True)
        x_ref = x.clone().detach().requires_grad_(True)

        # Two distributed calls in the same graph and one backward.
        out_dist_1 = distributed_processor(x)
        out_dist_2 = distributed_processor(x)
        loss_dist = out_dist_1.sum() + out_dist_2.sum()
        loss_dist.backward()

        local_weight_grad = denoiser.conv.weight.grad
        local_bias_grad = denoiser.conv.bias.grad
        if local_weight_grad is None or local_bias_grad is None:
            return "failure: no param gradients"

        from deepinv.distributed.strategies import create_strategy

        strategy = create_strategy(
            "overlap_tiling", x_ref.shape, patch_size=8, overlap=2
        )
        num_patches = strategy.get_num_patches()
        all_indices = list(range(num_patches))
        patch_pairs = strategy.get_local_patches(x_ref, all_indices)
        all_patches = [p for _, p in patch_pairs]

        out_ref_1 = torch.zeros_like(x_ref)
        batched_patches_1 = strategy.apply_batching(
            all_patches, max_batch_size=distributed_processor.max_batch_size
        )
        processed_batches_1 = [denoiser_ref(batch) for batch in batched_patches_1]
        processed_1 = strategy.unpack_batched_results(
            processed_batches_1, len(all_patches)
        )
        strategy.reduce_patches(out_ref_1, list(zip(all_indices, processed_1)))

        out_ref_2 = torch.zeros_like(x_ref)
        batched_patches_2 = strategy.apply_batching(
            all_patches, max_batch_size=distributed_processor.max_batch_size
        )
        processed_batches_2 = [denoiser_ref(batch) for batch in batched_patches_2]
        processed_2 = strategy.unpack_batched_results(
            processed_batches_2, len(all_patches)
        )
        strategy.reduce_patches(out_ref_2, list(zip(all_indices, processed_2)))

        loss_ref = out_ref_1.sum() + out_ref_2.sum()
        loss_ref.backward()

        assert torch.allclose(
            local_weight_grad, denoiser_ref.conv.weight.grad, atol=1e-5
        )
        assert torch.allclose(local_bias_grad, denoiser_ref.conv.bias.grad, atol=1e-5)
        return "success"


@pytest.mark.parametrize("tiling_strategy", ["overlap_tiling"])
def test_distributed_processor_backward(device_config, tiling_strategy):
    """
    Test that gradients flow correctly through DistributedProcessing (trainable denoiser).
    """
    test_args = {
        "device_mode": device_config["device_mode"],
        "tiling_strategy": tiling_strategy,
    }
    results = run_distributed_test(
        _test_processor_backward_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)


def test_distributed_processor_backward_multiple_calls(device_config):
    """
    Test gradients with two DistributedProcessing calls in one backward graph.
    """
    test_args = {"device_mode": device_config["device_mode"]}
    results = run_distributed_test(
        _test_processor_backward_multiple_calls_worker, device_config, test_args
    )
    assert all(r == "success" for r in results)
