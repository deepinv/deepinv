from __future__ import annotations

import os
from typing import Callable, Any
import warnings

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn


class DistributedContext:
    r"""
    Context manager for distributed computing.

    Handles:
      - Initialization/destruction of the process group (if `RANK` / `WORLD_SIZE` environment variables exist)
      - Backend choice: NCCL when one-GPU-per-process per node, else Gloo.
      - Device selection based on `LOCAL_RANK` and visible GPUs
      - Sharding helpers and tiny communication helpers

    :param str | None backend: backend to use for distributed communication. If `None` (default), automatically selects NCCL for GPU or Gloo for CPU.
    :param bool cleanup: whether to clean up the process group on exit. Default is `True`.
    :param int | None seed: random seed for reproducible results. If provided, behavior depends on `seed_offset`. Default is `None`.
    :param bool seed_offset: whether to add rank offset to seed (each rank gets `seed + rank`). Default is `True`.
        When `True`: each process uses a unique seed for diverse random sequences.
        When `False`: all processes share the same seed for synchronized randomness.
    :param bool deterministic: whether to use deterministic cuDNN operations. Default is `False`.
    :param str | None device_mode: device selection mode. Options are `'cpu'`, `'gpu'`, or `None` for automatic. Default is `None`.

    """

    def __init__(
        self,
        backend: str | None = None,
        cleanup: bool = True,
        seed: int | None = None,
        seed_offset: bool = True,
        deterministic: bool = False,
        device_mode: str | None = None,
    ):
        r"""
        Initialize the distributed context manager.
        """
        self.backend = backend
        self.cleanup = cleanup
        self.seed = seed
        self.seed_offset = seed_offset
        self.deterministic = deterministic
        self.device_mode = device_mode  # "cpu", "gpu", or None (auto)

        # set in __enter__
        self.created_dist = False
        self.use_dist = False
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.local_world_size = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dist_wrapper_cache: dict[str, Callable[..., Any]] = {}
        self._param_sync_scheduled_tasks: set[int] = set()
        self._param_sync_pending: dict[int, dict[int, torch.nn.Parameter]] = {}

    def __enter__(self):
        # Detect whether we should initialize a process group
        env_has_dist = ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)
        should_init_pg = (not dist.is_initialized()) and env_has_dist

        # Count GPUs *visible to this process* (respects CUDA_VISIBLE_DEVICES)
        visible_gpus = torch.cuda.device_count()
        cuda_ok = torch.cuda.is_available() and (visible_gpus > 0)

        # Get number of processes and ranks
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        if should_init_pg:
            backend = self.backend
            if backend is None:
                # Backend decision considering device_mode:
                #   - If device_mode is "cpu", always use Gloo
                #   - If device_mode is "gpu", always use NCCL (will fail if no GPU)
                #   - If auto (None), decide based on available resources:
                #     * If each node has at least as many *visible* GPUs as processes per node -> NCCL
                #     * Otherwise -> Gloo (e.g., GPU oversubscription or CPU)
                if self.device_mode == "cpu":
                    backend = "gloo"
                elif self.device_mode == "gpu":
                    if not dist.is_nccl_available():
                        raise RuntimeError(
                            "GPU mode requested but NCCL backend not available"
                        )
                    backend = "nccl"
                else:
                    # Auto mode
                    if (
                        cuda_ok
                        and dist.is_nccl_available()
                        and (self.local_world_size <= visible_gpus)
                    ):
                        backend = "nccl"
                    else:
                        backend = "gloo"

            dist.init_process_group(backend=backend)
            self.created_dist = True

        # Refresh flags from the actual PG (in case we didn't init)
        self.use_dist = dist.is_initialized()
        if self.use_dist:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()

        # ---- Device selection ----
        if self.device_mode == "cpu":
            # Force CPU mode
            self.device = torch.device("cpu")
        elif self.device_mode == "gpu":
            # Force GPU mode (require CUDA)
            if not cuda_ok:
                raise RuntimeError(
                    "GPU mode requested but CUDA not available or no visible GPUs"
                )
            if visible_gpus == 1:
                # GPU isolation is handled externally - use the only visible GPU
                dev_index = 0
            else:
                # Multiple GPUs visible - map local_rank to device index
                dev_index = self.local_rank % visible_gpus
            self.device = torch.device(f"cuda:{dev_index}")
            torch.cuda.set_device(self.device)
        else:
            # Auto mode
            if cuda_ok:
                # When CUDA_VISIBLE_DEVICES is set externally (e.g., by SLURM/submitit),
                # each process sees only its assigned GPU(s). In this case, if there's
                # only 1 visible GPU per process, just use cuda:0 for all processes.
                # Otherwise, map local_rank onto visible devices.
                if visible_gpus == 1:
                    # GPU isolation is handled externally - use the only visible GPU
                    dev_index = 0
                else:
                    # Multiple GPUs visible - map local_rank to device index
                    dev_index = self.local_rank % visible_gpus

                self.device = torch.device(f"cuda:{dev_index}")
                torch.cuda.set_device(self.device)
            else:
                self.device = torch.device("cpu")

        self._post_init_setup()
        return self

    def __exit__(self, exc_type, exc, tb):
        # Only destroy process group if:
        # 1. cleanup=True (caller wants cleanup)
        # 2. We initialized it (created_dist=True)
        # 3. It's still initialized
        if self.cleanup and self.created_dist and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()

    # ----------------------
    # Post-init knobs
    # ----------------------
    def _post_init_setup(self):
        # Seeding (rank offset for reproducible-but-unique RNG per process)
        if self.seed is not None:
            s = self.seed + (self.rank if (self.use_dist and self.seed_offset) else 0)
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)

        # Deterministic cuDNN
        if self.deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    # ----------------------
    # Sharding
    # ----------------------
    def local_indices(self, num_items: int) -> list[int]:
        r"""
        Get local indices for this rank based on round robin sharding.

        :param int num_items: total number of items to shard.
        :return: list of indices assigned to this rank.
        """
        indices = [i for i in range(num_items) if (i % self.world_size) == self.rank]

        # Warning for efficiency, but allow empty indices (don't raise error)
        if self.use_dist and len(indices) == 0:  # and self.rank == 0:
            warnings.warn(
                f"Rank {self.rank} has no work items to process "
                f"(num_items={num_items}, world_size={self.world_size}). "
                f"Consider reducing world_size or increasing the workload for better efficiency.",
                UserWarning,
            )

        return indices

    # ----------------------
    # Collectives
    # ----------------------
    def _collective(
        self, fn: Callable, fn_functional: Callable, x: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        if not self.use_dist:
            return x

        # Training/autograd path: return functional output (no copy-back).
        if x.requires_grad:
            if fn_functional is None:
                raise AttributeError(
                    f"No functional autograd path available for '{fn.__name__}'"
                )
            return fn_functional(x, **kwargs)

        # Inference path: use in-place distributed op for memory efficiency.
        fn(x, **kwargs)
        return x

    def all_reduce(
        self,
        x: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
        group=None,
    ) -> torch.Tensor:
        return self._collective(
            dist.all_reduce, dist_nn.all_reduce, x, op=op, group=group
        )

    def broadcast(self, x: torch.Tensor, src: int = 0, group=None) -> torch.Tensor:
        return self._collective(
            dist.broadcast, dist_nn.broadcast, x, src=src, group=group
        )

    def all_gather(self, x: torch.Tensor, group=None) -> torch.Tensor:
        r"""
        Gather one tensor per rank and return a stacked tensor of shape
        ``(world_size, *x.shape)``.
        """
        if not self.use_dist:
            return x.unsqueeze(0)

        if x.requires_grad:
            try:
                gathered = dist_nn.all_gather(x, group=group)
            except AttributeError as e:
                raise AttributeError(
                    "No functional autograd path available for 'all_gather'"
                ) from e
            if isinstance(gathered, torch.Tensor):
                return gathered
            return torch.stack(list(gathered), dim=0)

        out_list = [torch.empty_like(x) for _ in range(self.world_size)]
        dist.all_gather(out_list, x, group=group)
        return torch.stack(out_list, dim=0)

    def all_gather_object(self, obj_list: list, obj: Any, group=None):
        if not self.use_dist:
            if len(obj_list) > 0:
                obj_list[0] = obj
            return None
        return dist.all_gather_object(obj_list, obj, group=group)

    def broadcast_object_list(self, object_list: list, src: int = 0, device=None):
        if not self.use_dist:
            return object_list
        if device is None:
            return dist.broadcast_object_list(object_list, src=src)
        return dist.broadcast_object_list(object_list, src=src, device=device)

    def __getattr__(self, name):
        if name in self._dist_wrapper_cache:
            return self._dist_wrapper_cache[name]

        # Fallback to dist for less common APIs (barrier, new_group, etc.).
        if hasattr(dist, name):

            def wrapper(*args, **kwargs):
                if self.use_dist:
                    return getattr(dist, name)(*args, **kwargs)
                return None

            self._dist_wrapper_cache[name] = wrapper
            return wrapper
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
