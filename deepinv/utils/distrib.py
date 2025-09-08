from __future__ import annotations
import os
import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.distributed as dist

@dataclass
class DistInfo:
    is_dist: bool
    backend: Optional[str]
    rank: int
    world_size: int
    local_rank: Optional[int]
    device: torch.device

class DistributedContext:
    """
    A context manager that initializes torch.distributed (if needed),
    sets the CUDA device, seeds, and tears down cleanly.

    Usage:
        with DistributedContext(seed=1234) as ctx:
            print(ctx.info.rank, ctx.info.world_size, ctx.info.device)
            # construct DistributedPhysics / DistributedLinearPhysics here
    """
    def __init__(
        self,
        backend: Optional[str] = None,         # default: 'nccl' if CUDA else 'gloo'
        set_cuda_device: bool = True,          # set torch.cuda.set_device(local_rank)
        seed: Optional[int] = None,            # seed base, offset by rank if provided
        deterministic: bool = False,           # cuDNN knobs
        timeout_sec: int = 600,                # init_process_group timeout
        manage_process_group: bool = True,     # if already initialized, don't destroy unless we created it
        verbose: bool = False,
    ):
        self.backend = backend
        self.set_cuda_device = set_cuda_device
        self.seed = seed
        self.deterministic = deterministic
        self.timeout_sec = timeout_sec
        self.manage_process_group = manage_process_group
        self.verbose = verbose

        self._created_pg = False
        self.info: DistInfo = DistInfo(
            is_dist=False, backend=None, rank=0, world_size=1, local_rank=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )

    @staticmethod
    def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
        try:
            return int(os.environ[name])
        except KeyError:
            return default

    @staticmethod
    def _pick_backend() -> str:
        return "nccl" if torch.cuda.is_available() else "gloo"

    def __enter__(self) -> "DistributedContext":
        # If a process group already exists, attach to it (no-op init)
        if dist.is_initialized():
            rank = dist.get_rank()
            world = dist.get_world_size()
            local = self._env_int("LOCAL_RANK", 0 if torch.cuda.is_available() else None)
            if self.set_cuda_device and torch.cuda.is_available() and local is not None:
                torch.cuda.set_device(local % max(1, torch.cuda.device_count()))
            device = torch.device(f"cuda:{local}") if (torch.cuda.is_available() and local is not None) else torch.device("cpu")
            self.info = DistInfo(True, dist.get_backend(), rank, world, local, device)
            if self.verbose and (rank == 0):
                print(f"[DistributedContext] Attached to existing PG: backend={dist.get_backend()}, rank={rank}/{world}, device={self.info.device}")
            self._post_init_setup()
            return self

        # Otherwise, decide if we need to initialize distributed
        rank = self._env_int("RANK", 0)
        world = self._env_int("WORLD_SIZE", 1)
        local = self._env_int("LOCAL_RANK", 0 if torch.cuda.is_available() else None)

        is_multi = (world is not None and world > 1)
        backend = self.backend or self._pick_backend()

        # Set device *before* NCCL init
        if self.set_cuda_device and torch.cuda.is_available() and local is not None:
            torch.cuda.set_device(local % max(1, torch.cuda.device_count()))

        if is_multi:
            # Initialize process group using env vars (torchrun)
            kwargs: Dict[str, Any] = {"backend": backend}
            # timeout supported as timedelta in recent PyTorch
            try:
                kwargs["timeout"] = datetime.timedelta(seconds=self.timeout_sec)
            except Exception:
                pass
            dist.init_process_group(**kwargs)  # env:// is default when RANK/WORLD_SIZE present
            self._created_pg = self.manage_process_group

            device = torch.device(f"cuda:{local}") if (torch.cuda.is_available() and local is not None) else torch.device("cpu")
            self.info = DistInfo(True, backend, dist.get_rank(), dist.get_world_size(), local, device)
            if self.verbose and self.info.rank == 0:
                print(f"[DistributedContext] Initialized PG: backend={backend}, world={self.info.world_size}, device={device}")
        else:
            # Single-process fallback: no PG
            device = torch.device(f"cuda:{local}") if (torch.cuda.is_available() and local is not None) else torch.device("cpu")
            self.info = DistInfo(False, None, 0, 1, local, device)
            if self.verbose:
                print(f"[DistributedContext] Single-process mode on device={device}")

        self._post_init_setup()
        if self.info.is_dist:
            dist.barrier()
        return self

    def _post_init_setup(self):
        # Seeding
        if self.seed is not None:
            s = self.seed + (self.info.rank if self.info.is_dist else 0)
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)

        # cuDNN knobs
        if self.deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def __exit__(self, exc_type, exc, tb):
        # Cleanly tear down only if we created/own the PG
        if self.info.is_dist and self._created_pg:
            try:
                dist.barrier()
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception:
                pass
            if self.verbose and (self.info.rank == 0):
                print("[DistributedContext] Destroyed process group")

    @property
    def rank(self) -> int:
        return self.info.rank

    @property
    def world_size(self) -> int:
        return self.info.world_size

    def is_rank0(self) -> bool:
        return (not self.info.is_dist) or (self.info.rank == 0)

    def devices_for_process(self) -> list[torch.device]:
        """
        Return the list of devices this process might want to use.
        Typical torchrun uses 1 GPU per process (this device),
        but you can return multiple GPUs if your code does intra-process multi-GPU.
        """
        if not torch.cuda.is_available():
            return [torch.device("cpu")]
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
