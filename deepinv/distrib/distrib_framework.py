import os
from typing import Callable, Optional, Union, Sequence

import torch
import torch.distributed as dist

from deepinv.physics import Physics, LinearPhysics
from deepinv.optim import Prior
from deepinv.utils.tensorlist import TensorList

from .distribution_strategies.strategies import DistributedSignalStrategy

Index = tuple[Union[slice, int], ...]


# =========================
# Distributed Context
# =========================
class DistributedContext:
    r"""
    Small, opinionated context manager for distributed runs.

    Handles:
      - Init/destroy process group (if RANK/WORLD_SIZE envs exist)
      - Backend choice: NCCL when one-GPU-per-process per node, else Gloo
      - Device selection based on LOCAL_RANK and visible GPUs
      - Sharding helpers and tiny comm helpers

    :param str backend: backend to use for distributed communication. If `None`, automatically selects NCCL for GPU or Gloo for CPU.
    :param str sharding: sharding strategy for data distribution. Options are `'round_robin'` and `'block'`.
    :param bool cleanup: whether to clean up the process group on exit.
    :param None, int seed: random seed for reproducible results. If provided, each rank gets seed + rank.
    :param bool deterministic: whether to use deterministic cuDNN operations.
    :param None, str device_mode: device selection mode. Options are `'cpu'`, `'gpu'`, or `None` for automatic.
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        sharding: str = "round_robin",
        cleanup: bool = True,
        seed: Optional[int] = None,
        deterministic: bool = False,
        device_mode: Optional[str] = None,
    ):
        r"""
        Initialize the distributed context manager.

        :param str backend: backend to use for distributed communication. If `None`, automatically selects NCCL for GPU or Gloo for CPU.
        :param str sharding: sharding strategy for data distribution. Options are `'round_robin'` and `'block'`.
        :param bool cleanup: whether to clean up the process group on exit.
        :param None, int seed: random seed for reproducible results. If provided, each rank gets seed + rank.
        :param bool deterministic: whether to use deterministic cuDNN operations.
        :param None, str device_mode: device selection mode. Options are `'cpu'`, `'gpu'`, or `None` for automatic.
        """
        self.backend = backend
        self.sharding = sharding
        self.cleanup = cleanup
        self.seed = seed
        self.deterministic = deterministic
        self.device_mode = device_mode  # "cpu", "gpu", or None (auto)

        # set in __enter__
        self.initialized_here = False
        self.is_dist = False
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.local_world_size = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __enter__(self):
        # Detect whether we should initialize a process group
        env_has_dist = ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)
        should_init_pg = (not dist.is_initialized()) and env_has_dist

        # Count GPUs *visible to this process* (respects CUDA_VISIBLE_DEVICES)
        visible_gpus = torch.cuda.device_count()
        cuda_ok = torch.cuda.is_available() and (visible_gpus > 0)

        # Processes per node
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        # Ranks
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
                        raise RuntimeError("GPU mode requested but NCCL backend not available")
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
            self.initialized_here = True

        # Refresh flags from the actual PG (in case we didn't init)
        self.is_dist = dist.is_initialized()
        if self.is_dist:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()

        # ---- Device selection ----
        if self.device_mode == "cpu":
            # Force CPU mode
            self.device = torch.device("cpu")
        elif self.device_mode == "gpu":
            # Force GPU mode (require CUDA)
            if not torch.cuda.is_available() or visible_gpus == 0:
                raise RuntimeError(
                    "GPU mode requested but CUDA not available or no visible GPUs"
                )
            dev_index = self.local_rank % visible_gpus
            self.device = torch.device(f"cuda:{dev_index}")
            torch.cuda.set_device(self.device)
        else:
            # Auto mode (original behavior)
            if torch.cuda.is_available() and visible_gpus > 0:
                # map local_rank onto *visible* devices
                dev_index = self.local_rank % visible_gpus
                self.device = torch.device(f"cuda:{dev_index}")
                torch.cuda.set_device(self.device)
            else:
                self.device = torch.device("cpu")

        self._post_init_setup()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.cleanup and self.initialized_here and dist.is_initialized():
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
            s = self.seed + (self.rank if self.is_dist else 0)
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
        Get local indices for this rank based on sharding strategy.

        :param int num_items: total number of items to shard.
        :return: (:class:`list[int]`) list of indices assigned to this rank.
        """
        if self.sharding == "round_robin":
            indices = [
                i for i in range(num_items) if (i % self.world_size) == self.rank
            ]
        elif self.sharding == "block":
            per_rank = (num_items + self.world_size - 1) // self.world_size
            start = self.rank * per_rank
            end = min(start + per_rank, num_items)
            indices = list(range(start, end))
        else:
            raise ValueError("sharding must be either 'round_robin' or 'block'.")

        # Warning for efficiency, but allow empty indices (don't raise error)
        if self.is_dist and len(indices) == 0 and self.rank == 0:
            print(
                f"Warning: Some ranks have no work items to process "
                f"(num_items={num_items}, world_size={self.world_size}). "
                f"Consider reducing world_size or increasing the workload for better efficiency."
            )

        return indices

    # ----------------------
    # Collectives
    # ----------------------
    def all_reduce_(self, t: torch.Tensor, op: str = "sum"):
        r"""
        In-place all_reduce (returns `t`). Uses SUM for both cases; divides for mean.
        Works on both Gloo and NCCL for all dtypes that backend supports.

        :param torch.Tensor t: tensor to reduce.
        :param str op: reduction operation (`'sum'` or `'mean'`).
        :return: (:class:`torch.Tensor`) the reduced tensor.
        """
        if self.is_dist:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            if op == "mean":
                t /= float(self.world_size)
        return t

    def broadcast_(self, t: torch.Tensor, src: int = 0):
        if self.is_dist:
            dist.broadcast(t, src=src)
        return t

    def barrier(self):
        if self.is_dist:
            dist.barrier()


# =========================
# Distributed Measurements
# =========================
class DistributedMeasurements:
    r"""
    Holds only local measurement shards.

    You can supply either:
      - factory(i, device, shared) -> Tensor and num_items to build local shards
      - a full list of measurements (replicated across ranks); we'll select locals

    No collectives are used here; we assume each rank can construct its own locals.

    :param DistributedContext ctx: distributed context manager.
    :param None, int num_items: total number of measurement items. Required when using factory.
    :param None, Callable factory: factory function that creates measurements. Should have signature `factory(index, device, shared) -> torch.Tensor`.
    :param None, Sequence[torch.Tensor] measurements_list: list of all measurements to be distributed.
    :param None, dict shared: shared data dictionary passed to factory function.
    """

    def __init__(
        self,
        ctx: DistributedContext,
        num_items: Optional[int] = None,
        *,
        factory: Optional[
            Callable[[int, torch.device, Optional[dict]], torch.Tensor]
        ] = None,
        measurements_list: Optional[Sequence[torch.Tensor]] = None,
        shared: Optional[dict] = None,
    ):
        r"""
        Initialize distributed measurements.

        :param DistributedContext ctx: distributed context manager.
        :param None, int num_items: total number of measurement items. Required when using factory.
        :param None, Callable factory: factory function that creates measurements. Should have signature `factory(index, device, shared) -> torch.Tensor`.
        :param None, Sequence[torch.Tensor] measurements_list: list of all measurements to be distributed.
        :param None, dict shared: shared data dictionary passed to factory function.
        :param None, torch.dtype dtype: data type for measurements.
        """
        self.ctx = ctx
        self.shared = shared

        if (factory is None and measurements_list is None) or (
            factory is not None and measurements_list is not None
        ):
            raise ValueError("Provide either factory or measurements_list.")
        if num_items is None and factory is not None:
            raise ValueError("Must provide num_items if using factory.")

        self.num_items = num_items if num_items is not None else len(measurements_list)
        self.local_idx: list[int] = ctx.local_indices(num_items)
        self._global_to_local: dict[int, int] = {
            g: j for j, g in enumerate(self.local_idx)
        }

        self.local: list[torch.Tensor] = []
        if factory is not None:
            for i in self.local_idx:
                y = factory(i, ctx.device, shared)
                self.local.append(y.to(ctx.device, dtype=y.dtype))
        elif measurements_list is not None:
            for i in self.local_idx:
                self.local.append(measurements_list[i].to(ctx.device, dtype=measurements_list[i].dtype))
        else:
            raise ValueError("Provide factory or measurements_list.")

    def __len__(self):
        return self.num_items

    def indices(self) -> list[int]:
        return self.local_idx

    def get_local(self) -> list[torch.Tensor]:
        return self.local

    def get_by_global_index(self, i: int) -> torch.Tensor:
        """Returns the local tensor if owned; raises otherwise."""
        if i not in self._global_to_local:
            raise KeyError(f"Measurement {i} is not local to rank {self.ctx.rank}.")
        return self.local[self._global_to_local[i]]


# =========================
# Distributed Signal
# =========================
class DistributedSignal:
    r"""
    A wrapper around a replicated signal tensor with automatic synchronization.

    The signal is automatically synchronized across all processes after initialization
    and after any update operations. Users should not call sync_ manually.

    :param DistributedContext ctx: distributed context manager.
    :param Sequence[int] shape: shape of the signal tensor.
    :param None, torch.dtype dtype: data type for the signal tensor.
    :param None, torch.Tensor init: initial tensor data. If `None`, tensor is initialized with zeros.
    :param int sync_src: source rank for broadcasting during synchronization.

    |sep|

    :Examples:

        Create and update a distributed signal:

        >>> with DistributedContext() as ctx:
        ...     signal = DistributedSignal(ctx, (3, 32, 32))
        ...     signal.update_(torch.randn(3, 32, 32))  # Automatically synchronized
        ...     signal.add_(-0.1 * gradient)  # Automatically synchronized
    """

    def __init__(
        self,
        ctx: DistributedContext,
        shape: Sequence[int],
        dtype: Optional[torch.dtype] = None,
        init: Optional[torch.Tensor] = None,
        sync_src: int = 0,
    ):
        r"""
        Initialize the distributed signal.

        :param DistributedContext ctx: distributed context manager.
        :param Sequence[int] shape: shape of the signal tensor.
        :param None, torch.dtype dtype: data type for the signal tensor.
        :param None, torch.Tensor init: initial tensor data. If `None`, tensor is initialized with zeros.
        :param int sync_src: source rank for broadcasting during synchronization.
        """
        self.ctx = ctx
        self.dtype = dtype or torch.float32
        self._shape = torch.Size(shape)
        self._sync_src = sync_src

        if init is None:
            self._data = torch.zeros(self._shape, device=ctx.device, dtype=self.dtype)
        else:
            self.dtype = init.dtype
            self._data = init.to(ctx.device, dtype=init.dtype)

        # Auto-sync after initialization
        self._sync()

    def _sync(self):
        """Internal synchronization method."""
        if self.ctx.is_dist:
            self.ctx.broadcast_(self._data, src=self._sync_src)

    def update_(self, new_data: torch.Tensor):
        """Update signal data and automatically sync across processes."""
        self._data.copy_(new_data)
        self._sync()
        return self

    def copy_(self, other):
        """Copy from another tensor/signal with automatic sync."""
        if isinstance(other, DistributedSignal):
            self._data.copy_(other._data)
        else:
            self._data.copy_(other)
        self._sync()
        return self

    def clone(self):
        """Clone the signal (creates new DistributedSignal)."""
        new_signal = DistributedSignal(
            self.ctx, self.shape, self.dtype, sync_src=self._sync_src
        )
        new_signal._data.copy_(self._data)
        new_signal._sync()
        return new_signal

    def all_reduce_(self, op: str = "mean"):
        """Optional: consensus ops if your algorithm needs it."""
        self.ctx.all_reduce_(self._data, op=op)
        return self

    @property
    def shape(self) -> torch.Size:
        return self._data.shape

    @property
    def data(self) -> torch.Tensor:
        """Access to underlying tensor data (read-only recommended)."""
        return self._data

    @data.setter
    def data(self, value: torch.Tensor):
        """Set data with automatic sync."""
        self._data = value.to(self.ctx.device, dtype=self.dtype)
        self._sync()

    @property
    def tensor(self) -> torch.Tensor:
        """Access to underlying tensor (alias for data)."""
        return self._data


# =========================
# Distributed Physics
# =========================
class DistributedPhysics(Physics):
    r"""
    Holds only local physics operators. Exposes fast local and compatible global APIs.

    This class distributes physics operators across multiple processes, where each process
    owns a subset of the operators. It provides both efficient local operations and
    compatibility methods that gather results globally.

    :param DistributedContext ctx: distributed context manager.
    :param int num_ops: total number of physics operators.
    :param Callable factory: factory function that creates physics operators. Should have signature `factory(index, device, shared) -> Physics`.
    :param None, dict shared: shared data dictionary passed to factory function.
    :param None, torch.dtype dtype: data type for operations.
    """

    def __init__(
        self,
        ctx: DistributedContext,
        num_ops: int,
        factory: Callable[[int, torch.device, Optional[dict]], Physics],
        *,
        shared: Optional[dict] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        r"""
        Initialize distributed physics operators.

        :param DistributedContext ctx: distributed context manager.
        :param int num_ops: total number of physics operators.
        :param Callable factory: factory function that creates physics operators. Should have signature `factory(index, device, shared) -> Physics`.
        :param None, dict shared: shared data dictionary passed to factory function.
        :param None, torch.dtype dtype: data type for operations.
        """
        super().__init__(**kwargs)
        self.ctx = ctx
        self.dtype = dtype or torch.float32
        self.num_ops = num_ops
        self.local_idx: list[int] = ctx.local_indices(num_ops)

        # Broadcast shared object (lightweight) once (root=0) if present
        self.shared = shared
        if ctx.is_dist and shared is not None:
            obj = [shared if ctx.rank == 0 else None]
            dist.broadcast_object_list(obj, src=0)
            self.shared = obj[0]

        # Build local physics
        self.local_physics: list[Physics] = []
        for i in self.local_idx:
            p_i = factory(i, ctx.device, self.shared)
            self.local_physics.append(p_i)

    # -------- local fast path --------
    def A_local(self, x: torch.Tensor, **kwargs) -> list[torch.Tensor]:
        return [p.A(x, **kwargs) for p in self.local_physics]

    # Optional: global assembly (for compatibility/debug)
    def A(self, x: torch.Tensor, **kwargs) -> TensorList:
        pairs = list(zip(self.local_idx, self.A_local(x, **kwargs), strict=False))
        if not self.ctx.is_dist:
            out = [None] * self.num_ops
            for i, t in pairs:
                out[i] = t
            return TensorList(out)

        # NOTE: use all_gather_object only for small N or debugging.
        # For production, prefer tensor collectives with metadata. Kept simple here.
        gathered = [None] * self.ctx.world_size
        # All ranks participate in all_gather_object, even if pairs is empty
        dist.all_gather_object(gathered, pairs)
        all_pairs = []
        for part in gathered:
            all_pairs.extend(part)
        out = [None] * self.num_ops
        for i, t in all_pairs:
            out[i] = t
        return TensorList(out)

    def forward(self, x, **kwargs):
        return self.A(x, **kwargs)


class DistributedLinearPhysics(DistributedPhysics, LinearPhysics):
    r"""
    Linear extension with local adjoint / vjp that reduce with a single all_reduce.

    This class extends DistributedPhysics for linear operators, providing efficient
    distributed implementations of adjoint and vector-Jacobian product operations.

    :param DistributedContext ctx: distributed context manager.
    :param int num_ops: total number of physics operators.
    :param Callable factory: factory function that creates linear physics operators.
    :param None, dict shared: shared data dictionary passed to factory function.
    :param str reduction: reduction mode for distributed operations. Options are `'sum'` and `'mean'`.
    :param None, torch.dtype dtype: data type for operations.
    """

    def __init__(
        self,
        ctx: DistributedContext,
        num_ops: int,
        factory,
        *,
        shared: Optional[dict] = None,
        reduction: str = "sum",
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        r"""
        Initialize distributed linear physics operators.

        :param DistributedContext ctx: distributed context manager.
        :param int num_ops: total number of physics operators.
        :param Callable factory: factory function that creates linear physics operators.
        :param None, dict shared: shared data dictionary passed to factory function.
        :param str reduction: reduction mode for distributed operations. Options are `'sum'` and `'mean'`.
        :param None, torch.dtype dtype: data type for operations.
        """
        LinearPhysics.__init__(self, A=lambda x: x, A_adjoint=lambda y: y, **kwargs)
        self.reduction_mode = reduction
        super().__init__(ctx, num_ops, factory, shared=shared, dtype=dtype, **kwargs)

        for p in self.local_physics:
            if not isinstance(p, LinearPhysics):
                raise ValueError("factory must return LinearPhysics instances.")

    # ---- local (fast) ----
    def A_adjoint_local(self, y_local: list[torch.Tensor], **kwargs) -> torch.Tensor:
        if len(y_local) == 0:
            # Return zeros with proper shape for empty local set
            return torch.zeros((), device=self.ctx.device, dtype=self.dtype)
        contribs = [
            p.A_adjoint(y_i, **kwargs)
            for p, y_i in zip(self.local_physics, y_local, strict=False)
        ]
        return torch.stack(contribs, dim=0).sum(0)

    def A_vjp_local(
        self, x: torch.Tensor, v_local: list[torch.Tensor], **kwargs
    ) -> torch.Tensor:
        if len(v_local) == 0:
            # Return zeros with proper shape for empty local set
            return torch.zeros_like(x)
        contribs = [
            p.A_vjp(x, v_i, **kwargs)
            for p, v_i in zip(self.local_physics, v_local, strict=False)
        ]
        return torch.stack(contribs, dim=0).sum(0)

    # ---- global (compat) ----
    def _reduce_global(self, x_like: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x_like):
            # handle 0.0 placeholder for empty local set
            x_like = torch.zeros((), device=self.ctx.device, dtype=self.dtype)
            x_like = x_like.expand(())  # scalar

        # Ensure all ranks participate in collective even with empty data
        if self.ctx.is_dist:
            # For ranks with empty local sets, x_like should be zeros
            self.ctx.all_reduce_(x_like, op="sum")

        if self.reduction_mode == "mean":
            x_like = x_like / float(self.num_ops)
        return x_like

    def A_adjoint(self, y: TensorList, **kwargs) -> torch.Tensor:
        y_local = [y[i] for i in self.local_idx]
        local = self.A_adjoint_local(y_local, **kwargs)
        return self._reduce_global(local)

    def A_vjp(self, x: torch.Tensor, v: TensorList, **kwargs) -> torch.Tensor:
        v_local = [v[i] for i in self.local_idx]
        local = self.A_vjp_local(x, v_local, **kwargs)
        return self._reduce_global(local)


# =========================
# Distributed Data Fidelity
# =========================
class DistributedDataFidelity:
    r"""
    Truly distributed data fidelity.

    Builds/owns only local fidelity blocks (by index):
      - computes loss locally and all-reduces a single scalar
      - computes grad via local VJP and all-reduces a single x-shaped tensor

    :param DistributedContext ctx: distributed context manager.
    :param Union[DistributedPhysics, DistributedLinearPhysics] physics: distributed physics operators.
    :param DistributedMeasurements measurements: distributed measurements.
    :param None, Callable data_fidelity_factory: factory function for creating data fidelity terms. Should have signature `factory(index, device, shared) -> object`.
    :param None, Sequence[object] data_fidelity_list: list of all data fidelity terms to be distributed.
    :param None, dict shared: shared data dictionary passed to factory function.
    :param str reduction: reduction mode for loss aggregation. Options are `'sum'` and `'mean'`.
    """

    def __init__(
        self,
        ctx: DistributedContext,
        physics: Union[DistributedPhysics, DistributedLinearPhysics],
        measurements: DistributedMeasurements,
        *,
        data_fidelity_factory: Optional[
            Callable[[int, torch.device, Optional[dict]], object]
        ] = None,
        data_fidelity_list: Optional[Sequence[object]] = None,
        shared: Optional[dict] = None,
        reduction: str = "sum",
    ):
        r"""
        Initialize distributed data fidelity.

        :param DistributedContext ctx: distributed context manager.
        :param Union[DistributedPhysics, DistributedLinearPhysics] physics: distributed physics operators.
        :param DistributedMeasurements measurements: distributed measurements.
        :param None, Callable data_fidelity_factory: factory function for creating data fidelity terms. Should have signature `factory(index, device, shared) -> object`.
        :param None, Sequence[object] data_fidelity_list: list of all data fidelity terms to be distributed.
        :param None, dict shared: shared data dictionary passed to factory function.
        :param str reduction: reduction mode for loss aggregation. Options are `'sum'` and `'mean'`.
        """
        self.ctx = ctx
        self.physics = physics
        self.meas = measurements
        self.reduction = reduction
        self.shared = shared

        local_idx = physics.local_idx
        if data_fidelity_factory is not None:
            self.local_df = [
                data_fidelity_factory(i, ctx.device, shared) for i in local_idx
            ]
        elif data_fidelity_list is not None:
            self.local_df = [data_fidelity_list[i] for i in local_idx]
        else:
            raise ValueError("Provide data_fidelity_factory or data_fidelity_list.")

        # Sanity: indices alignment
        assert (
            local_idx == self.meas.indices()
        ), "Physics and measurements must shard the same global indices under the same context."

    # ---- loss ----
    @torch.no_grad()
    def fn(self, X: DistributedSignal, **kwargs) -> torch.Tensor:
        # Local forward
        y_pred_local = self.physics.A_local(
            X.tensor, **kwargs
        )  # list aligned with local_idx
        y_true_local = self.meas.get_local()

        # Local accumulation (each DF returns a scalar per-batch; we sum)
        loss = X.tensor.new_zeros(())
        for df, yhat, y in zip(self.local_df, y_pred_local, y_true_local, strict=False):
            # Use the distance function directly since we already have A(x) = yhat
            loss = loss + df.d(yhat, y, **kwargs)

        # Global reduction - all ranks participate even if they have no local data
        if self.ctx.is_dist:
            self.ctx.all_reduce_(loss, op="sum")
        if self.reduction == "mean":
            loss = loss / float(self.physics.num_ops)
        return loss

    # ---- grad wrt x ----
    def grad(self, X: DistributedSignal, **kwargs) -> torch.Tensor:
        # Local forward (recompute or cache as you like)
        y_pred_local = self.physics.A_local(X.tensor, **kwargs)
        y_true_local = self.meas.get_local()

        # Local residuals: dℓ/dy_i
        v_local = []
        for df, yhat, y in zip(self.local_df, y_pred_local, y_true_local, strict=False):
            v_local.append(df.d.grad(yhat, y, **kwargs))

        # Local VJP into x-shape
        if isinstance(self.physics, DistributedLinearPhysics):
            x_grad_local = self.physics.A_vjp_local(X.tensor, v_local, **kwargs)
        else:
            # Fallback: generic physics with manual vjp (requires per-op support).
            if len(v_local) > 0:
                contribs = [
                    p.A_vjp(X.tensor, v_i, **kwargs)
                    for p, v_i in zip(self.physics.local_physics, v_local, strict=False)
                ]
                x_grad_local = torch.stack(contribs, dim=0).sum(0)
            else:
                x_grad_local = torch.zeros_like(X.tensor)

        if not torch.is_tensor(x_grad_local):
            x_grad_local = torch.zeros_like(X.tensor)

        # Global reduction - all ranks participate even if they have no local data
        if self.ctx.is_dist:
            self.ctx.all_reduce_(x_grad_local, op="sum")
        if self.reduction == "mean":
            x_grad_local = x_grad_local / float(self.physics.num_ops)

        return x_grad_local


class DistributedPrior:
    r"""
    Distributed prior using pluggable signal processing strategies.

    This class enables distributed processing of prior terms by using signal processing
    strategies that define how to split, process, and combine signal patches across
    multiple processes.

    :param DistributedContext ctx: distributed context manager.
    :param deepinv.optim.Prior prior: prior term to be applied in a distributed manner.
    :param Union[str, DistributedSignalStrategy] strategy: signal processing strategy. Either a strategy name (`'basic'`, `'smart_tiling'`) or a custom strategy instance.
    :param Sequence[int] signal_shape: full tensor shape of the signal to be processed (e.g. BCHW).
    :param None, dict strategy_kwargs: extra arguments for the strategy (when using string strategy names).
    """

    def __init__(
        self,
        ctx: DistributedContext,
        prior: Prior,
        *,
        strategy: Union[str, DistributedSignalStrategy] = "smart_tiling",
        signal_shape: Sequence[int],
        strategy_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        r"""
        Initialize distributed prior.

        :param DistributedContext ctx: distributed context manager.
        :param deepinv.optim.Prior prior: prior term to be applied in a distributed manner.
        :param Union[str, DistributedSignalStrategy] strategy: signal processing strategy. Either a strategy name (`'basic'`, `'smart_tiling'`) or a custom strategy instance.
        :param Sequence[int] signal_shape: full tensor shape of the signal to be processed (e.g. BCHW).
        :param None, dict strategy_kwargs: extra arguments for the strategy (when using string strategy names).
        """
        self.ctx = ctx
        self.prior = prior

        if hasattr(prior, "to"):
            self.prior.to(ctx.device)

        self.signal_shape = torch.Size(signal_shape)

        # Create or set the strategy
        if isinstance(strategy, str):
            from .distribution_strategies.strategies import create_strategy

            strategy_kwargs = strategy_kwargs or {}
            self.strategy = create_strategy(strategy, signal_shape, **strategy_kwargs)
        else:
            self.strategy = strategy

        if self.strategy is None:
            raise RuntimeError("Strategy is None - failed to create or import strategy")

        self.num_patches = self.strategy.get_num_patches()
        if self.num_patches == 0:
            raise ValueError("Strategy produced zero patches.")

        # determine local patch indices for this rank
        self.local_indices: list[int] = list(self.ctx.local_indices(self.num_patches))

        # Check for insufficient work distribution
        if self.ctx.is_dist and self.ctx.rank == 0:
            ranks_with_work = sum(
                1
                for rank in range(self.ctx.world_size)
                if len(self.ctx.local_indices(self.num_patches)) > 0
            )
            if ranks_with_work < self.ctx.world_size:
                print(
                    f"Warning: Only {ranks_with_work}/{self.ctx.world_size} ranks have patches to process. "
                    f"Some ranks will have no work. "
                    f"Current: {self.num_patches} patches for {self.ctx.world_size} ranks."
                )

    # ---- public ops -------------------------------------------------------

    def grad(self, X: DistributedSignal, **kwargs) -> torch.Tensor:
        return self._apply_op(X, op="grad", **kwargs)

    def prox(self, X: DistributedSignal, **kwargs) -> torch.Tensor:
        return self._apply_op(X, op="prox", **kwargs)

    # ---- internals --------------------------------------------------------

    def _apply_op(self, X: DistributedSignal, *, op: str, **kwargs) -> torch.Tensor:
        """
        Apply prior operation (grad/prox) using the distributed strategy.

        This method:
        1. Extracts local patches using the strategy
        2. Applies batching as defined by the strategy
        3. Applies the prior operation to batched patches
        4. Unpacks and reduces results using the strategy
        5. All-reduces the final result across ranks
        """
        # Handle empty case early
        if not self.local_indices:
            out_local = torch.zeros(
                self.signal_shape, device=self.ctx.device, dtype=X.tensor.dtype
            )
            if self.ctx.is_dist:
                self.ctx.all_reduce_(out_local, op="sum")
            return out_local

        # Get the prior operation function
        if op == "grad":
            if not hasattr(self.prior, "grad"):
                raise ValueError("Prior does not implement .grad()")
            prior_fn = lambda t: self.prior.grad(t, **kwargs)
        elif op == "prox":
            if not hasattr(self.prior, "prox"):
                raise ValueError("Prior does not implement .prox()")
            prior_fn = lambda t: self.prior.prox(t, **kwargs)
        else:
            raise ValueError(f"Unknown operation: {op}")

        # 1. Extract local patches using strategy
        local_pairs = self.strategy.get_local_patches(X.tensor, self.local_indices)
        patches = [patch for _, patch in local_pairs]

        # 2. Apply batching strategy
        batched_patches = self.strategy.apply_batching(patches)

        # 3. Apply prior to each batch
        processed_batches = []
        for batch in batched_patches:
            result = prior_fn(batch)
            processed_batches.append(result)

        # 4. Unpack results back to individual patches
        processed_patches = self.strategy.unpack_batched_results(
            processed_batches, len(patches)
        )

        # 5. Pair with global indices
        if len(processed_patches) != len(self.local_indices):
            raise RuntimeError(
                f"Mismatch between processed patches ({len(processed_patches)}) "
                f"and local indices ({len(self.local_indices)})"
            )

        processed_pairs = list(zip(self.local_indices, processed_patches, strict=False))

        # 6. Initialize output tensor and apply reduction strategy
        out_local = torch.zeros(
            self.signal_shape, device=self.ctx.device, dtype=X.tensor.dtype
        )
        self.strategy.reduce_patches(out_local, processed_pairs)

        # 7. All-reduce to combine results from all ranks
        if self.ctx.is_dist:
            self.ctx.all_reduce_(out_local, op="sum")

        return out_local
