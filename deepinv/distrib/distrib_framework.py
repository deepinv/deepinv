import os
from typing import Callable, Optional, List, Dict, Tuple, Union, Sequence

import torch
import torch.distributed as dist

from deepinv.physics import Physics, LinearPhysics
from deepinv.optim import Prior
from deepinv.utils.tensorlist import TensorList
from deepinv.distrib.utils import tiling_splitting_strategy, tiling_reduce_fn

Index = Tuple[Union[slice, int], ...]


# =========================
# Distributed Context
# =========================
class DistributedContext:
    """
    Small, opinionated context manager for distributed runs.

    Handles:
      - Init/destroy process group (if RANK/WORLD_SIZE envs exist)
      - Backend choice: NCCL when one-GPU-per-process per node, else Gloo
      - Device selection based on LOCAL_RANK and visible GPUs
      - Sharding helpers and tiny comm helpers
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        sharding: str = "round_robin",
        cleanup: bool = True,
        seed: Optional[int] = None,
        deterministic: bool = False,
    ):
        self.backend = backend
        self.sharding = sharding
        self.cleanup = cleanup
        self.seed = seed
        self.deterministic = deterministic

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
                # Backend decision is per-node:
                #   - If each node has at least as many *visible* GPUs as processes per node -> NCCL
                #   - Otherwise -> Gloo (e.g., GPU oversubscription or CPU)
                if cuda_ok and dist.is_nccl_available() and (self.local_world_size <= visible_gpus):
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
    def local_indices(self, num_items: int) -> List[int]:
        if self.sharding == "round_robin":
            return [i for i in range(num_items) if (i % self.world_size) == self.rank]
        elif self.sharding == "block":
            per_rank = (num_items + self.world_size - 1) // self.world_size
            start = self.rank * per_rank
            end = min(start + per_rank, num_items)
            return list(range(start, end))
        else:
            raise ValueError("sharding must be either 'round_robin' or 'block'.")

    # ----------------------
    # Collectives
    # ----------------------
    def all_reduce_(self, t: torch.Tensor, op: str = "sum"):
        """
        In-place all_reduce (returns `t`). Uses SUM for both cases; divides for mean.
        Works on both Gloo and NCCL for all dtypes that backend supports.
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
    """
    Holds only local measurement shards.

    You can supply either:
      - factory(i, device, shared) -> Tensor and num_items to build local shards
      - a full list of measurements (replicated across ranks); we'll select locals

    No collectives are used here; we assume each rank can construct its own locals.
    """

    def __init__(self,
                 ctx: DistributedContext,
                 num_items: Optional[int] = None,
                 *,
                 factory: Optional[Callable[[int, torch.device, Optional[Dict]], torch.Tensor]] = None,
                 measurements_list: Optional[Sequence[torch.Tensor]] = None,
                 shared: Optional[Dict] = None,
                 dtype: Optional[torch.dtype] = None):
        self.ctx = ctx
        self.dtype = dtype
        self.shared = shared

        if (factory is None and measurements_list is None) or (factory is not None and measurements_list is not None):
            raise ValueError("Provide either factory or measurements_list.")
        if num_items is None and factory is not None:
            raise ValueError("Must provide num_items if using factory.")
        
        self.num_items = num_items if num_items is not None else len(measurements_list)
        self.local_idx: List[int] = ctx.local_indices(num_items)
        self._global_to_local: Dict[int, int] = {g: j for j, g in enumerate(self.local_idx)}
        
        self.local: List[torch.Tensor] = []
        if factory is not None:
            for i in self.local_idx:
                y = factory(i, ctx.device, shared)
                if self.dtype is None:
                    self.dtype = y.dtype
                self.local.append(y.to(ctx.device, dtype=self.dtype))
        elif measurements_list is not None:
            for i in self.local_idx:
                self.local.append(measurements_list[i].to(ctx.device, dtype=self.dtype))
        else:
            raise ValueError("Provide factory or measurements_list.")

    def __len__(self):
        return self.num_items

    def indices(self) -> List[int]:
        return self.local_idx

    def get_local(self) -> List[torch.Tensor]:
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
    """
    A wrapper around a replicated signal tensor `x` with automatic synchronization.

    The signal is automatically synchronized across all processes after initialization
    and after any update operations. Users should not call sync_ manually.

    Usage:
        signal = DistributedSignal(ctx, shape)
        signal.update_(new_data)  # Automatically synchronized
        signal.add_(-lr * gradient)  # Automatically synchronized
    """
    def __init__(self,
                 ctx: DistributedContext,
                 shape: Sequence[int],
                 dtype: Optional[torch.dtype] = None,
                 init: Optional[Callable[[torch.device, torch.Size], torch.Tensor]] = None,
                 sync_src: int = 0):
        self.ctx = ctx
        self.dtype = dtype or torch.float32
        self._shape = torch.Size(shape)
        self._sync_src = sync_src
        
        if init is None:
            self._data = torch.zeros(self._shape, device=ctx.device, dtype=self.dtype)
        else:
            self._data = init(ctx.device, self._shape).to(ctx.device, dtype=self.dtype)

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
        new_signal = DistributedSignal(self.ctx, self.shape, self.dtype, sync_src=self._sync_src)
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
    """
    Holds only local physics operators. Exposes fast local and compatible global APIs.
    """

    def __init__(self,
                 ctx: DistributedContext,
                 num_ops: int,
                 factory: Callable[[int, torch.device, Optional[Dict]], Physics],
                 *,
                 shared: Optional[Dict] = None,
                 dtype: Optional[torch.dtype] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.ctx = ctx
        self.dtype = dtype or torch.float32
        self.num_ops = num_ops
        self.local_idx: List[int] = ctx.local_indices(num_ops)

        # Broadcast shared object (lightweight) once (root=0) if present
        self.shared = shared
        if ctx.is_dist and shared is not None:
            obj = [shared if ctx.rank == 0 else None]
            dist.broadcast_object_list(obj, src=0)
            self.shared = obj[0]

        # Build local physics
        self.local_physics: List[Physics] = []
        for i in self.local_idx:
            p_i = factory(i, ctx.device, self.shared)
            self.local_physics.append(p_i)

    # -------- local fast path --------
    def A_local(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        return [p.A(x, **kwargs) for p in self.local_physics]

    # Optional: global assembly (for compatibility/debug)
    def A(self, x: torch.Tensor, **kwargs) -> TensorList:
        pairs = list(zip(self.local_idx, self.A_local(x, **kwargs)))
        if not self.ctx.is_dist:
            out = [None] * self.num_ops
            for i, t in pairs:
                out[i] = t
            return TensorList(out)

        # NOTE: use all_gather_object only for small N or debugging.
        # For production, prefer tensor collectives with metadata. Kept simple here.
        gathered = [None] * self.ctx.world_size
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
    """
    Linear extension with local adjoint / vjp that reduce with a single all_reduce.
    """

    def __init__(self, ctx: DistributedContext, num_ops: int, factory, *,
                 shared: Optional[Dict] = None,
                 reduction: str = "sum",
                 dtype: Optional[torch.dtype] = None,
                 **kwargs):
        LinearPhysics.__init__(self, A=lambda x: x, A_adjoint=lambda y: y, **kwargs)
        self.reduction_mode = reduction
        super().__init__(ctx, num_ops, factory, shared=shared, dtype=dtype, **kwargs)

        for p in self.local_physics:
            if not isinstance(p, LinearPhysics):
                raise ValueError("factory must return LinearPhysics instances.")

    # ---- local (fast) ----
    def A_adjoint_local(self, y_local: List[torch.Tensor], **kwargs) -> torch.Tensor:
        if len(y_local) == 0:
            return 0.0
        contribs = [p.A_adjoint(y_i, **kwargs) for p, y_i in zip(self.local_physics, y_local)]
        return torch.stack(contribs, dim=0).sum(0)

    def A_vjp_local(self, x: torch.Tensor, v_local: List[torch.Tensor], **kwargs) -> torch.Tensor:
        if len(v_local) == 0:
            return 0.0
        contribs = [p.A_vjp(x, v_i, **kwargs) for p, v_i in zip(self.local_physics, v_local)]
        return torch.stack(contribs, dim=0).sum(0)

    # ---- global (compat) ----
    def _reduce_global(self, x_like: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x_like):
            # handle 0.0 placeholder for empty local set
            x_like = torch.zeros((), device=self.ctx.device, dtype=self.dtype)
            x_like = x_like.expand(())  # scalar
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
    """
    Truly distributed data fidelity:
      - builds/owns only local fidelity blocks (by index)
      - computes loss locally and all-reduces a single scalar
      - computes grad via local VJP and all-reduces a single x-shaped tensor
    """

    def __init__(self,
                 ctx: DistributedContext,
                 physics: Union[DistributedPhysics, DistributedLinearPhysics],
                 measurements: DistributedMeasurements,
                 *,
                 data_fidelity_factory: Optional[Callable[[int, torch.device, Optional[Dict]], object]] = None,
                 data_fidelity_list: Optional[Sequence[object]] = None,
                 shared: Optional[Dict] = None,
                 reduction: str = "sum"):
        self.ctx = ctx
        self.physics = physics
        self.meas = measurements
        self.reduction = reduction
        self.shared = shared

        local_idx = physics.local_idx
        if data_fidelity_factory is not None:
            self.local_df = [data_fidelity_factory(i, ctx.device, shared) for i in local_idx]
        elif data_fidelity_list is not None:
            self.local_df = [data_fidelity_list[i] for i in local_idx]
        else:
            raise ValueError("Provide data_fidelity_factory or data_fidelity_list.")

        # Sanity: indices alignment
        assert local_idx == self.meas.indices(), \
            "Physics and measurements must shard the same global indices under the same context."

    # ---- loss ----
    @torch.no_grad()
    def fn(self, X: DistributedSignal, **kwargs) -> torch.Tensor:
        # Local forward
        y_pred_local = self.physics.A_local(X.tensor, **kwargs)  # list aligned with local_idx
        y_true_local = self.meas.get_local()

        # Local accumulation (each DF returns a scalar per-batch; we sum)
        loss = X.tensor.new_zeros(())
        for df, yhat, y in zip(self.local_df, y_pred_local, y_true_local):
            # Use the distance function directly since we already have A(x) = yhat
            loss = loss + df.d(yhat, y, **kwargs)

        # Global reduction
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
        for df, yhat, y in zip(self.local_df, y_pred_local, y_true_local):
            v_local.append(df.d.grad(yhat, y, **kwargs))

        # Local VJP into x-shape
        if isinstance(self.physics, DistributedLinearPhysics):
            x_grad_local = self.physics.A_vjp_local(X.tensor, v_local, **kwargs)
        else:
            # Fallback: generic physics with manual vjp (requires per-op support).
            contribs = [p.A_vjp(X.tensor, v_i, **kwargs) for p, v_i in zip(self.physics.local_physics, v_local)]
            x_grad_local = torch.stack(contribs, dim=0).sum(0) if len(contribs) else 0.0

        if not torch.is_tensor(x_grad_local):
            x_grad_local = torch.zeros_like(X.tensor)

        # Global reduction
        self.ctx.all_reduce_(x_grad_local, op="sum")
        if self.reduction == "mean":
            x_grad_local = x_grad_local / float(self.physics.num_ops)

        return x_grad_local


class DistributedPrior:
    """
    Distributed prior where:
      • Splits are computed at __init__ from the *global* signal shape.
      • Each rank processes only its local split indices.
      • grad/prox: compute local results, gather to rank 0, call user reduce_fn(List[Tensor])->Tensor,
        broadcast the final tensor to all ranks.

    Parameters
    ----------
    ctx : DistributedContext
    prior : deepinv.optim.Prior
    splitting_strategy : Union[str, Callable[[torch.Size], List[Index]]]
        If callable, it must take the full signal shape and return an *ordered* list of slice-indexers.
        If 'tiling2d', pass tile options via splitting_kwargs.
    signal_shape : Sequence[int]
        Full tensor shape of the signal to be processed (e.g. BCHW).
    reduce_fn : Callable[[List[torch.Tensor]], torch.Tensor]
        User-provided reduction function applied on rank 0 that receives the global ordered list of
        processed pieces and returns a single tensor (same shape as signal).
    splitting_kwargs : dict
        Extra args to the splitting strategy (e.g., tile size/stride).
    """

    def __init__(
        self,
        ctx: DistributedContext,
        prior: Prior,
        *,
        splitting_strategy: Union[str, Callable[[torch.Size], Tuple[List[Index], Dict]]],
        signal_shape: Sequence[int],
        reduce_fn: Callable[[List[torch.Tensor], Dict], torch.Tensor],
        splitting_kwargs: Optional[dict] = None,
    ):
        self.ctx = ctx
        self.prior = prior

        if hasattr(prior, "to"):
            self.prior.to(ctx.device)

        self.signal_shape = torch.Size(signal_shape)
        self.reduce_fn = reduce_fn
        self.splitting_kwargs = splitting_kwargs or {}

        # --- compute global ordered list of slices once (independent of rank) ---
        if callable(splitting_strategy):
            self._global_slices, self._global_metadata = splitting_strategy(self.signal_shape, **self.splitting_kwargs)
        elif splitting_strategy == "tiling2d":
            self._global_slices, self._global_metadata = tiling_splitting_strategy(
                self.signal_shape, **self.splitting_kwargs
            )
            self.reduce_fn = tiling_reduce_fn
        else:
            raise ValueError(f"Unknown splitting_strategy: {splitting_strategy}")

        self.num_splits = len(self._global_slices)
        if self.num_splits == 0:
            raise ValueError("Splitting strategy produced zero splits.")

        # determine local split indices for this rank
        self.local_split_indices: List[int] = list(self.ctx.local_indices(self.num_splits))

        # If distributed, ensure all ranks share the same slices ordering (broadcast from rank 0)
        if self.ctx.is_dist:
            obj = [self._global_slices if self.ctx.rank == 0 else None]
            dist.broadcast_object_list(obj, src=0)
            self._global_slices = obj[0]

    # ---- public ops -------------------------------------------------------

    def grad(self, X: DistributedSignal, **kwargs) -> torch.Tensor:
        return self._apply_op(X, op="grad", **kwargs)

    def prox(self, X: DistributedSignal, **kwargs) -> torch.Tensor:
        return self._apply_op(X, op="prox", **kwargs)

    # ---- internals --------------------------------------------------------

    def _apply_op(self, X: DistributedSignal, *, op: str, **kwargs) -> torch.Tensor:
        """
        op in {"grad", "prox"} – calls prior.op on each local split, gathers, reduces on rank 0, broadcasts.
        """
        # 1) local compute
        local_pairs = self._compute_local_pairs(X, op=op, **kwargs)

        # 2) gather all (idx, tensor) pairs
        if self.ctx.is_dist:
            gathered = [None] * self.ctx.world_size
            dist.all_gather_object(gathered, local_pairs)
            all_pairs = []
            for part in gathered:
                all_pairs.extend(part)
        else:
            all_pairs = local_pairs

        # 3) assemble into the global ordered list (by index)
        pieces: List[Optional[torch.Tensor]] = [None] * self.num_splits
        for idx, tens in all_pairs:
            pieces[idx] = tens
        # sanity check
        if any(p is None for p in pieces):
            missing = [i for i, p in enumerate(pieces) if p is None]
            raise RuntimeError(f"Missing pieces for splits (did all ranks produce their parts?): {missing}")

        # 4) reduce on rank 0, broadcast tensor result
        if self.ctx.is_dist:
            if self.ctx.rank == 0:
                out = self.reduce_fn(pieces, self._global_metadata)
                out.to(self.ctx.device, dtype=X.tensor.dtype)
            else:
                out = torch.empty(self.signal_shape, device=self.ctx.device, dtype=X.tensor.dtype)

            # broadcast to everyone
            self.ctx.broadcast_(out, src=0)
            return out
        else:
            out = self.reduce_fn(pieces, self._global_metadata).to(self.ctx.device, dtype=X.tensor.dtype)
            return out

    def _compute_local_pairs(self, X: DistributedSignal, *, op: str, **kwargs) -> List[Tuple[int, torch.Tensor]]:
        """
        Returns a list of (global_index, processed_tensor) for this rank's local splits.
        """
        fn: Callable[[torch.Tensor], torch.Tensor]
        if op == "grad":
            if hasattr(self.prior, "grad"):
                fn = lambda t: self.prior.grad(t, **kwargs)
            else:
                raise ValueError("Prior does not implement .grad()")
        elif op == "prox":
            if hasattr(self.prior, "prox"):
                fn = lambda t: self.prior.prox(t, **kwargs)
            else:
                raise ValueError("Prior does not implement .prox()")
        else:
            raise ValueError(op)

        pairs: List[Tuple[int, torch.Tensor]] = []
        for idx in self.local_split_indices:
            slc = self._global_slices[idx]
            # NOTE: use a copy for safety (some priors may write in-place)
            piece = X.tensor[slc].clone()
            res = fn(piece)
            # Keep result on current device; reduce_fn will run on rank 0 and convert as needed
            pairs.append((idx, res))
        return pairs
