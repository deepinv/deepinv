import os
from typing import Callable, Optional, List, Dict, Tuple, Union, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

from deepinv.physics import Physics, LinearPhysics
from deepinv.optim import Prior
from deepinv.utils.tensorlist import TensorList

from deepinv.distrib.utils_new import tiling_splitting_strategy, tiling2d_reduce_fn

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
        device_mode: Optional[str] = None,
    ):
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
        if self.device_mode == "cpu":
            # Force CPU mode
            self.device = torch.device("cpu")
        elif self.device_mode == "gpu":
            # Force GPU mode (require CUDA)
            if not torch.cuda.is_available() or visible_gpus == 0:
                raise RuntimeError("GPU mode requested but CUDA not available or no visible GPUs")
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
    def local_indices(self, num_items: int) -> List[int]:
        if self.sharding == "round_robin":
            indices = [i for i in range(num_items) if (i % self.world_size) == self.rank]
        elif self.sharding == "block":
            per_rank = (num_items + self.world_size - 1) // self.world_size
            start = self.rank * per_rank
            end = min(start + per_rank, num_items)
            indices = list(range(start, end))
        else:
            raise ValueError("sharding must be either 'round_robin' or 'block'.")
        
        # Warning for efficiency, but allow empty indices (don't raise error)
        if self.is_dist and len(indices) == 0 and self.rank == 0:
            print(f"Warning: Some ranks have no work items to process "
                  f"(num_items={num_items}, world_size={self.world_size}). "
                  f"Consider reducing world_size or increasing the workload for better efficiency.")
        
        return indices

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
                 init: Optional[torch.Tensor] = None,
                 sync_src: int = 0):
        self.ctx = ctx
        self.dtype = dtype or torch.float32
        self._shape = torch.Size(shape)
        self._sync_src = sync_src
        
        if init is None:
            self._data = torch.zeros(self._shape, device=ctx.device, dtype=self.dtype)
        else:
            self._data = init.to(ctx.device, dtype=self.dtype)

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
            # Return zeros with proper shape for empty local set
            return torch.zeros((), device=self.ctx.device, dtype=self.dtype)
        contribs = [p.A_adjoint(y_i, **kwargs) for p, y_i in zip(self.local_physics, y_local)]
        return torch.stack(contribs, dim=0).sum(0)

    def A_vjp_local(self, x: torch.Tensor, v_local: List[torch.Tensor], **kwargs) -> torch.Tensor:
        if len(v_local) == 0:
            # Return zeros with proper shape for empty local set
            return torch.zeros_like(x)
        contribs = [p.A_vjp(x, v_i, **kwargs) for p, v_i in zip(self.local_physics, v_local)]
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
        for df, yhat, y in zip(self.local_df, y_pred_local, y_true_local):
            v_local.append(df.d.grad(yhat, y, **kwargs))

        # Local VJP into x-shape
        if isinstance(self.physics, DistributedLinearPhysics):
            x_grad_local = self.physics.A_vjp_local(X.tensor, v_local, **kwargs)
        else:
            # Fallback: generic physics with manual vjp (requires per-op support).
            if len(v_local) > 0:
                contribs = [p.A_vjp(X.tensor, v_i, **kwargs) for p, v_i in zip(self.physics.local_physics, v_local)]
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
    """
    Distributed prior where:
      • Splits are computed at __init__ from the *global* signal shape.
      • Each rank processes only its local split indices.
      • grad/prox: compute local results, apply custom reduction function locally, then all-reduce.

    Parameters
    ----------
    ctx : DistributedContext
    prior : deepinv.optim.Prior
    splitting_strategy : Union[str, Callable[[torch.Size], List[Index]]]
        If callable, it must take the full signal shape and return an *ordered* list of slice-indexers.
        If 'tiling2d', pass tile options via splitting_kwargs.
    signal_shape : Sequence[int]
        Full tensor shape of the signal to be processed (e.g. BCHW).
    reduce_fn : Callable[[torch.Tensor, List[Tuple[int, torch.Tensor]], Dict], None]
        User-provided reduction function that takes (out_local, local_pairs, global_metadata) and
        fills out_local tensor in-place. Default is tiling2d_reduce_fn for tiling strategies.
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
        reduce_fn: Optional[Callable[[torch.Tensor, List[Tuple[int, torch.Tensor]], Dict], None]] = None,
        splitting_kwargs: Optional[dict] = None,
        batching: bool = True,
    ):
        self.ctx = ctx
        self.prior = prior

        if hasattr(prior, "to"):
            self.prior.to(ctx.device)

        self.signal_shape = torch.Size(signal_shape)
        self.splitting_kwargs = splitting_kwargs or {}
        self.batching = batching  # Keep the user-specified batching preference

        # --- compute global ordered list of slices once (independent of rank) ---
        if callable(splitting_strategy):
            self._global_slices, self._global_metadata = splitting_strategy(self.signal_shape, **self.splitting_kwargs)
        elif splitting_strategy == "tiling2d":
            self._global_slices, self._global_metadata = tiling_splitting_strategy(
                self.signal_shape, **self.splitting_kwargs
            )
        else:
            raise ValueError(f"Unknown splitting_strategy: {splitting_strategy}")

        # Set default reduce_fn if not provided
        if reduce_fn is None:
            self.reduce_fn = tiling2d_reduce_fn
        else:
            self.reduce_fn = reduce_fn

        self.num_splits = len(self._global_slices)
        if self.num_splits == 0:
            raise ValueError("Splitting strategy produced zero splits.")

        # determine local split indices for this rank
        self.local_split_indices: List[int] = list(self.ctx.local_indices(self.num_splits))

        # Check for insufficient work distribution
        if self.ctx.is_dist and self.ctx.rank == 0:
            ranks_with_work = sum(1 for rank in range(self.ctx.world_size) 
                                 if len(self.ctx.local_indices(self.num_splits)) > 0)
            if ranks_with_work < self.ctx.world_size:
                print(f"Warning: Only {ranks_with_work}/{self.ctx.world_size} ranks have patches to process. "
                      f"Some ranks will have no work. "
                      f"Current: {self.num_splits} patches for {self.ctx.world_size} ranks.")

    # ---- public ops -------------------------------------------------------

    def grad(self, X: DistributedSignal, **kwargs) -> torch.Tensor:
        return self._apply_op(X, op="grad", **kwargs)

    def prox(self, X: DistributedSignal, **kwargs) -> torch.Tensor:
        return self._apply_op(X, op="prox", **kwargs)

    # ---- internals --------------------------------------------------------

    def _apply_op(self, X: DistributedSignal, *, op: str, **kwargs) -> torch.Tensor:
        """
        op in {"grad", "prox"} – calls prior.op on each local split, applies custom reduction locally, 
        then all-reduces the result tensor.
        """
        # 1) Compute local processed patches
        local_pairs = self._compute_local_pairs(X, op=op, **kwargs)

        # 2) Initialize output tensor with zeros
        out_local = torch.zeros(self.signal_shape, device=self.ctx.device, dtype=X.tensor.dtype)
        
        # 3) Apply user-defined reduction function locally
        self.reduce_fn(out_local, local_pairs, self._global_metadata)
        
        # 4) All-reduce to combine results from all ranks
        if self.ctx.is_dist:
            self.ctx.all_reduce_(out_local, op="sum")
            
        return out_local

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

        if self.batching:
            pad_specs = self._global_metadata["pad_specs"]           # list indexed by GLOBAL tile index
            pad_mode = self._global_metadata.get("pad_mode", "reflect")
            win_h, win_w = self._global_metadata["window_shape"]

            pieces = []
            for idx in self.local_split_indices:                      # idx is GLOBAL tile index
                sl = self._global_slices[idx]                         # slice into original tensor
                piece = X.tensor[sl]                                  # [B,C,h_var,w_var]
                piece = F.pad(piece, pad=pad_specs[idx], mode=pad_mode)
                # sanity: all windows must now be uniform
                if piece.shape[-2] != win_h or piece.shape[-1] != win_w:
                    raise RuntimeError(
                        f"Uniform window check failed for tile {idx}: "
                        f"got {piece.shape[-2:]} vs expected {(win_h, win_w)}"
                    )
                pieces.append(piece)

            # Handle empty case (rank has no work)
            if len(pieces) == 0:
                return pairs  # Return empty list
            
            batch = torch.cat(pieces, dim=0)
            results = fn(batch)                                       # same leading batch dim
            if results.shape[0] != len(self.local_split_indices):
                raise RuntimeError(
                    f"Prior {op} returned wrong batch size: "
                    f"{results.shape[0]} != {len(self.local_split_indices)}"
                )
            pairs = list(zip(self.local_split_indices, [results[i][None, :] for i in range(results.shape[0])]))
            return pairs
        
        # Non-batching case: apply proper padding workflow for consistency
        pad_specs = self._global_metadata["pad_specs"]
        pad_mode = self._global_metadata.get("pad_mode", "reflect")
        
        for idx in self.local_split_indices:
            slc = self._global_slices[idx]
            # Extract piece and apply padding (same as batching case)
            piece = X.tensor[slc].clone()
            piece = F.pad(piece, pad=pad_specs[idx], mode=pad_mode)
            
            # Apply prior to padded piece
            res = fn(piece)
            
            # Keep result on current device; reduce_fn will run on rank 0 and convert as needed
            pairs.append((idx, res))
        return pairs
