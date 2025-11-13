from __future__ import annotations

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
    Context manager for distributed runs.

    Handles:
      - Init/destroy process group (if RANK/WORLD_SIZE envs exist)
      - Backend choice: NCCL when one-GPU-per-process per node, else Gloo
      - Device selection based on LOCAL_RANK and visible GPUs
      - Sharding helpers and tiny comm helpers

    :param str backend: backend to use for distributed communication. If `None`, automatically selects NCCL for GPU or Gloo for CPU.
    :param bool cleanup: whether to clean up the process group on exit.
    :param None, int seed: random seed for reproducible results. If provided, each rank gets seed + rank.
    :param bool deterministic: whether to use deterministic cuDNN operations.
    :param None, str device_mode: device selection mode. Options are `'cpu'`, `'gpu'`, or `None` for automatic.
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        cleanup: bool = True,
        seed: Optional[int] = None,
        deterministic: bool = False,
        device_mode: Optional[str] = None,
    ):
        r"""
        Initialize the distributed context manager.

        :param str backend: backend to use for distributed communication. If `None`, automatically selects NCCL for GPU or Gloo for CPU.
        :param bool cleanup: whether to clean up the process group on exit.
        :param None, int seed: random seed for reproducible results. If provided, each rank gets seed + rank.
        :param bool deterministic: whether to use deterministic cuDNN operations.
        :param None, str device_mode: device selection mode. Options are `'cpu'`, `'gpu'`, or `None` for automatic.
        """
        self.backend = backend
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
            # Auto mode
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
        Get local indices for this rank based on round robin sharding.

        :param int num_items: total number of items to shard.
        :return: (list) list of indices assigned to this rank.
        """
        indices = [
            i for i in range(num_items) if (i % self.world_size) == self.rank
        ]

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

    # ----------------------
    # Gather Strategies
    # ----------------------
    def gather_tensorlist_naive(
        self, local_indices: list[int], local_results: list[torch.Tensor], num_ops: int
    ) -> TensorList:
        """
        Naive gather strategy using object serialization.
        
        Best for: Small tensors where serialization overhead is negligible.
        
        Communication pattern: 1 all_gather_object call (high overhead, simple)
        
        :param list[int] local_indices: indices owned by this rank
        :param list[torch.Tensor] local_results: local tensor results
        :param int num_ops: total number of operators
        :return: TensorList with all results
        """
        if not self.is_dist:
            # Single process: just build the list
            out: list = [None] * num_ops
            for idx, result in zip(local_indices, local_results, strict=False):
                out[idx] = result
            return TensorList(out)
        
        # Pair indices with tensors
        pairs = list(zip(local_indices, local_results, strict=False))
        
        # Gather all pairs from all ranks
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, pairs)
        
        # Assemble into output list
        out: list = [None] * num_ops
        for rank_pairs in gathered:
            if rank_pairs is not None:
                for idx, tensor in rank_pairs:
                    out[idx] = tensor
        
        return TensorList(out)
    
    def gather_tensorlist_concatenated(
        self, local_indices: list[int], local_results: list[torch.Tensor], num_ops: int
    ) -> TensorList:
        """
        Efficient gather strategy using a single concatenated tensor.
        
        Best for: Medium to large tensors where minimizing communication calls matters.
        
        Communication pattern:
        - 1 all_gather_object for metadata (lightweight)
        - 1 all_gather for concatenated tensor data (efficient)
        
        :param list[int] local_indices: indices owned by this rank
        :param list[torch.Tensor] local_results: local tensor results
        :param int num_ops: total number of operators
        :return: TensorList with all results
        """
        if not self.is_dist:
            # Single process: just build the list
            out: list = [None] * num_ops
            for idx, result in zip(local_indices, local_results, strict=False):
                out[idx] = result
            return TensorList(out)
        
        # Step 1: Share metadata (indices, shapes, dtypes)
        local_metadata = [
            (idx, tuple(result.shape), result.dtype, result.numel())
            for idx, result in zip(local_indices, local_results, strict=False)
        ]
        
        gathered_metadata = [None] * self.world_size
        dist.all_gather_object(gathered_metadata, local_metadata)
        
        # Flatten all metadata
        all_metadata = []
        for rank_metadata in gathered_metadata:
            if rank_metadata is not None:
                all_metadata.extend(rank_metadata)
        
        # Step 2: Flatten and pad local tensors to max size
        if len(local_results) > 0:
            # Find max numel across all tensors globally
            max_numel = max(numel for _, _, _, numel in all_metadata)
            
            # Flatten and pad each local tensor
            flattened_padded = []
            for result in local_results:
                flat = result.flatten()
                if flat.numel() < max_numel:
                    padding = torch.zeros(
                        max_numel - flat.numel(), 
                        dtype=flat.dtype, 
                        device=self.device
                    )
                    flat = torch.cat([flat, padding])
                flattened_padded.append(flat)
            
            # Concatenate all local flattened tensors
            local_concat = torch.stack(flattened_padded)  # Shape: [num_local, max_numel]
        else:
            # This rank has no tensors - create empty placeholder
            max_numel = max((numel for _, _, _, numel in all_metadata), default=1)
            local_concat = torch.zeros((0, max_numel), dtype=torch.float32, device=self.device)
        
        # Step 3: All-gather the concatenated tensor
        # First, share how many tensors each rank has
        num_local_tensors = torch.tensor([len(local_results)], dtype=torch.int64, device=self.device)
        all_num_tensors = [torch.zeros(1, dtype=torch.int64, device=self.device) for _ in range(self.world_size)]
        dist.all_gather(all_num_tensors, num_local_tensors)
        
        # Prepare buffers for all_gather - need to use all_gather_into_tensor for variable sizes
        # Create a single flattened buffer for all ranks' data
        max_local_count = max(count.item() for count in all_num_tensors)
        
        # Pad local_concat to max_local_count if needed
        if len(local_results) < max_local_count:
            padding_rows = max_local_count - len(local_results)
            padding = torch.zeros((padding_rows, max_numel), dtype=local_concat.dtype, device=self.device)
            local_concat_padded = torch.cat([local_concat, padding], dim=0)
        else:
            local_concat_padded = local_concat
        
        # All-gather with fixed size
        tensor_lists = [
            torch.zeros((max_local_count, max_numel), dtype=local_concat_padded.dtype, device=self.device)
            for _ in range(self.world_size)
        ]
        
        # All-gather the actual data
        dist.all_gather(tensor_lists, local_concat_padded)
        
        # Step 4: Reconstruct TensorList from gathered data
        out: list = [None] * num_ops
        metadata_idx = 0
        
        for rank_idx, rank_metadata in enumerate(gathered_metadata):
            if rank_metadata is None:
                continue
                
            rank_tensors = tensor_lists[rank_idx]
            
            for local_pos, (idx, shape, dtype, numel) in enumerate(rank_metadata):
                # Extract the flattened tensor
                flat_tensor = rank_tensors[local_pos, :numel]
                
                # Reshape to original shape
                out[idx] = flat_tensor.reshape(shape).to(dtype)
        
        return TensorList(out)
    
    def gather_tensorlist_broadcast(
        self, local_indices: list[int], local_results: list[torch.Tensor], num_ops: int
    ) -> TensorList:
        """
        Broadcast-based gather strategy.
        
        Best for: Heterogeneous tensor sizes where different operators produce 
        very different sized outputs, or when you want to overlap computation 
        with communication (each operator can be broadcast as soon as it's ready).
        
        Use cases:
        - Different physics operators produce vastly different measurement sizes
        - Streaming/pipelined execution where operators complete at different times
        - Very large tensors where memory for concatenation is prohibitive
        
        Communication pattern: num_ops broadcasts (can be overlapped with computation)
        
        :param list[int] local_indices: indices owned by this rank
        :param list[torch.Tensor] local_results: local tensor results
        :param int num_ops: total number of operators
        :return: TensorList with all results
        """
        if not self.is_dist:
            # Single process: just build the list
            out: list = [None] * num_ops
            for idx, result in zip(local_indices, local_results, strict=False):
                out[idx] = result
            return TensorList(out)
        
        # Step 1: Share metadata (indices, shapes, dtypes)
        local_metadata = [
            (idx, tuple(result.shape), result.dtype)
            for idx, result in zip(local_indices, local_results, strict=False)
        ]
        
        gathered_metadata = [None] * self.world_size
        dist.all_gather_object(gathered_metadata, local_metadata)
        
        # Build shape map
        shape_map = {}
        for rank_metadata in gathered_metadata:
            if rank_metadata is not None:
                for idx, shape, dtype in rank_metadata:
                    if idx not in shape_map:
                        shape_map[idx] = (shape, dtype)
        
        # Step 2: Broadcast each operator's result from its owner
        out: list = [None] * num_ops
        
        for idx in range(num_ops):
            shape, dtype = shape_map[idx]
            responsible_rank = idx % self.world_size  # Round-robin sharding
            
            # Prepare tensor to send/receive
            if idx in local_indices:
                # This rank owns this operator
                local_pos = local_indices.index(idx)
                tensor_to_send = local_results[local_pos].contiguous()
            else:
                # Create receive buffer
                tensor_to_send = torch.zeros(shape, dtype=dtype, device=self.device)
            
            # Broadcast from owner to all ranks
            dist.broadcast(tensor_to_send, src=responsible_rank)
            out[idx] = tensor_to_send
        
        return TensorList(out)


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
    :param str gather_strategy: strategy for gathering distributed results. Options are:
        - `'naive'`: Simple object serialization (best for small tensors)
        - `'concatenated'`: Single concatenated tensor (best for medium/large tensors, minimal communication)
        - `'broadcast'`: Per-operator broadcasts (best for heterogeneous sizes or streaming)
        Default is `'concatenated'`.
    """

    def __init__(
        self,
        ctx: DistributedContext,
        num_ops: int,
        factory: Callable[[int, torch.device, Optional[dict]], Physics],
        *,
        shared: Optional[dict] = None,
        dtype: Optional[torch.dtype] = None,
        gather_strategy: str = "concatenated",
        **kwargs,
    ):
        r"""
        Initialize distributed physics operators.

        :param DistributedContext ctx: distributed context manager.
        :param int num_ops: total number of physics operators.
        :param Callable factory: factory function that creates physics operators. Should have signature `factory(index, device, shared) -> Physics`.
        :param None, dict shared: shared data dictionary passed to factory function.
        :param None, torch.dtype dtype: data type for operations.
        :param str gather_strategy: strategy for gathering distributed results. Options are:
            - `'naive'`: Simple object serialization (best for small tensors)
            - `'concatenated'`: Single concatenated tensor (best for medium/large tensors, minimal communication)
            - `'broadcast'`: Per-operator broadcasts (best for heterogeneous sizes or streaming)
            Default is `'concatenated'`.
        """
        super().__init__(**kwargs)
        self.ctx = ctx
        self.dtype = dtype or torch.float32
        self.num_ops = num_ops
        self.local_idx: list[int] = ctx.local_indices(num_ops)
        
        # Validate and set gather strategy
        valid_strategies = {'naive', 'concatenated', 'broadcast'}
        if gather_strategy not in valid_strategies:
            raise ValueError(
                f"gather_strategy must be one of {valid_strategies}, got '{gather_strategy}'"
            )
        self.gather_strategy = gather_strategy

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

    # -------- Factorized map-reduce logic --------
    def _map_reduce(
        self, x: torch.Tensor, local_op: Callable, **kwargs
    ) -> TensorList:
        """
        Efficient map-reduce pattern for distributed operations.
        
        Maps local_op over local physics operators, then gathers results using
        the configured gather strategy.
        
        :param torch.Tensor x: input tensor
        :param Callable local_op: operation to apply, e.g., lambda p, x: p.A(x)
        :return: TensorList of gathered results
        """
        # Step 1: Map - apply operation to local physics operators
        local_results = [local_op(p, x, **kwargs) for p in self.local_physics]
        
        # Step 2: Reduce - gather results using selected strategy
        if self.gather_strategy == 'naive':
            return self.ctx.gather_tensorlist_naive(
                self.local_idx, local_results, self.num_ops
            )
        elif self.gather_strategy == 'concatenated':
            return self.ctx.gather_tensorlist_concatenated(
                self.local_idx, local_results, self.num_ops
            )
        elif self.gather_strategy == 'broadcast':
            return self.ctx.gather_tensorlist_broadcast(
                self.local_idx, local_results, self.num_ops
            )
        else:
            raise ValueError(f"Unknown gather strategy: {self.gather_strategy}")

    def A_local(self, x: torch.Tensor, **kwargs) -> list[torch.Tensor]:
        return [p.A(x, **kwargs) for p in self.local_physics]

    def A(self, x: torch.Tensor, **kwargs) -> TensorList:
        return self._map_reduce(x, lambda p, x, **kw: p.A(x, **kw), **kwargs)
    
    def forward(self, x, **kwargs):
        """Apply full forward model: sensor(noise(A(x)))"""
        return self._map_reduce(x, lambda p, x, **kw: p.forward(x, **kw), **kwargs)


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
        gather_strategy: str = "concatenated",
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
        :param str gather_strategy: strategy for gathering distributed results.
        """
        LinearPhysics.__init__(self, A=lambda x: x, A_adjoint=lambda y: y, **kwargs)
        self.reduction_mode = reduction
        super().__init__(ctx, num_ops, factory, shared=shared, dtype=dtype, 
                        gather_strategy=gather_strategy, **kwargs)

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



class DistributedProcessing:
    r"""
    Distributed processing using pluggable signal processing strategies (denoiser, prior, etc)

    This class enables distributed processing by using signal processing
    strategies that define how to split, process, and combine signal patches across
    multiple processes.

    :param DistributedContext ctx: distributed context manager.
    :param Callable processor: processing function to be applied in a distributed manner.
    :param Union[str, DistributedSignalStrategy] strategy: signal processing strategy. Either a strategy name (`'basic'`, `'smart_tiling'`) or a custom strategy instance.
    """

    def __init__(
        self,
        ctx: DistributedContext,
        processor: Callable,
        *,
        strategy: Optional[Union[str, DistributedSignalStrategy]] = None,
        strategy_kwargs: Optional[dict] = None,
        max_batch_size: Optional[int] = None,
        **kwargs,
    ):
        r"""
        Initialize distributed prior.

        :param DistributedContext ctx: distributed context manager.
        :param deepinv.optim.Prior prior: prior term to be applied in a distributed manner.
        :param Union[str, DistributedSignalStrategy] strategy: signal processing strategy. Either a strategy name (`'basic'`, `'smart_tiling'`) or a custom strategy instance.
        :param Sequence[int] signal_shape: full tensor shape of the signal to be processed (e.g. BCHW).
        :param None, dict strategy_kwargs: extra arguments for the strategy (when using string strategy names).
        :param None, int max_batch_size: maximum number of patches to process in a single batch. If `None`, all patches are batched together. Set to 1 for sequential processing.
        """
        self.ctx = ctx
        self.processor = processor
        self.max_batch_size = max_batch_size
        self.strategy = strategy if strategy is not None else "smart_tiling"
        self.strategy_kwargs = strategy_kwargs

        if hasattr(processor, "to"):
            self.processor.to(ctx.device)

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._init_shape_and_strategy(x.shape)
        return self._apply_op(x, *args, **kwargs)

    # ---- internals --------------------------------------------------------

    def _init_shape_and_strategy(
        self,
        signal_shape: Sequence[int],
    ):

        self.signal_shape = torch.Size(signal_shape)

        # Create or set the strategy
        if isinstance(self.strategy, str):
            from .distribution_strategies.strategies import create_strategy

            strategy_kwargs = self.strategy_kwargs or {}
            self._strategy = create_strategy(self.strategy, signal_shape, **strategy_kwargs)
        else:
            self._strategy = self.strategy
        if self._strategy is None:
            raise RuntimeError("Strategy is None - failed to create or import strategy")

        self.num_patches = self._strategy.get_num_patches()
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

    def _apply_op(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Apply processor using the distributed strategy.

        This method:
        1. Extracts local patches using the strategy
        2. Applies batching as defined by the strategy
        3. Applies the processor to batched patches
        4. Unpacks and reduces results using the strategy
        5. All-reduces the final result across ranks
        """
        # Handle empty case early
        if not self.local_indices:
            out_local = torch.zeros(
                self.signal_shape, device=self.ctx.device, dtype=x.dtype
            )
            if self.ctx.is_dist:
                self.ctx.all_reduce_(out_local, op="sum")
            return out_local

        # 1. Extract local patches using strategy
        local_pairs = self._strategy.get_local_patches(x, self.local_indices)
        patches = [patch for _, patch in local_pairs]

        # 2. Apply batching strategy with max_batch_size
        batched_patches = self._strategy.apply_batching(patches, max_batch_size=self.max_batch_size)

        # 3. Apply processor to each batch
        processed_batches = []
        for batch in batched_patches:
            result = self.processor(batch, *args, **kwargs)
            processed_batches.append(result)

        # 4. Unpack results back to individual patches
        processed_patches = self._strategy.unpack_batched_results(
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
            self.signal_shape, device=self.ctx.device, dtype=x.dtype
        )
        self._strategy.reduce_patches(out_local, processed_pairs)

        # 7. All-reduce to combine results from all ranks
        if self.ctx.is_dist:
            self.ctx.all_reduce_(out_local, op="sum")

        return out_local
