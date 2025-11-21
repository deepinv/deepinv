from __future__ import annotations

import os
from typing import Callable, Optional, Union, Sequence

import torch
import torch.distributed as dist

from deepinv.physics import Physics, LinearPhysics
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
        indices = [i for i in range(num_items) if (i % self.world_size) == self.rank]

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
        self,
        local_indices: list[int],
        local_results: list[torch.Tensor],
        num_operators: int,
    ) -> TensorList:
        """
        Naive gather strategy using object serialization.

        Best for: Small tensors where serialization overhead is negligible.

        Communication pattern: 1 all_gather_object call (high overhead, simple)

        :param list[int] local_indices: indices owned by this rank
        :param list[torch.Tensor] local_results: local tensor results
        :param int num_operators: total number of operators
        :return: TensorList with all results
        """
        if not self.is_dist:
            # Single process: just build the list
            out: list = [None] * num_operators
            for idx, result in zip(local_indices, local_results, strict=False):
                out[idx] = result
            return TensorList(out)

        # Pair indices with tensors
        pairs = list(zip(local_indices, local_results, strict=False))

        # Gather all pairs from all ranks
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, pairs)

        # Assemble into output list
        out: list = [None] * num_operators
        for rank_pairs in gathered:
            if rank_pairs is not None:
                for idx, tensor in rank_pairs:
                    out[idx] = tensor

        return TensorList(out)

    def gather_tensorlist_concatenated(
        self,
        local_indices: list[int],
        local_results: list[torch.Tensor],
        num_operators: int,
    ) -> TensorList:
        """
        Efficient gather strategy using a single concatenated tensor.

        Best for: Medium to large tensors where minimizing communication calls matters.

        Communication pattern:
        - 1 all_gather_object for metadata (lightweight)
        - 1 all_gather for concatenated tensor data (efficient)

        :param list[int] local_indices: indices owned by this rank
        :param list[torch.Tensor] local_results: local tensor results
        :param int num_operators: total number of operators
        :return: TensorList with all results
        """
        if not self.is_dist:
            # Single process: just build the list
            out: list = [None] * num_operators
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
                        max_numel - flat.numel(), dtype=flat.dtype, device=self.device
                    )
                    flat = torch.cat([flat, padding])
                flattened_padded.append(flat)

            # Concatenate all local flattened tensors
            local_concat = torch.stack(
                flattened_padded
            )  # Shape: [num_local, max_numel]
        else:
            # This rank has no tensors - create empty placeholder
            max_numel = max((numel for _, _, _, numel in all_metadata), default=1)
            local_concat = torch.zeros(
                (0, max_numel), dtype=torch.float32, device=self.device
            )

        # Step 3: All-gather the concatenated tensor
        # First, share how many tensors each rank has
        num_local_tensors = torch.tensor(
            [len(local_results)], dtype=torch.int64, device=self.device
        )
        all_num_tensors = [
            torch.zeros(1, dtype=torch.int64, device=self.device)
            for _ in range(self.world_size)
        ]
        dist.all_gather(all_num_tensors, num_local_tensors)

        # Prepare buffers for all_gather - need to use all_gather_into_tensor for variable sizes
        # Create a single flattened buffer for all ranks' data
        max_local_count = max(count.item() for count in all_num_tensors)

        # Pad local_concat to max_local_count if needed
        if len(local_results) < max_local_count:
            padding_rows = max_local_count - len(local_results)
            padding = torch.zeros(
                (padding_rows, max_numel), dtype=local_concat.dtype, device=self.device
            )
            local_concat_padded = torch.cat([local_concat, padding], dim=0)
        else:
            local_concat_padded = local_concat

        # All-gather with fixed size
        tensor_lists = [
            torch.zeros(
                (max_local_count, max_numel),
                dtype=local_concat_padded.dtype,
                device=self.device,
            )
            for _ in range(self.world_size)
        ]

        # All-gather the actual data
        dist.all_gather(tensor_lists, local_concat_padded)

        # Step 4: Reconstruct TensorList from gathered data
        out: list = [None] * num_operators
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
        self,
        local_indices: list[int],
        local_results: list[torch.Tensor],
        num_operators: int,
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

        Communication pattern: num_operators broadcasts (can be overlapped with computation)

        :param list[int] local_indices: indices owned by this rank
        :param list[torch.Tensor] local_results: local tensor results
        :param int num_operators: total number of operators
        :return: TensorList with all results
        """
        if not self.is_dist:
            # Single process: just build the list
            out: list = [None] * num_operators
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
        out: list = [None] * num_operators

        for idx in range(num_operators):
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
    :param int num_operators: total number of physics operators.
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
        num_operators: int,
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
        :param int num_operators: total number of physics operators.
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
        self.num_operators = num_operators
        self.local_indexes: list[int] = ctx.local_indices(num_operators)

        # Validate and set gather strategy
        valid_strategies = {"naive", "concatenated", "broadcast"}
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
        for i in self.local_indexes:
            p_i = factory(i, ctx.device, self.shared)
            self.local_physics.append(p_i)

    # -------- Factorized map-reduce logic --------
    def _map_reduce(self, x: torch.Tensor, local_op: Callable, **kwargs) -> TensorList:
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

        if not self.ctx.is_dist:
            # Single process: just build the list
            out: list = [None] * self.num_operators
            for idx, result in zip(self.local_indexes, local_results, strict=False):
                out[idx] = result
            return TensorList(out)

        # Step 2: Reduce - gather results using selected strategy
        if self.gather_strategy == "naive":
            return self.ctx.gather_tensorlist_naive(
                self.local_indexes, local_results, self.num_operators
            )
        elif self.gather_strategy == "concatenated":
            return self.ctx.gather_tensorlist_concatenated(
                self.local_indexes, local_results, self.num_operators
            )
        elif self.gather_strategy == "broadcast":
            return self.ctx.gather_tensorlist_broadcast(
                self.local_indexes, local_results, self.num_operators
            )
        else:
            raise ValueError(f"Unknown gather strategy: {self.gather_strategy}")

    def A_local(self, x: torch.Tensor, **kwargs) -> list[torch.Tensor]:
        r"""
        Apply forward operator to local physics operators only (no communication).

        :param torch.Tensor x: input signal.
        :param dict kwargs: optional parameters for the forward operator.
        :return: (list[torch.Tensor]) list of measurements from local operators.
        """
        return [p.A(x, **kwargs) for p in self.local_physics]

    def A(self, x: torch.Tensor, **kwargs) -> TensorList:
        r"""
        Apply forward operator to all distributed physics operators with automatic gathering.

        Applies the forward operator :math:`A(x)` by computing local measurements and gathering
        results from all ranks using the configured gather strategy.

        :param torch.Tensor x: input signal.
        :param dict kwargs: optional parameters for the forward operator.
        :return: (TensorList) complete list of measurements from all operators.
        """
        return self._map_reduce(x, lambda p, x, **kw: p.A(x, **kw), **kwargs)

    def forward(self, x, **kwargs):
        r"""
        Apply full forward model with sensor and noise models.

        Applies the complete forward model (sensor + noise + physics) to the input signal,
        gathering results from all distributed operators.

        :param torch.Tensor x: input signal.
        :param dict kwargs: optional parameters for the forward model.
        :return: (TensorList) complete list of noisy measurements from all operators.
        """
        return self._map_reduce(x, lambda p, x, **kw: p.forward(x, **kw), **kwargs)


class DistributedLinearPhysics(DistributedPhysics, LinearPhysics):
    r"""
    Distributed linear physics operators with efficient adjoint and reduction operations.

    This class extends DistributedPhysics for linear operators, providing two types of methods:

    **Local methods** (``*_local``): Compute operations only on local operators owned by this rank.
    These methods are efficient as they perform no inter-rank communication. They return partial
    results that must be manually reduced across ranks if needed.

    **Global methods** (``A_adjoint``, ``A_vjp``, etc.): Complete distributed operations that
    automatically handle communication and reductions. These methods call their corresponding
    local method and then perform a single all-reduce to combine results from all ranks.

    The local/global pattern enables:
    - Efficient single-reduction operations (vs. multiple reductions)
    - Flexibility to defer reductions when composing multiple operations
    - Clear separation between computation and communication

    :param DistributedContext ctx: distributed context manager.
    :param int num_operators: total number of physics operators to distribute.
    :param Callable factory: factory function that creates linear physics operators.
        Should have signature ``factory(index: int, device: torch.device, shared: Optional[dict]) -> LinearPhysics``.
    :param None, dict shared: shared data dictionary passed to factory function for all operators.
    :param str reduction: reduction mode for distributed operations. Options are ``'sum'`` (stack operators)
        or ``'mean'`` (average operators). Default is ``'sum'``.
    :param None, torch.dtype dtype: data type for operations.
    :param str gather_strategy: strategy for gathering distributed results in forward operations.
        Options are ``'naive'``, ``'concatenated'``, or ``'broadcast'``. Default is ``'concatenated'``.
    """

    def __init__(
        self,
        ctx: DistributedContext,
        num_operators: int,
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
        :param int num_operators: total number of physics operators.
        :param Callable factory: factory function that creates linear physics operators.
        :param None, dict shared: shared data dictionary passed to factory function.
        :param str reduction: reduction mode for distributed operations. Options are `'sum'` and `'mean'`.
        :param None, torch.dtype dtype: data type for operations.
        :param str gather_strategy: strategy for gathering distributed results.
        """
        LinearPhysics.__init__(self, A=lambda x: x, A_adjoint=lambda y: y, **kwargs)
        self.reduction_mode = reduction
        super().__init__(
            ctx,
            num_operators,
            factory,
            shared=shared,
            dtype=dtype,
            gather_strategy=gather_strategy,
            **kwargs,
        )

        for p in self.local_physics:
            if not isinstance(p, LinearPhysics):
                raise ValueError("factory must return LinearPhysics instances.")

    # ---- local (fast) ----
    def A_adjoint_local(self, y_local: list[torch.Tensor], **kwargs) -> torch.Tensor:
        r"""
        Compute local adjoint operation without inter-rank communication.

        Applies the adjoint operator to measurements owned by this rank only.
        The result is a partial contribution that must be summed across all ranks
        to obtain the complete adjoint operation.

        :param list[torch.Tensor] y_local: list of local measurements, one per local operator.
        :param dict kwargs: optional parameters for the adjoint operator.
        :return: (torch.Tensor) partial adjoint result (sum of local contributions).
        """
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
        r"""
        Compute local vector-Jacobian product without inter-rank communication.

        Computes the VJP for local operators only. The result is a partial contribution
        that must be summed across all ranks for the complete VJP.

        :param torch.Tensor x: input tensor.
        :param list[torch.Tensor] v_local: list of local cotangent vectors, one per local operator.
        :param dict kwargs: optional parameters for the VJP operation.
        :return: (torch.Tensor) partial VJP result (sum of local contributions).
        """
        if len(v_local) == 0:
            # Return zeros with proper shape for empty local set
            return torch.zeros_like(x)
        contribs = [
            p.A_vjp(x, v_i, **kwargs)
            for p, v_i in zip(self.local_physics, v_local, strict=False)
        ]
        return torch.stack(contribs, dim=0).sum(0)

    def A_adjoint_A_local(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Compute local :math:`A^T A` operation without inter-rank communication.

        Computes the normal operator :math:`A^T A` for local operators only.
        For stacked operators, this computes :math:`\sum_{i \in \text{local}} A_i^T A_i x`.
        The result must be summed across ranks to obtain :math:`\sum_i A_i^T A_i x`.

        :param torch.Tensor x: input tensor.
        :param dict kwargs: optional parameters for the operation.
        :return: (torch.Tensor) partial :math:`A^T A` result (sum of local contributions).
        """
        if len(self.local_physics) == 0:
            # Return zeros with proper shape for empty local set
            return torch.zeros_like(x)
        contribs = [p.A_adjoint_A(x, **kwargs) for p in self.local_physics]
        return torch.stack(contribs, dim=0).sum(0)

    def A_A_adjoint_local(self, y_local: list[torch.Tensor], **kwargs) -> torch.Tensor:
        r"""
        Compute local :math:`A A^T` operation without inter-rank communication.

        Computes the Gram operator :math:`A A^T` for local operators only.
        For stacked operators, this computes :math:`\sum_{i \in \text{local}} A_i A_i^T y_i`.
        The result must be summed across ranks to obtain the complete Gram operator.

        :param list[torch.Tensor] y_local: list of local measurement tensors, one per local operator.
        :param dict kwargs: optional parameters for the operation.
        :return: (torch.Tensor) partial :math:`A A^T` result (sum of local contributions).
        """
        if len(y_local) == 0:
            # Return zeros with proper shape for empty local set
            return torch.zeros((), device=self.ctx.device, dtype=self.dtype)
        contribs = [
            p.A_A_adjoint(y_i, **kwargs)
            for p, y_i in zip(self.local_physics, y_local, strict=False)
        ]
        return torch.stack(contribs, dim=0).sum(0)

    def A_dagger_local(self, y_local: list[torch.Tensor], **kwargs) -> torch.Tensor:
        r"""
        Compute local pseudoinverse operation without inter-rank communication.

        Computes the pseudoinverse (least squares solution) for local operators only.
        For stacked operators, computes :math:`\sum_{i \in \text{local}} A_i^\dagger y_i`.
        This is an approximation of the true pseudoinverse of the stacked operator.

        Note: This provides an approximation. For the exact pseudoinverse, use the global
        :meth:`A_dagger` method with ``local_only=False``.

        :param list[torch.Tensor] y_local: list of local measurement tensors, one per local operator.
        :param dict kwargs: optional parameters for the pseudoinverse operation.
        :return: (torch.Tensor) partial pseudoinverse result (sum of local contributions).
        """
        if len(y_local) == 0:
            # Return zeros with proper shape for empty local set
            return torch.zeros((), device=self.ctx.device, dtype=self.dtype)
        contribs = [
            p.A_dagger(y_i, **kwargs)
            for p, y_i in zip(self.local_physics, y_local, strict=False)
        ]
        return torch.stack(contribs, dim=0).sum(0)

    # ---- global (compat) ----
    def _reduce_global(self, x_like: torch.Tensor, reduction="sum") -> torch.Tensor:
        r"""
        Perform all-reduce operation to combine local results across all ranks.

        This is an internal helper that converts local (per-rank) results into global
        results by summing contributions from all ranks. Optionally normalizes by the
        number of operators if mean reduction is requested.

        :param torch.Tensor x_like: local tensor to reduce.
        :param str reduction: reduction mode, either ``'sum'`` or ``'mean'``.
        :return: (torch.Tensor) globally reduced tensor.
        """
        if not torch.is_tensor(x_like):
            # handle 0.0 placeholder for empty local set
            x_like = torch.zeros((), device=self.ctx.device, dtype=self.dtype)
            x_like = x_like.expand(())  # scalar

        # Ensure all ranks participate in collective even with empty data
        if self.ctx.is_dist:
            # For ranks with empty local sets, x_like should be zeros
            self.ctx.all_reduce_(x_like, op="sum")

        if self.reduction_mode == "mean" or reduction == "mean":
            x_like = x_like / float(self.num_operators)
        return x_like

    def A_adjoint(self, y: TensorList, **kwargs) -> torch.Tensor:
        r"""
        Compute global adjoint operation with automatic reduction.

        Extracts local measurements, computes local adjoint contributions, and reduces
        across all ranks to obtain the complete :math:`A^T y` where :math:`A` is the
        stacked operator :math:`A = [A_1; A_2; \ldots; A_n]`.

        :param TensorList y: full list of measurements from all operators.
        :param dict kwargs: optional parameters for the adjoint operation.
        :return: (torch.Tensor) complete adjoint result :math:`A^T y`.
        """
        y_local = [y[i] for i in self.local_indexes]
        local = self.A_adjoint_local(y_local, **kwargs)
        return self._reduce_global(local)

    def A_vjp(self, x: torch.Tensor, v: TensorList, **kwargs) -> torch.Tensor:
        r"""
        Compute global vector-Jacobian product with automatic reduction.

        Extracts local cotangent vectors, computes local VJP contributions, and reduces
        across all ranks to obtain the complete VJP.

        :param torch.Tensor x: input tensor.
        :param TensorList v: full list of cotangent vectors from all operators.
        :param dict kwargs: optional parameters for the VJP operation.
        :return: (torch.Tensor) complete VJP result.
        """
        v_local = [v[i] for i in self.local_indexes]
        local = self.A_vjp_local(x, v_local, **kwargs)
        return self._reduce_global(local)

    def A_adjoint_A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Compute global :math:`A^T A` operation with automatic reduction.

        Computes the complete normal operator :math:`A^T A x = \sum_i A_i^T A_i x` by
        combining local contributions from all ranks.

        :param torch.Tensor x: input tensor.
        :param dict kwargs: optional parameters for the operation.
        :return: (torch.Tensor) complete :math:`A^T A x` result.
        """
        local = self.A_adjoint_A_local(x, **kwargs)
        return self._reduce_global(local)

    def A_A_adjoint(self, y: TensorList, **kwargs) -> torch.Tensor:
        r"""
        Compute global :math:`A A^T` operation with automatic reduction.

        Computes the complete Gram operator by combining local contributions from all ranks.
        For stacked operators, this computes :math:`\sum_i A_i A_i^T y_i`.

        :param TensorList y: full list of measurements from all operators.
        :param dict kwargs: optional parameters for the operation.
        :return: (torch.Tensor) complete :math:`A A^T` result.
        """
        y_local = [y[i] for i in self.local_indexes]
        local = self.A_A_adjoint_local(y_local, **kwargs)
        return self._reduce_global(local)

    def A_dagger(
        self,
        y: TensorList,
        solver: str = "CG",
        max_iter: int | None = None,
        tol: float | None = None,
        verbose: bool = False,
        local_only: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Computes the pseudoinverse (least squares solution) of the distributed operator.

        This method provides two strategies:

        1. **Local approximation** (``local_only=True``, default): Each rank computes the pseudoinverse
           of its local operators independently, then sums the results with a single reduction.
           This is efficient (minimal communication) and provides an approximation.
           For stacked operators, :math:`A^\dagger y \approx \sum_i A_i^\dagger y_i`.

        2. **Global computation** (``local_only=False``): Uses the full least squares solver
           with distributed :meth:`A_adjoint_A` and :meth:`A_A_adjoint` operations.
           This computes the exact pseudoinverse but requires communication at every iteration.

        :param TensorList y: measurements to invert.
        :param str solver: least squares solver to use (only for ``local_only=False``).
            Choose between ``'CG'``, ``'lsqr'``, ``'BiCGStab'`` and ``'minres'``.
        :param None, int max_iter: maximum number of iterations for least squares solver.
        :param None, float tol: relative tolerance for least squares solver.
        :param bool verbose: print information (only on rank 0).
        :param bool local_only: If ``True`` (default), compute local daggers and sum-reduce (efficient).
            If ``False``, compute exact global pseudoinverse with full communication (expensive).
        :param dict kwargs: optional parameters for the forward operator.

        :return: (torch.Tensor) pseudoinverse solution. If ``local_only=True``, returns approximation.
            If ``local_only=False``, returns exact least squares solution.
        """
        if local_only:
            # Efficient local computation with single sum reduction
            y_local = [y[i] for i in self.local_indexes]
            local = self.A_dagger_local(y_local, **kwargs)
            return self._reduce_global(local, reduction="mean")
        else:
            # Global computation: call parent class A_dagger which uses least squares
            # This will use our distributed A, A_adjoint, A_adjoint_A and A_A_adjoint
            verbose_flag = verbose and self.ctx.rank == 0
            return super().A_dagger(
                y,
                solver=solver,
                max_iter=max_iter,
                tol=tol,
                verbose=verbose_flag,
                **kwargs,
            )

    def compute_sqnorm(
        self,
        x0: torch.Tensor,
        *,
        max_iter: int = 50,
        tol: float = 1e-3,
        verbose: bool = True,
        local_only: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Computes the squared spectral :math:`\ell_2` norm of the distributed operator.

        This method provides two strategies:

        1. **Local approximation** (``local_only=True``, default): Each rank computes the norm
           of its local operators independently, then a single max-reduction provides an upper bound.
           This is efficient (minimal communication) and valid for conservative estimates.
           For stacked operators :math:`A = [A_1; A_2; \ldots; A_n]`, we have
           :math:`\|A\|^2 \leq \sum_i \|A_i\|^2`, and we use :math:`\max_i \|A_i\|^2` as
           a conservative upper bound.

        2. **Global computation** (``local_only=False``): Uses the full distributed :meth:`A_adjoint_A`
           with communication at every power iteration. This computes the exact norm but is
           communication-intensive.

        :param torch.Tensor x0: an unbatched tensor sharing its shape, dtype and device with the initial iterate
        :param int max_iter: maximum number of iterations for power method
        :param float tol: relative variation criterion for convergence
        :param bool verbose: print information (only on rank 0)
        :param bool local_only: If ``True`` (default), compute local norms and max-reduce (efficient).
            If ``False``, compute exact global norm with full communication (expensive).
        :param dict kwargs: optional parameters for the forward operator

        :return: (torch.Tensor) squared spectral norm. If ``local_only=True``, returns upper bound.
            If ``local_only=False``, returns exact value.
        """

        if local_only:
            # Efficient local computation with single max reduction
            if len(self.local_physics) == 0:
                # Empty local set - contribute zero to max
                local_sqnorm = torch.tensor(
                    0.0, device=self.ctx.device, dtype=self.dtype
                )
            else:
                # Compute max norm across local operators
                local_norms = []
                for p in self.local_physics:
                    norm_p = p.compute_sqnorm(
                        x0, max_iter=max_iter, tol=tol, verbose=False, **kwargs
                    )
                    local_norms.append(norm_p)
                local_sqnorm = torch.stack(local_norms).sum()

            # Single sum reduction across all ranks
            if self.ctx.is_dist:
                self.ctx.all_reduce_(local_sqnorm, op="sum")

            if verbose and self.ctx.rank == 0:
                print(
                    f"Computed local norm upper bound: ||A||_2^2 ≤ {local_sqnorm.item():.2f}"
                )

            return local_sqnorm

        else:
            # Global computation: call parent class compute_sqnorm which uses power method
            # This will use our distributed A_adjoint_A
            verbose_flag = verbose and self.ctx.rank == 0
            return super().compute_sqnorm(
                x0, max_iter=max_iter, tol=tol, verbose=verbose_flag, **kwargs
            )


class DistributedProcessing:
    r"""
    Distributed signal processing using pluggable tiling and reduction strategies.

    This class enables distributed processing of large signals (images, volumes, etc.) by:
    1. Splitting the signal into patches using a chosen strategy
    2. Distributing patches across multiple processes/GPUs
    3. Processing each patch independently using a provided processor function
    4. Combining processed patches back into the full signal with proper overlap handling

    The processor can be any callable that operates on tensors (e.g., denoisers, priors,
    neural networks, etc.). The class handles all distributed coordination automatically.

    **Example use cases:**
    - Distributed denoising of large images/volumes
    - Applying neural network priors across multiple GPUs
    - Processing signals too large to fit on a single device

    :param DistributedContext ctx: distributed context manager.
    :param Callable[[torch.Tensor], torch.Tensor] processor: processing function to apply to signal patches.
        Should accept a batched tensor of shape ``(N, C, ...)`` and return a tensor of the same shape.
        Examples: denoiser, neural network, prior gradient function, etc.
    :param Union[str, DistributedSignalStrategy] strategy: signal processing strategy for patch extraction
        and reduction. Either a strategy name (``'basic'``, ``'smart_tiling'``) or a custom strategy instance.
        Default is ``'smart_tiling'`` which handles overlapping patches with smooth blending.
    :param None, dict strategy_kwargs: additional keyword arguments passed to the strategy constructor
        when using string strategy names. Examples: ``patch_size``, ``overlap``, ``blend_mode``.
    :param None, int max_batch_size: maximum number of patches to process in a single batch.
        If ``None``, all local patches are batched together. Set to ``1`` for sequential processing
        (useful for memory-constrained scenarios). Higher values increase throughput but require more memory.
    """

    def __init__(
        self,
        ctx: DistributedContext,
        processor: Callable[[torch.Tensor], torch.Tensor],
        *,
        strategy: Optional[Union[str, DistributedSignalStrategy]] = None,
        strategy_kwargs: Optional[dict] = None,
        max_batch_size: Optional[int] = None,
        **kwargs,
    ):
        r"""
        Initialize distributed signal processor.

        :param DistributedContext ctx: distributed context manager.
        :param Callable[[torch.Tensor], torch.Tensor] processor: processing function that takes a batched
            tensor ``(N, C, ...)`` and returns a processed tensor of the same shape. Examples include
            denoisers, neural networks, prior gradient functions, etc.
        :param Union[str, DistributedSignalStrategy] strategy: signal processing strategy. Either a strategy
            name (``'basic'``, ``'smart_tiling'``) or a custom :class:`DistributedSignalStrategy` instance.
            Default is ``'smart_tiling'``.
        :param None, dict strategy_kwargs: additional keyword arguments for the strategy constructor when
            using string strategy names (e.g., ``patch_size``, ``overlap``, ``blend_mode``).
        :param None, int max_batch_size: maximum number of patches to process in a single batch. If ``None``,
            all local patches are batched together for maximum throughput. Set to ``1`` for sequential
            processing (lowest memory usage). Intermediate values balance throughput and memory.
        """
        self.ctx = ctx
        self.processor = processor
        self.max_batch_size = max_batch_size
        self.strategy = strategy if strategy is not None else "smart_tiling"
        self.strategy_kwargs = strategy_kwargs

        if hasattr(processor, "to"):
            self.processor.to(ctx.device)

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Apply distributed processing to input signal.

        :param torch.Tensor x: input signal tensor to process, typically of shape ``(B, C, H, W)`` for 2D
            or ``(B, C, D, H, W)`` for 3D signals.
        :param args: additional positional arguments passed to the processor.
        :param kwargs: additional keyword arguments passed to the processor.
        :return: (torch.Tensor) processed signal with the same shape as input.
        """
        self._init_shape_and_strategy(x.shape)
        return self._apply_op(x, *args, **kwargs)

    # ---- internals --------------------------------------------------------

    def _init_shape_and_strategy(
        self,
        signal_shape: Sequence[int],
    ):
        r"""
        Initialize or update the signal shape and processing strategy.

        This method is called automatically on the first forward pass to set up the
        tiling strategy based on the input signal dimensions. It creates the strategy,
        determines the number of patches, and assigns patches to ranks.

        :param Sequence[int] signal_shape: shape of the input signal tensor (e.g., ``(B, C, H, W)``).
        """

        self.signal_shape = torch.Size(signal_shape)

        # Create or set the strategy
        if isinstance(self.strategy, str):
            from .distribution_strategies.strategies import create_strategy

            strategy_kwargs = self.strategy_kwargs or {}
            # Assume standard layout (B, C, D1, D2, ...) -> n_dimension = len - 2
            n_dimension = len(signal_shape) - 2
            self._strategy = create_strategy(
                self.strategy, signal_shape, n_dimension=n_dimension, **strategy_kwargs
            )
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
        r"""
        Apply processor using the distributed strategy (internal method).

        This method orchestrates the complete distributed processing pipeline:
        1. Extracts local patches from the input signal using the strategy
        2. Applies batching as defined by the strategy and max_batch_size
        3. Applies the processor function to each batch of patches
        4. Unpacks batched results back to individual patches
        5. Reduces patches back to the output signal using the strategy's blending
        6. All-reduces the final result across ranks to combine overlapping regions

        :param torch.Tensor x: input signal to process.
        :param args: additional positional arguments passed to the processor.
        :param kwargs: additional keyword arguments passed to the processor.
        :return: (torch.Tensor) processed signal with the same shape as input.
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
        batched_patches = self._strategy.apply_batching(
            patches, max_batch_size=self.max_batch_size
        )

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
