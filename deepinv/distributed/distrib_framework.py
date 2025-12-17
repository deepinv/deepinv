from __future__ import annotations

import os
import copy
import platform
from typing import Callable, Sequence

import torch
import torch.distributed as dist

from deepinv.physics import Physics, LinearPhysics
from deepinv.optim.data_fidelity import DataFidelity
from deepinv.utils.tensorlist import TensorList

from deepinv.distributed.strategies import DistributedSignalStrategy

Index = tuple[slice | int, ...]


# =========================
# Distributed Context
# =========================
class DistributedContext:
    r"""
    Context manager for distributed computing.

    Handles:
      - Initialization/destruction of the process group (if `RANK` / `WORLD_SIZE` environment variables exist)
      - Backend choice: NCCL when one-GPU-per-process per node, else Gloo.
      - Device selection based on `LOCAL_RANK` and visible GPUs
      - Sharding helpers and tiny communication helpers

    :param str backend: backend to use for distributed communication. If `None` (default), automatically selects NCCL for GPU or Gloo for CPU.
    :param bool cleanup: whether to clean up the process group on exit. Default is `True`.
    :param int seed: random seed for reproducible results. If provided, each rank gets `seed + rank`. Default is `None`.
    :param bool deterministic: whether to use deterministic cuDNN operations. Default is `False`.
    :param str device_mode: device selection mode. Options are `'cpu'`, `'gpu'`, or `None` for automatic. Default is `None`.

    """

    def __init__(
        self,
        backend: str | None = None,
        cleanup: bool = True,
        seed: int | None = None,
        deterministic: bool = False,
        device_mode: str | None = None,
    ):
        r"""
        Initialize the distributed context manager.
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

            # Windows can hit Gloo initialization failures like:
            # This typically happens when Gloo selects a transport unsupported by
            # the Windows build (or the env var is set to an invalid/empty value).
            if backend == "gloo":
                self._configure_gloo_environment()
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
            if torch.cuda.is_available() and visible_gpus > 0:
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
        # 2. We initialized it (initialized_here=True)
        # 3. It's still initialized
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

    def _configure_gloo_environment(self) -> None:
        """
        Stable Gloo initialization for Windows
        """
        # PyTorch's Gloo transport options are typically "tcp" or "uv".
        # Some Windows wheels do not support all transports; forcing TCP avoids
        # failures like `makeDeviceForHostname(): unsupported gloo device`.
        if platform.system() == "Windows":
            os.environ.setdefault("GLOO_DEVICE_TRANSPORT", "tcp")

            # Ensure loopback for local multi-process tests when users pass
            # `localhost` (some setups resolve it unexpectedly).
            if os.environ.get("MASTER_ADDR", "").lower() == "localhost":
                os.environ["MASTER_ADDR"] = "127.0.0.1"

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
    def all_reduce_(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        r"""
        In-place reduction across all processes (returns `tensor`).
        Supports `'sum'` and `'mean'` operations.


        :param torch.Tensor tensor: tensor input and output of the reduction. The function operates in-place.
        :param str op: reduction operation (`'sum'` or `'mean'`). Default is `'sum'`.
        :return: the reduced tensor.
        """
        if self.is_dist:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            if op.lower() == "mean":
                tensor /= float(self.world_size)
        return tensor

    def broadcast_(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        r"""
        Broadcast tensor from source rank to all other ranks (in-place).

        :param torch.Tensor tensor: tensor to broadcast (modified in-place).
        :param int src: source rank to broadcast from. Default is `0`.
        """

        if self.is_dist:
            dist.broadcast(tensor, src=src)
        return tensor

    def barrier(self):
        r"""
        Synchronize all processes.
        """
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
        r"""
        Naive gather strategy using object serialization.

        Best for small tensors where serialization overhead is negligible.
        This function calls :func:`torch.distributed.all_gather_object` (high overhead, simple) under the hood.

        .. note::

            This strategy only supports the Gloo backend.
            For NCCL (GPU) environments, use `gather_tensorlist_concatenated` or `gather_tensorlist_broadcast` instead.

        :param list[int] local_indices: indices owned by this rank
        :param list[torch.Tensor] local_results: local tensor results
        :param int num_operators: total number of operators
        :return: TensorList with all results
        :raises RuntimeError: if using NCCL backend (GPU multi-process mode)
        """
        if not self.is_dist:
            # Single process: just build the list
            out: list = [None] * num_operators
            for idx, result in zip(local_indices, local_results):
                out[idx] = result
            return TensorList(out)

        # Check if we're using NCCL backend (not compatible with all_gather_object)
        if dist.get_backend() == "nccl":
            raise RuntimeError(
                "The 'naive' gather strategy uses all_gather_object which is not supported "
                "by the NCCL backend (GPU multi-process mode). Please use 'concatenated' or "
                "'broadcast' gather strategies instead."
            )

        # Pair indices with tensors
        pairs = list(zip(local_indices, local_results))

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
            - 1 call to :func:`torch.distributed.all_gather_object` for metadata (lightweight)
            - 1 call to :func:`torch.distributed.all_gather` for concatenated tensor data (efficient)

        :param list[int] local_indices: indices owned by this rank
        :param list[torch.Tensor] local_results: local tensor results
        :param int num_operators: total number of operators
        :return: TensorList with all results
        """
        if not self.is_dist:
            # Single process: just build the list
            out: list = [None] * num_operators
            for idx, result in zip(local_indices, local_results):
                out[idx] = result
            return TensorList(out)

        # Step 1: Share metadata (indices, shapes, dtypes)
        local_metadata = [
            (idx, tuple(result.shape), result.dtype, result.numel())
            for idx, result in zip(local_indices, local_results)
        ]

        gathered_metadata = [None] * self.world_size
        dist.all_gather_object(gathered_metadata, local_metadata)

        # Flatten all metadata
        all_metadata = []
        for rank_metadata in gathered_metadata:
            if rank_metadata is not None:
                all_metadata.extend(rank_metadata)

        # Determine canonical dtype for all_gather (use first tensor's dtype for consistency)
        if all_metadata:
            canonical_dtype = all_metadata[0][2]  # dtype from first tensor
        else:
            canonical_dtype = torch.float32

        # Step 2: Flatten and pad local tensors to max size
        if len(local_results) > 0:
            # Find max numel across all tensors globally
            max_numel = max(numel for _, _, _, numel in all_metadata)

            # Flatten and pad each local tensor
            flattened_padded = []
            for result in local_results:
                # Convert to canonical dtype for consistency across ranks
                flat = result.to(canonical_dtype).flatten()
                if flat.numel() < max_numel:
                    padding = torch.zeros(
                        max_numel - flat.numel(),
                        dtype=canonical_dtype,
                        device=self.device,
                    )
                    flat = torch.cat([flat, padding])
                flattened_padded.append(flat)

            # Concatenate all local flattened tensors
            local_concat = torch.stack(
                flattened_padded
            )  # Shape: [num_local, max_numel]
        else:
            # This rank has no tensors - create empty placeholder
            if all_metadata:
                max_numel = max(numel for _, _, _, numel in all_metadata)
            else:
                max_numel = 1

            local_concat = torch.zeros(
                (0, max_numel), dtype=canonical_dtype, device=self.device
            )

        # Step 3: All-gather the concatenated tensor
        # Calculate max_local_count from gathered_metadata (no need for extra communication)
        max_local_count = 0
        for rank_metadata in gathered_metadata:
            if rank_metadata is not None:
                max_local_count = max(max_local_count, len(rank_metadata))

        # Prepare buffers for all_gather - need to use all_gather_into_tensor for variable sizes
        # Create a single flattened buffer for all ranks' data

        # Pad local_concat to max_local_count if needed
        # Use shape[0] instead of len(local_results) to handle empty tensor case correctly
        if local_concat.shape[0] < max_local_count:
            padding_rows = max_local_count - local_concat.shape[0]
            # Handle edge case: if local_concat is completely empty (0, max_numel),
            # directly create the padded tensor instead of concatenating
            if local_concat.shape[0] == 0:
                local_concat_padded = torch.zeros(
                    (max_local_count, max_numel),
                    dtype=canonical_dtype,
                    device=self.device,
                )
            else:
                padding = torch.zeros(
                    (padding_rows, max_numel),
                    dtype=local_concat.dtype,
                    device=self.device,
                )
                local_concat_padded = torch.cat([local_concat, padding], dim=0)
        else:
            local_concat_padded = local_concat

        # Ensure contiguous memory layout (for Gloo backend)
        local_concat_padded = local_concat_padded.contiguous()

        # All-gather with fixed size - use canonical dtype for consistency
        tensor_lists = [
            torch.zeros(
                (max_local_count, max_numel),
                dtype=canonical_dtype,
                device=self.device,
            )
            for _ in range(self.world_size)
        ]

        # All-gather the actual data
        dist.all_gather(tensor_lists, local_concat_padded)

        # Step 4: Reconstruct TensorList from gathered data
        out: list = [None] * num_operators

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
            for idx, result in zip(local_indices, local_results):
                out[idx] = result
            return TensorList(out)

        # Step 1: Share metadata (indices, shapes, dtypes)
        local_metadata = [
            (idx, tuple(result.shape), result.dtype)
            for idx, result in zip(local_indices, local_results)
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

    This class distributes a *collection* of physics operators across multiple processes,
    where each process owns a subset of the operators.

    .. note::

        It is intended to parallelize models naturally expressed as a stack/list of operators
        (e.g., :class:`deepinv.physics.StackedPhysics` or an explicit Python list of
        :class:`deepinv.physics.Physics` objects) and is **not** meant to split a
        single monolithic physics operator across ranks.

    If your forward model is a single operator that can be decomposed into multiple
    sub-operators, it is up to you to perform that decomposition (e.g., build a
    :class:`deepinv.physics.StackedPhysics`) and then pass that collection to
    :class:`DistributedPhysics` via the ``factory``.

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
        factory: Callable[[int, torch.device, dict | None], Physics],
        *,
        shared: dict | None = None,
        dtype: torch.dtype | None = None,
        gather_strategy: str = "concatenated",
        **kwargs,
    ):
        r"""
        Initialize distributed physics operators.
        """
        super().__init__(**kwargs)
        self.ctx = ctx
        self.dtype = dtype or torch.float32
        self.num_operators = num_operators
        self.local_indexes: list[int] = ctx.local_indices(num_operators)

        # Validate and set gather strategy
        valid_strategies = ("naive", "concatenated", "broadcast")
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
    def _map_reduce(
        self,
        x: torch.Tensor | list[torch.Tensor],
        local_op: Callable,
        reduce: bool = True,
        sum_results: bool = False,
        **kwargs,
    ) -> TensorList | list[torch.Tensor] | torch.Tensor:
        """
        Map-reduce pattern for distributed operations.

        Maps local_op over local physics operators, then gathers results using
        the configured gather strategy.

        :param torch.Tensor | list[torch.Tensor] x: input tensor(s). If a list,
            should match the number of local operators (per-operator inputs).
        :param Callable local_op: operation to apply, e.g., lambda p, x: p.A(x)
        :param bool reduce: whether to gather results across ranks.
        :param bool sum_results: if True, sum all gathered results into a single tensor
        :return: TensorList of gathered results, list of local results, or summed tensor
        """
        # Handle per-operator inputs (e.g., for A_adjoint where each operator gets different y_i)
        if isinstance(x, (list, tuple)):
            if len(x) != len(self.local_physics):
                raise ValueError(
                    f"When passing list/tuple to _map_reduce, length must match local operators: "
                    f"got {len(x)}, expected {len(self.local_physics)}"
                )
            local_results = [
                local_op(p, x_i, **kwargs) for p, x_i in zip(self.local_physics, x)
            ]
        else:
            # Single input shared by all operators (e.g., for A(x))
            local_results = [local_op(p, x, **kwargs) for p in self.local_physics]

        if not reduce:
            if sum_results and len(local_results) > 0:
                return torch.stack(local_results, dim=0).sum(0)
            return local_results

        if not self.ctx.is_dist:
            # Single process: just build the list
            out: list = [None] * self.num_operators
            for idx, result in zip(self.local_indexes, local_results):
                out[idx] = result

            if sum_results:
                return torch.stack(
                    [out[i] for i in range(self.num_operators)], dim=0
                ).sum(0)
            return TensorList(out)

        # Step 2: Reduce - gather results using selected strategy
        if self.gather_strategy == "naive":
            gathered = self.ctx.gather_tensorlist_naive(
                self.local_indexes, local_results, self.num_operators
            )
        elif self.gather_strategy == "concatenated":
            gathered = self.ctx.gather_tensorlist_concatenated(
                self.local_indexes, local_results, self.num_operators
            )
        elif self.gather_strategy == "broadcast":
            gathered = self.ctx.gather_tensorlist_broadcast(
                self.local_indexes, local_results, self.num_operators
            )
        else:
            raise ValueError(f"Unknown gather strategy: {self.gather_strategy}")

        if sum_results:
            return torch.stack(
                [gathered[i] for i in range(self.num_operators)], dim=0
            ).sum(0)
        return gathered

    def A(
        self, x: torch.Tensor, reduce: bool = True, **kwargs
    ) -> TensorList | list[torch.Tensor]:
        r"""
        Apply forward operator to all distributed physics operators with automatic gathering.

        Applies the forward operator :math:`A(x)` by computing local measurements and gathering
        results from all ranks using the configured gather strategy.

        :param torch.Tensor x: input signal.
        :param bool reduce: whether to gather results across ranks. If `False`, returns local measurements.
        :param dict kwargs: optional parameters for the forward operator.
        :return: complete list of measurements from all operators (or local list if `reduce=False`).
        """
        return self._map_reduce(
            x, lambda p, x, **kw: p.A(x, **kw), reduce=reduce, **kwargs
        )

    def forward(self, x, reduce: bool = True, **kwargs):
        r"""
        Apply full forward model with sensor and noise models to the input signal and gather results.

        .. math::

            y = N(A(x))

        :param torch.Tensor x: input signal.
        :param bool reduce: whether to gather results across ranks. If `False`, returns local measurements.
        :param dict kwargs: optional parameters for the forward model.
        :return: complete list of noisy measurements from all operators.
        """
        return self._map_reduce(
            x, lambda p, x, **kw: p.forward(x, **kw), reduce=reduce, **kwargs
        )


class DistributedLinearPhysics(DistributedPhysics, LinearPhysics):
    r"""
    Distributed linear physics operators.

    This class extends :class:`DistributedPhysics` for linear operators. It provides distributed
    operations that automatically handle communication and reductions.

    .. note::

        This class is intended to distribute a *collection* of linear operators (e.g.,
        :class:`deepinv.physics.StackedLinearPhysics` or an explicit Python list of
        :class:`deepinv.physics.LinearPhysics` objects) across ranks. It is **not** a
        mechanism to shard a single linear operator internally.

    If you have one linear physics operator that can naturally be split into multiple
    operators, you must do that split yourself (build a stacked/list representation) and
    provide those operators through the `factory`.

    All linear operations (`A_adjoint`, `A_vjp`, etc.) support a `reduce` parameter:

        - If `reduce=True` (default): The method computes the global result by performing a single all-reduce across all ranks.
        - If `reduce=False`: The method computes only the local contribution from operators owned by this rank, without any inter-rank communication. This is useful for deferring reductions in custom algorithms.

    :param DistributedContext ctx: distributed context manager.
    :param int num_operators: total number of physics operators to distribute.
    :param Callable factory: factory function that creates linear physics operators.
        Should have signature `factory(index: int, device: torch.device, shared: dict | None) -> LinearPhysics`.
    :param None, dict shared: shared data dictionary passed to factory function for all operators.
    :param str reduction: reduction mode for distributed operations. Options are `'sum'` (stack operators)
        or `'mean'` (average operators). Default is `'sum'`.
    :param None, torch.dtype dtype: data type for operations.
    :param str gather_strategy: strategy for gathering distributed results in forward operations.
        Options are `'naive'`, `'concatenated'`, or `'broadcast'`. Default is `'concatenated'`.
    """

    def __init__(
        self,
        ctx: DistributedContext,
        num_operators: int,
        factory,
        *,
        shared: dict | None = None,
        reduction: str = "sum",
        dtype: torch.dtype | None = None,
        gather_strategy: str = "concatenated",
        **kwargs,
    ):
        r"""
        Initialize distributed linear physics operators.
        """
        self.reduction_mode = reduction
        super().__init__(
            ctx=ctx,
            num_operators=num_operators,
            factory=factory,
            shared=shared,
            dtype=dtype,
            gather_strategy=gather_strategy,
            A=lambda x, **kw: x,
            A_adjoint=lambda y, **kw: y,
            **kwargs,
        )

        for p in self.local_physics:
            if not isinstance(p, LinearPhysics):
                raise ValueError("factory must return LinearPhysics instances.")

    def A_adjoint(
        self, y: TensorList | list[torch.Tensor], reduce: bool = True, **kwargs
    ) -> torch.Tensor:
        r"""
        Compute global adjoint operation with automatic reduction.

        Extracts local measurements, computes local adjoint contributions, and reduces
        across all ranks to obtain the complete :math:`A^T y` where :math:`A` is the
        stacked operator :math:`A = [A_1, A_2, \ldots, A_n]` and :math:`A_i` are the individual linear operators.

        :param TensorList y: full list of measurements from all operators.
        :param bool reduce: whether to reduce results across ranks. If False, returns local contribution.
        :param dict kwargs: optional parameters for the adjoint operation.
        :return: complete adjoint result :math:`A^T y` (or local contribution if reduce=False).
        """
        # Extract local measurements
        if isinstance(y, TensorList):
            y_local = [y[i] for i in self.local_indexes]
        elif len(y) == self.num_operators:
            y_local = [y[i] for i in self.local_indexes]
        elif len(y) == len(self.local_indexes):
            y_local = y
        else:
            raise ValueError(
                f"Input y has length {len(y)}, expected {self.num_operators} (global) or {len(self.local_indexes)} (local)."
            )

        # Use _map_reduce with per-operator inputs and sum_results=True
        # This gathers all A_i^T(y_i) and sums them automatically
        result = self._map_reduce(
            y_local,
            lambda p, y_i, **kw: p.A_adjoint(y_i, **kw),
            reduce=reduce,
            sum_results=True,
            **kwargs,
        )

        # Apply reduction mode normalization if needed
        if reduce and self.reduction_mode == "mean":
            result = result / float(self.num_operators)

        return result

    def A_vjp(
        self,
        x: torch.Tensor,
        v: TensorList | list[torch.Tensor],
        reduce: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Compute global vector-Jacobian product with automatic reduction.

        Extracts local cotangent vectors, computes local VJP contributions, and reduces
        across all ranks to obtain the complete VJP.

        :param torch.Tensor x: input tensor.
        :param TensorList v: full list of cotangent vectors from all operators.
        :param bool reduce: whether to reduce results across ranks. If False, returns local contribution.
        :param dict kwargs: optional parameters for the VJP operation.
        :return: complete VJP result (or local contribution if reduce=False).
        """
        if isinstance(v, TensorList):
            v_local = [v[i] for i in self.local_indexes]
        elif len(v) == self.num_operators:
            v_local = [v[i] for i in self.local_indexes]
        elif len(v) == len(self.local_indexes):
            v_local = v
        else:
            raise ValueError(
                f"Input v has length {len(v)}, expected {self.num_operators} (global) or {len(self.local_indexes)} (local)."
            )

        if len(v_local) == 0:
            # Return zeros with proper shape for empty local set
            local = torch.zeros_like(x)
        else:
            contribs = [
                p.A_vjp(x, v_i, **kwargs) for p, v_i in zip(self.local_physics, v_local)
            ]
            local = torch.stack(contribs, dim=0).sum(0)

        if not reduce:
            return local

        return self._reduce_global(local)

    def A_adjoint_A(
        self, x: torch.Tensor, reduce: bool = True, **kwargs
    ) -> torch.Tensor:
        r"""
        Compute global :math:`A^T A` operation with automatic reduction.

        Computes the complete normal operator :math:`A^T A x = \sum_i A_i^T A_i x` by
        combining local contributions from all ranks.

        :param torch.Tensor x: input tensor.
        :param bool reduce: whether to reduce results across ranks. If False, returns local contribution.
        :param dict kwargs: optional parameters for the operation.
        :return: complete :math:`A^T A x` result (or local contribution if reduce=False).
        """
        if len(self.local_physics) == 0:
            # Return zeros with proper shape for empty local set
            local = torch.zeros_like(x)
        else:
            contribs = [p.A_adjoint_A(x, **kwargs) for p in self.local_physics]
            local = torch.stack(contribs, dim=0).sum(0)

        if not reduce:
            return local

        return self._reduce_global(local)

    def A_A_adjoint(
        self, y: TensorList | list[torch.Tensor], reduce: bool = True, **kwargs
    ) -> TensorList | list[torch.Tensor]:
        r"""
        Compute global :math:`A A^T` operation with automatic reduction.

        For stacked operators, this computes :math:`A A^T y` where :math:`A^T y = \sum_i A_i^T y_i`
        and then applies the forward operator to get :math:`[A_1(A^T y), A_2(A^T y), \ldots, A_n(A^T y)]`.

        .. note::

            Unlike other operations, the adjoint step `A^T y` is always computed globally (with full
            reduction across ranks) even when `reduce=False`. This is because computing the correct
            `A_A_adjoint` requires the full adjoint `sum_i A_i^T y_i`. The `reduce` parameter only
            controls whether the final forward operation `A(...)` is gathered across ranks.

        :param TensorList y: full list of measurements from all operators.
        :param bool reduce: whether to gather final results across ranks. If `False`, returns only local
            operators' contributions (but still uses the global adjoint).
        :param dict kwargs: optional parameters for the operation.
        :return: TensorList with entries :math:`A_i A^T y` for all operators (or local list if `reduce=False`).
        """
        # First compute A^T y globally (always with reduction to get the full adjoint)
        # This is necessary because A_A_adjoint(y) = A(A^T y) and A^T y = sum_i A_i^T y_i
        x_adjoint = self.A_adjoint(y, reduce=True, **kwargs)

        # Then compute A(A^T y) which returns a TensorList (or list if reduce=False)
        return self.A(x_adjoint, reduce=reduce, **kwargs)

    # ---- global (compat) ----
    def _reduce_global(self, x_like: torch.Tensor, reduction="sum") -> torch.Tensor:
        r"""
        Perform all-reduce operation to combine local results across all ranks.

        This is an internal helper that converts local (per-rank) results into global
        results by summing contributions from all ranks. Optionally normalizes by the
        number of operators if mean reduction is requested.

        .. note::

            This method is primarily used by operations like `A_adjoint_A` where
            the result is already a proper tensor from all ranks. For `A_adjoint`, we use
            the `_map_reduce` pattern which handles empty local sets more robustly.

        :param torch.Tensor x_like: local tensor to reduce.
        :param str reduction: reduction mode, either `'sum'` or `'mean'`.
        :return: globally reduced tensor.
        """
        if not torch.is_tensor(x_like):
            # handle 0.0 placeholder for empty local set
            x_like = torch.zeros((), device=self.ctx.device, dtype=self.dtype)

        # Ensure all ranks participate in collective
        if self.ctx.is_dist:
            self.ctx.all_reduce_(x_like, op="sum")

        if self.reduction_mode == "mean" or reduction == "mean":
            x_like = x_like / float(self.num_operators)
        return x_like

    def A_dagger(
        self,
        y: TensorList | list[torch.Tensor],
        solver: str = "CG",
        max_iter: int | None = None,
        tol: float | None = None,
        verbose: bool = False,
        *,
        local_only: bool = True,
        reduce: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Distributed pseudoinverse computation. This method provides two strategies:

            1. **Local approximation** (`local_only=True`, default): Each rank computes the pseudoinverse
            of its local operators independently, then averages the results with a single reduction.
            This is efficient (minimal communication) but **provides only an approximation**.
            In other words, for stacked operators this computes

            .. math::

                A^\dagger y = \frac{1}{n} \sum_i A_i^\dagger y_i

            2. **Global computation** (`local_only=False`): Uses the full least squares solver
            with distributed :meth:`A_adjoint_A` and :meth:`A_A_adjoint` operations.
            This computes the exact pseudoinverse but requires communication at every iteration.

        :param TensorList y: measurements to invert.
        :param str solver: least squares solver to use (only for `local_only=False`).
            Choose between `'CG'`, `'lsqr'`, `'BiCGStab'` and `'minres'`.
        :param None, int max_iter: maximum number of iterations for least squares solver.
        :param None, float tol: relative tolerance for least squares solver.
        :param bool verbose: print information (only on rank 0).
        :param bool local_only: If `True` (default), compute local daggers and sum-reduce (efficient).
            If `False`, compute exact global pseudoinverse with full communication (expensive).
        :param bool reduce: whether to reduce results across ranks (only applies if local_only=True).
        :param dict kwargs: optional parameters for the forward operator.

        :return: pseudoinverse solution. If `local_only=True`, returns approximation.
            If `local_only=False`, returns exact least squares solution.
        """
        if local_only:
            # Efficient local computation with single sum reduction
            if isinstance(y, TensorList):
                y_local = [y[i] for i in self.local_indexes]
            elif len(y) == self.num_operators:
                y_local = [y[i] for i in self.local_indexes]
            elif len(y) == len(self.local_indexes):
                y_local = y
            else:
                raise ValueError(
                    f"Input y has length {len(y)}, expected {self.num_operators} (global) or {len(self.local_indexes)} (local)."
                )

            if len(y_local) == 0:
                local = torch.zeros((), device=self.ctx.device, dtype=self.dtype)
            else:
                contribs = [
                    p.A_dagger(y_i, **kwargs)
                    for p, y_i in zip(self.local_physics, y_local)
                ]
                local = torch.stack(contribs, dim=0).sum(0)

            if not reduce:
                return local

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
        reduce: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Computes the squared spectral :math:`\ell_2` norm of the distributed operator.

        This method provides two strategies:

            1. **Local approximation** (`local_only=True`, default): Each rank computes the norm
            of its local operators independently, then a single max-reduction provides an upper bound.
            This is efficient (minimal communication) and valid for conservative estimates.
            For stacked operators :math:`A = [A_1; A_2; \ldots; A_n]`, we have
            :math:`\|A\|^2 \leq \sum_i \|A_i\|^2`, and we use :math:`\max_i \|A_i\|^2` as
            a conservative upper bound.

            2. **Global computation** (`local_only=False`): Uses the full distributed :meth:`A_adjoint_A`
            with communication at every power iteration. This computes the exact norm but is
            communication-intensive.

        :param torch.Tensor x0: an unbatched tensor sharing its shape, dtype and device with the initial iterate
        :param int max_iter: maximum number of iterations for power method
        :param float tol: relative variation criterion for convergence
        :param bool verbose: print information (only on rank 0)
        :param bool local_only: If `True` (default), compute local norms and max-reduce (efficient).
            If `False`, compute exact global norm with full communication (expensive).
        :param bool reduce: whether to reduce results across ranks (only applies if local_only=True).
        :param dict kwargs: optional parameters for the forward operator

        :return: Squared spectral norm. If `local_only=True`, returns upper bound.
            If `local_only=False`, returns exact value.
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

            if not reduce:
                return local_sqnorm

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

    |sep|

    **Example use cases:**

        - Distributed denoising of large images/volumes
        - Applying neural network priors across multiple GPUs
        - Processing signals too large to fit on a single device

    :param DistributedContext ctx: distributed context manager.
    :param Callable[[torch.Tensor], torch.Tensor] processor: processing function to apply to signal patches.
        Should accept a batched tensor of shape ``(N, C, ...)`` and return a tensor of the same shape.
        Examples: denoiser, neural network, prior gradient function, etc.
    :param str | DistributedSignalStrategy strategy: signal processing strategy for patch extraction
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
        strategy: str | DistributedSignalStrategy | None = None,
        strategy_kwargs: dict | None = None,
        max_batch_size: int | None = None,
        **kwargs,
    ):
        r"""
        Initialize distributed signal processor.
        """
        self.ctx = ctx
        self.processor = processor
        self.max_batch_size = max_batch_size
        self.strategy = strategy if strategy is not None else "smart_tiling"
        self.strategy_kwargs = strategy_kwargs

        if hasattr(processor, "to"):
            self.processor.to(ctx.device)

    def __call__(
        self, x: torch.Tensor, *args, reduce: bool = True, **kwargs
    ) -> torch.Tensor:
        r"""
        Apply distributed processing to input signal.

        :param torch.Tensor x: input signal tensor to process, typically of shape ``(B, C, H, W)`` for 2D
            or ``(B, C, D, H, W)`` for 3D signals.
        :param bool reduce: whether to reduce results across ranks. If False, returns local contribution.
        :param args: additional positional arguments passed to the processor.
        :param kwargs: additional keyword arguments passed to the processor.
        :return: processed signal with the same shape as input.
        """
        self._init_shape_and_strategy(x.shape)
        return self._apply_op(x, *args, reduce=reduce, **kwargs)

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
            from .strategies import create_strategy

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

    def _apply_op(
        self, x: torch.Tensor, *args, reduce: bool = True, **kwargs
    ) -> torch.Tensor:
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
        :param bool reduce: whether to reduce results across ranks. If False, returns local contribution.
        :param args: additional positional arguments passed to the processor.
        :param kwargs: additional keyword arguments passed to the processor.
        :return: processed signal with the same shape as input.
        """
        # Handle empty case early
        if not self.local_indices:
            out_local = torch.zeros(
                self.signal_shape, device=self.ctx.device, dtype=x.dtype
            )
            if self.ctx.is_dist and reduce:
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

        processed_pairs = list(zip(self.local_indices, processed_patches))

        # 6. Initialize output tensor and apply reduction strategy
        out_local = torch.zeros(
            self.signal_shape, device=self.ctx.device, dtype=x.dtype
        )
        self._strategy.reduce_patches(out_local, processed_pairs)

        # 7. All-reduce to combine results from all ranks
        if self.ctx.is_dist and reduce:
            self.ctx.all_reduce_(out_local, op="sum")

        return out_local


# =========================
# Distributed Data Fidelity
# =========================
class DistributedDataFidelity:
    r"""
    Distributed data fidelity term for use with distributed physics operators.

    This class wraps a standard DataFidelity object and makes it compatible with
    DistributedLinearPhysics by implementing efficient distributed computation patterns.
    It computes data fidelity terms and gradients using local operations followed by
    a single reduction, avoiding redundant communication.

    The key operations are:

        - ``fn(x, y, physics)``: Computes the data fidelity :math:`\sum_i d(A_i(x), y_i)`
        - ``grad(x, y, physics)``: Computes the gradient :math:`\sum_i A_i^T \nabla d(A_i(x), y_i)`

    Both operations use an efficient pattern:

        1. Compute local forward operations (A_local)
        2. Apply distance function and compute gradients locally
        3. Perform a single reduction across ranks

    :param DistributedContext ctx: distributed context manager.
    :param DataFidelity | Callable data_fidelity: either a DataFidelity instance
        or a factory function that creates DataFidelity instances for each operator.
        The factory should have signature
        ``factory(index: int, device: torch.device, shared: dict | None) -> DataFidelity``.
    :param None, int num_operators: number of operators (required if data_fidelity is a factory).
    :param None, dict shared: shared data dictionary passed to factory function for all operators.
    :param str reduction: reduction mode matching the distributed physics. Options are ``'sum'`` or ``'mean'``.
        Default is ``'sum'``.

    |sep|

    :Example:

        >>> from deepinv.distributed import DistributedContext, distribute
        >>> from deepinv.optim import L2
        >>> # Create distributed physics and data fidelity
        >>> with DistributedContext(device_mode="cpu") as ctx:
        ...     physics_list = [create_physics(i) for i in range(4)]
        ...     dist_physics = distribute(physics_list, ctx=ctx)
        ...     data_fidelity = L2()
        ...     dist_fidelity = DistributedDataFidelity(ctx, data_fidelity)
        ...     # Compute fidelity and gradient
        ...     x = torch.randn(1, 1, 16, 16)
        ...     y = dist_physics(x)
        ...     fid = dist_fidelity.fn(x, y, dist_physics)
        ...     grad = dist_fidelity.grad(x, y, dist_physics)
    """

    def __init__(
        self,
        ctx: DistributedContext,
        data_fidelity: (
            DataFidelity | Callable[[int, torch.device, dict | None], DataFidelity]
        ),
        num_operators: int | None = None,
        *,
        shared: dict | None = None,
        reduction: str = "sum",
    ):
        r"""
        Initialize distributed data fidelity.

        :param DistributedContext ctx: distributed context manager.
        :param DataFidelity | Callable data_fidelity: data fidelity term or factory.
        :param int | None num_operators: number of operators (required if data_fidelity is a factory).
        :param str reduction: reduction mode for distributed operations. Options are ``'sum'`` and ``'mean'``.
        """
        self.ctx = ctx
        self.reduction_mode = reduction
        self.local_data_fidelities = []
        self.single_fidelity = None

        if isinstance(data_fidelity, DataFidelity):
            self.single_fidelity = copy.deepcopy(data_fidelity)
            if hasattr(self.single_fidelity, "to"):
                self.single_fidelity.to(ctx.device)
        elif callable(data_fidelity):
            if num_operators is None:
                raise ValueError("num_operators must be provided when using a factory.")
            # Create local data fidelity instances using factory
            local_indexes = list(ctx.local_indices(num_operators))
            for i in local_indexes:
                df = data_fidelity(i, ctx.device, shared)
                self.local_data_fidelities.append(df)
        else:
            raise ValueError(
                "data_fidelity must be a DataFidelity instance or a factory callable."
            )

    def _get_fidelity(self, i: int) -> DataFidelity:
        if self.single_fidelity is not None:
            return self.single_fidelity
        return self.local_data_fidelities[i]

    def _check_is_distributed_physics(self, physics: DistributedLinearPhysics):
        if not isinstance(physics, DistributedLinearPhysics):
            raise ValueError(
                "physics must be a DistributedLinearPhysics instance to be used with DistributedDataFidelity."
            )

    def fn(
        self,
        x: torch.Tensor,
        y: list[torch.Tensor],
        physics: DistributedLinearPhysics,
        reduce: bool = True,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Compute the distributed data fidelity term.

        For distributed physics with operators :math:`\{A_i\}` and measurements :math:`\{y_i\}`,
        computes:

        .. math::

            f(x) = \sum_i d(A_i(x), y_i)

        This is computed efficiently by:

            1. Each rank computes :math:`A_i(x)` for its local operators
            2. Each rank computes :math:`\sum_{i \in \text{local}} d(A_i(x), y_i)`
            3. Results are reduced across all ranks

        :param torch.Tensor x: input signal at which to evaluate the data fidelity.
        :param list[torch.Tensor] y: measurements (TensorList or list of tensors).
        :param DistributedLinearPhysics physics: distributed physics operator.
        :param dict kwargs: additional arguments passed to the distance function.
        :return: scalar data fidelity value.
        """
        self._check_is_distributed_physics(physics)

        # Get local measurements
        y_local = [y[i] for i in physics.local_indexes]

        # Compute A(x) locally
        Ax_local = physics.A(x, reduce=False, **kwargs)

        # Compute distance function for each local operator
        if len(Ax_local) == 0:
            # Empty local set - return zero contribution
            result_local = torch.tensor(0.0, device=self.ctx.device)
        else:
            contribs = [
                self._get_fidelity(i).d.fn(Ax_i, y_i, *args, **kwargs)
                for i, (Ax_i, y_i) in enumerate(zip(Ax_local, y_local))
            ]
            result_local = torch.stack(contribs, dim=0).sum(0)

        if not reduce:
            return result_local

        # Reduce across ranks
        return self.ctx.all_reduce_(result_local, op=self.reduction_mode)

    def grad(
        self,
        x: torch.Tensor,
        y: list[torch.Tensor],
        physics: DistributedLinearPhysics,
        reduce: bool = True,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Compute the gradient of the distributed data fidelity term.

        For distributed physics with operators :math:`\{A_i\}` and measurements :math:`\{y_i\}`,
        computes:

        .. math::

            \nabla_x f(x) = \sum_i A_i^T \nabla d(A_i(x), y_i)

        This is computed efficiently by:

            1. Each rank computes :math:`A_i(x)` for its local operators
            2. Each rank computes :math:`\nabla d(A_i(x), y_i)` for its local operators
            3. Each rank computes :math:`\sum_{i \in \text{local}} A_i^T \nabla d(A_i(x), y_i)` using A_vjp_local
            4. Results are reduced across all ranks

        :param torch.Tensor x: input signal at which to compute the gradient.
        :param list[torch.Tensor] y: measurements (TensorList or list of tensors).
        :param DistributedLinearPhysics physics: distributed physics operator.
        :param dict kwargs: additional arguments passed to the distance function gradient.
        :return: gradient with same shape as x.
        """
        self._check_is_distributed_physics(physics)

        # Get local measurements
        y_local = [y[i] for i in physics.local_indexes]

        # Compute A(x) locally
        Ax_local = physics.A(x, reduce=False, **kwargs)

        # Compute gradient of distance for each local operator
        if len(Ax_local) == 0:
            # Empty local set - return zero contribution
            grad_local = torch.zeros_like(x)
        else:
            # Compute gradients w.r.t. Ax
            grad_d_local = [
                self._get_fidelity(i).d.grad(Ax_i, y_i, *args, **kwargs)
                for i, (Ax_i, y_i) in enumerate(zip(Ax_local, y_local))
            ]
            # Apply A_vjp locally (this is A^T @ grad_d)
            grad_local = physics.A_vjp(x, grad_d_local, reduce=False, **kwargs)

        if not reduce:
            return grad_local

        # Reduce across ranks
        return self.ctx.all_reduce_(grad_local, op=self.reduction_mode)
