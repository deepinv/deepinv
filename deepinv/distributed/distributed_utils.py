from __future__ import annotations
import torch
import torch.distributed as dist
from typing import TYPE_CHECKING, Callable, Any, Sequence

from deepinv.utils.tensorlist import TensorList

if TYPE_CHECKING:
    from deepinv.distributed.distrib_framework import DistributedContext


def map_reduce_gather(
    ctx: DistributedContext,
    local_items: Sequence[Any],
    x: torch.Tensor | list[Any],
    local_op: Callable[[Any, Any], torch.Tensor],
    local_indices: list[int],
    num_operators: int,
    gather_strategy: str = "concatenated",
    dtype: torch.dtype = torch.float32,
    gather: bool = True,
    reduce_op: str | None = None,
    **kwargs,
) -> TensorList | list[torch.Tensor] | torch.Tensor:
    r"""
    Generic map-reduce-gather pattern for distributed operations.

    Iterates over local items and input data, applies a local operation, and then
    computes the result globally using reduction or gathering.
    """

    # Handle inputs
    if isinstance(x, (list, tuple)):
        if len(x) != len(local_items):
            raise ValueError(
                f"When passing list/tuple input, length must match local items: "
                f"got {len(x)}, expected {len(local_items)}"
            )
        local_results = [
            local_op(item, xi, **kwargs) for item, xi in zip(local_items, x)
        ]
    else:
        local_results = [local_op(item, x, **kwargs) for item in local_items]

    if reduce_op is not None:
        # Map-Reduce path
        return reduce_local_results(
            ctx,
            local_results,
            reduce_op=reduce_op,
            reduce_globally=gather,
            dtype=dtype,
            num_operators=num_operators,
        )

    if not gather:
        return local_results

    if not ctx.use_dist:
        out: list = [None] * num_operators
        for idx, result in zip(local_indices, local_results):
            out[idx] = result
        return TensorList(out)

    # Map-Gather path
    if gather_strategy == "naive":
        gathered = gather_tensorlist_naive(
            ctx,
            local_indices,
            local_results,
            num_operators,
        )
    elif gather_strategy == "concatenated":
        gathered = gather_tensorlist_concatenated(
            ctx,
            local_indices,
            local_results,
            num_operators,
        )
    elif gather_strategy == "broadcast":
        gathered = gather_tensorlist_broadcast(
            ctx,
            local_indices,
            local_results,
            num_operators,
        )
    else:
        raise ValueError(f"Unknown gather strategy: {gather_strategy}")

    return TensorList([gathered[i] for i in range(num_operators)])


def single_process_fallback(
    local_indices: list[int],
    local_results: list[torch.Tensor],
    num_operators: int,
) -> TensorList:
    r"""
    Fallback gather strategy for single-process (non-distributed) execution.

    :param list[int] local_indices: indices owned by this rank
    :param list[torch.Tensor] local_results: local tensor results
    :param int num_operators: total number of operators
    :return: TensorList with all results
    """
    out: list = [None] * num_operators
    for idx, result in zip(local_indices, local_results):
        out[idx] = result
    return TensorList(out)


def reduce_local_results(
    ctx: DistributedContext,
    local_results: list[torch.Tensor],
    reduce_op: str = "sum",
    reduce_globally: bool = True,
    dtype: torch.dtype = torch.float32,
    num_operators: int | None = None,
) -> torch.Tensor:
    r"""
    Reduce results from local operators across all ranks (Map-Reduce pattern).

    Sums local results and optionally reduces them across all ranks.
    Ensures consistent tensor shapes across ranks before reduction to handle cases
    where some ranks might have no operators (empty local results).

    :param DistributedContext ctx: distributed context manager.
    :param list[torch.Tensor] local_results: list of results from local operators.
    :param str reduce_op: reduction operation, 'sum' or 'mean'.
    :param bool reduce_globally: whether to reduce across ranks (all_reduce).
        If `False`, returns the local sum (but still synchronizes shapes).
    :param torch.dtype dtype: data type for operations.
    :param int | None num_operators: total number of operators (required for 'mean' reduction).
    :return: reduced tensor (global or local).
    """

    if reduce_op not in ("sum", "mean"):
        raise ValueError(f"reduce_op must be 'sum' or 'mean', got '{reduce_op}'")

    # 1. Local Sum
    if len(local_results) == 0:
        local_val = torch.zeros((), device=ctx.device, dtype=dtype)
    else:
        local_val = torch.stack(local_results, dim=0).sum(dim=0)

    # 2. Shape synchronization (Broadcast from Rank 0)
    # Required for all_reduce to work if ranks have different Shapes (e.g. rank 0 has results, rank 1 has none)
    if ctx.use_dist and reduce_op is not None:
        if ctx.rank == 0:
            ndim = torch.tensor([local_val.ndim], device=ctx.device, dtype=torch.long)
        else:
            ndim = torch.zeros(1, device=ctx.device, dtype=torch.long)
        ctx.broadcast(ndim, src=0)

        D = ndim.item()

        if D > 0:  # Only if rank 0 result is not a scalar
            if ctx.rank == 0:
                shape = torch.tensor(
                    local_val.shape, device=ctx.device, dtype=torch.long
                )
            else:
                shape = torch.zeros(D, device=ctx.device, dtype=torch.long)
            ctx.broadcast(shape, src=0)

            target_shape = tuple(shape.tolist())

            # Resize local zero-scalar if needed to match target shape
            if local_val.ndim == 0 and local_val.numel() == 1 and local_val.item() == 0:
                if local_val.shape != target_shape:
                    local_val = torch.zeros(
                        target_shape, device=ctx.device, dtype=local_val.dtype
                    )

    if not reduce_globally:
        return local_val

    # 3. Global Reduction
    if ctx.use_dist:
        ctx.all_reduce(local_val, op=dist.ReduceOp.SUM)

    if reduce_op == "mean":
        if num_operators is None:
            raise ValueError("num_operators is required for 'mean' reduction")
        # In local mode (use_dist=False), local_val is the sum over ALL operators (since all local)
        # So we divide by num_operators too.
        local_val = local_val / float(num_operators)

    return local_val


def gather_tensorlist_naive(
    ctx: DistributedContext,
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
    if not ctx.use_dist:
        return single_process_fallback(local_indices, local_results, num_operators)

    # Check if we're using NCCL backend (not compatible with all_gather_object)
    if ctx.get_backend() == "nccl":
        raise RuntimeError(
            "The 'naive' gather strategy uses all_gather_object which is not supported "
            "by the NCCL backend (GPU multi-process mode). Please use 'concatenated' or "
            "'broadcast' gather strategies instead."
        )

    # Pair indices with tensors
    pairs = list(zip(local_indices, local_results))

    # Gather all pairs from all ranks
    gathered = [None] * ctx.world_size
    ctx.all_gather_object(gathered, pairs)

    # Assemble into output list
    out: list = [None] * num_operators
    for rank_pairs in gathered:
        if rank_pairs is not None:
            for idx, tensor in rank_pairs:
                out[idx] = tensor

    return TensorList(out)


def gather_tensorlist_concatenated(
    ctx: DistributedContext,
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
    if not ctx.use_dist:
        return single_process_fallback(local_indices, local_results, num_operators)

    # Step 1: Share metadata (indices, shapes, dtypes)
    local_metadata = [
        (idx, tuple(result.shape), result.dtype, result.numel())
        for idx, result in zip(local_indices, local_results)
    ]

    gathered_metadata = [None] * ctx.world_size
    ctx.all_gather_object(gathered_metadata, local_metadata)

    # Flatten all metadata
    all_metadata = []
    for rank_metadata in gathered_metadata:
        if rank_metadata is not None:
            all_metadata.extend(rank_metadata)

    # Determine canonical dtype for all_gather (use first tensor's dtype for consistency)

    # Check that all tensors have the same dtype
    dtypes = {meta[2] for meta in all_metadata}
    if len(dtypes) > 1:
        raise RuntimeError(
            f"All tensors must have the same dtype for concatenated gather, but found: {dtypes}"
        )

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
                    device=ctx.device,
                )
                flat = torch.cat([flat, padding])
            flattened_padded.append(flat)

        # Concatenate all local flattened tensors
        local_concat = torch.stack(flattened_padded)  # Shape: [num_local, max_numel]
    else:
        # This rank has no tensors - create empty placeholder
        if all_metadata:
            max_numel = max(numel for _, _, _, numel in all_metadata)
        else:
            max_numel = 1

        local_concat = torch.zeros(
            (0, max_numel), dtype=canonical_dtype, device=ctx.device
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
                device=ctx.device,
            )
        else:
            padding = torch.zeros(
                (padding_rows, max_numel),
                dtype=local_concat.dtype,
                device=ctx.device,
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
            device=ctx.device,
        )
        for _ in range(ctx.world_size)
    ]

    # All-gather the actual data
    ctx.all_gather(tensor_lists, local_concat_padded)

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
    ctx: DistributedContext,
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
    if not ctx.use_dist:
        # Single process: just build the list
        return single_process_fallback(local_indices, local_results, num_operators)

    # Step 1: Share metadata (indices, shapes, dtypes)
    local_metadata = [
        (idx, tuple(result.shape), result.dtype)
        for idx, result in zip(local_indices, local_results)
    ]

    gathered_metadata = [None] * ctx.world_size
    ctx.all_gather_object(gathered_metadata, local_metadata)

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
        responsible_rank = idx % ctx.world_size  # Round-robin sharding

        # Prepare tensor to send/receive
        if idx in local_indices:
            # This rank owns this operator
            local_pos = local_indices.index(idx)
            tensor_to_send = local_results[local_pos].contiguous()
        else:
            # Create receive buffer
            tensor_to_send = torch.zeros(shape, dtype=dtype, device=ctx.device)

        # Broadcast from owner to all ranks
        ctx.broadcast(tensor_to_send, src=responsible_rank)
        out[idx] = tensor_to_send

    return TensorList(out)
