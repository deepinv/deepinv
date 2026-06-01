from __future__ import annotations

from typing import Callable, Sequence
import warnings

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint as torch_checkpoint


from deepinv.distributed.strategies import DistributedSignalStrategy, create_strategy
from deepinv.distributed.framework.distributed_utils import DistributedParameterSync
from deepinv.distributed.framework.distributed_context import DistributedContext


class DistributedProcessing(torch.nn.Module):
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
    :param str | DistributedSignalStrategy | None strategy: signal processing strategy for patch extraction
        and reduction. Either a strategy name (``'basic'``, ``'overlap_tiling'``) or a custom strategy instance.
        Default is ``'overlap_tiling'`` which handles overlapping patches with smooth blending.
    :param dict | None strategy_kwargs: additional keyword arguments passed to the strategy constructor
        when using string strategy names. Examples: ``patch_size``, ``overlap``, ``tiling_dims``. Default is `None`.
    :param int | None max_batch_size: maximum number of patches to process in a single batch.
        If ``None``, all local patches are batched together. Set to ``1`` for sequential processing
        (useful for memory-constrained scenarios). Higher values increase throughput but require more memory. Default is `None`.
    :param str checkpoint_batches: activation checkpointing mode for patch-batches during backward.
        Supported values are ``'auto'``, ``'always'`` and ``'never'``.

        - ``'auto'`` (default): enable checkpointing only when gradients are enabled and there are multiple local patch-batches.
        - ``'always'``: always checkpoint patch-batches when gradients are enabled.
        - ``'never'``: disable checkpointing.

    :param bool checkpoint_use_reentrant: reentrant mode passed to :func:`torch.utils.checkpoint.checkpoint`.
        Default is ``False`` (recommended by PyTorch).
    :param bool checkpoint_preserve_rng_state: whether to preserve RNG state across forward recomputation when
        checkpointing. Default is ``True``.
    """

    def __init__(
        self,
        ctx: DistributedContext,
        processor: Callable[[torch.Tensor], torch.Tensor],
        *,
        strategy: str | DistributedSignalStrategy | None = None,
        strategy_kwargs: dict | None = None,
        max_batch_size: int | None = None,
        checkpoint_batches: str = "auto",
        checkpoint_use_reentrant: bool = False,
        checkpoint_preserve_rng_state: bool = True,
        **kwargs,
    ):
        r"""
        Initialize distributed signal processor.
        """
        super().__init__()
        self.ctx = ctx
        self.processor = processor
        self.max_batch_size = max_batch_size
        valid_checkpoint_modes = ("auto", "always", "never")
        if checkpoint_batches not in valid_checkpoint_modes:
            raise ValueError(
                "checkpoint_batches must be one of "
                f"{valid_checkpoint_modes}, got '{checkpoint_batches}'."
            )
        self.checkpoint_batches = checkpoint_batches
        self.checkpoint_use_reentrant = checkpoint_use_reentrant
        self.checkpoint_preserve_rng_state = checkpoint_preserve_rng_state
        self.strategy = strategy if strategy is not None else "overlap_tiling"
        self.strategy_kwargs = strategy_kwargs or {}
        self.current_shape: torch.Size | None = None

        if hasattr(processor, "to"):
            self.processor.to(ctx.device)

    def forward(
        self, x: torch.Tensor, *args, gather: bool = True, **kwargs
    ) -> torch.Tensor:
        r"""
        Apply distributed processing to input signal.

        :param torch.Tensor x: input signal tensor to process, typically of shape ``(B, C, H, W)`` for 2D
            or ``(B, C, D, H, W)`` for 3D signals.
        :param args: additional positional arguments passed to the processor.
        :param bool gather: whether to gather results across ranks. If False, returns local contribution. Default is `True`.
        :param kwargs: additional keyword arguments passed to the processor.
        :return: processed signal with the same shape as input.
        """

        if self.current_shape != x.shape:
            self._init_shape_and_strategy(x.shape)
            self.current_shape = x.shape
        return self._apply_op(x, *args, gather=gather, **kwargs)

    # ---- internals --------------------------------------------------------

    def _init_shape_and_strategy(
        self,
        img_size: Sequence[int],
    ):
        r"""
        Initialize or update the signal shape and processing strategy.

        This method is called automatically on the first forward pass to set up the
        tiling strategy based on the input signal dimensions. It creates the strategy,
        determines the number of patches, and assigns patches to ranks.

        :param Sequence[int] img_size: full shape of the input signal tensor (e.g., ``(B, C, H, W)``).
        """

        self.img_size = torch.Size(img_size)
        strategy_kwargs = dict(self.strategy_kwargs)
        tiling_dims = strategy_kwargs.pop("tiling_dims", None)

        # Create or set the strategy
        self._strategy = (
            create_strategy(
                self.strategy, img_size, tiling_dims=tiling_dims, **strategy_kwargs
            )
            if isinstance(self.strategy, str)
            else self.strategy
        )

        if self._strategy is None:
            raise RuntimeError(
                "Distributed strategy is None - failed to create or import strategy"
            )

        self.num_patches = self._strategy.get_num_patches()
        if self.num_patches == 0:
            raise ValueError("Distributed strategy produced zero patches.")

        # determine local patch indices for this rank
        self.local_indices: list[int] = list(self.ctx.local_indices(self.num_patches))

        # Check for insufficient work distribution
        if self.ctx.use_dist and self.ctx.rank == 0:
            ranks_with_work = min(self.num_patches, self.ctx.world_size)
            if ranks_with_work < self.ctx.world_size:
                warnings.warn(
                    f"Only {ranks_with_work}/{self.ctx.world_size} ranks have patches to process. "
                    f"Some ranks will have no work. "
                    f"Current: {self.num_patches} patches for {self.ctx.world_size} ranks.",
                    UserWarning,
                )

    def _apply_op(
        self, x: torch.Tensor, *args, gather: bool = True, **kwargs
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
        :param bool gather: whether to gather results across ranks. If False, returns local contribution.
        :param args: additional positional arguments passed to the processor.
        :param kwargs: additional keyword arguments passed to the processor.
        :return: processed signal with the same shape as input.
        """
        # Determine if we should sync parameters (only if we hold a module)
        should_sync = self.ctx.use_dist and isinstance(self.processor, torch.nn.Module)

        if should_sync:
            params = [p for p in self.processor.parameters() if p.requires_grad]
            if params:
                x = DistributedParameterSync.apply(x, self.ctx, *params)

        # Handle empty case early
        if not self.local_indices:
            out_local = torch.zeros(
                self.img_size, device=self.ctx.device, dtype=x.dtype
            )
            # Ensure dependency on x if we are syncing, to trigger backward hook
            if should_sync and x.requires_grad:
                # Minimal operation to link graph
                out_local = out_local + 0 * x.view(-1)[0]

            if gather:
                out_local = self.ctx.all_reduce(out_local, op=dist.ReduceOp.SUM)
            return out_local

        # 1. Extract local patches using strategy
        local_pairs = self._strategy.get_local_patches(x, self.local_indices)
        patches = [patch for _, patch in local_pairs]

        # 2. Apply batching strategy with max_batch_size
        batched_patches = self._strategy.apply_batching(
            patches, max_batch_size=self.max_batch_size
        )

        processor_has_trainable_params = False
        if isinstance(self.processor, torch.nn.Module):
            processor_has_trainable_params = any(
                p.requires_grad for p in self.processor.parameters()
            )
        grad_enabled = torch.is_grad_enabled() and (
            x.requires_grad or processor_has_trainable_params
        )
        use_batch_checkpointing = False
        if grad_enabled:
            if self.checkpoint_batches == "always":
                use_batch_checkpointing = True
            elif self.checkpoint_batches == "auto":
                use_batch_checkpointing = len(batched_patches) > 1

        # 3. Apply processor to each batch
        processed_batches = []
        for batch in batched_patches:
            if use_batch_checkpointing:
                result = torch_checkpoint(
                    lambda b: self.processor(b, *args, **kwargs),
                    batch,
                    use_reentrant=self.checkpoint_use_reentrant,
                    preserve_rng_state=self.checkpoint_preserve_rng_state,
                )
            else:
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

        processed_pairs = list(zip(self.local_indices, processed_patches, strict=True))

        # 6. Initialize output tensor and apply reduction strategy
        out_local = torch.zeros(self.img_size, device=self.ctx.device, dtype=x.dtype)
        self._strategy.reduce_patches(out_local, processed_pairs)

        # 7. All-reduce to combine results from all ranks
        if gather:
            out_local = self.ctx.all_reduce(out_local, op=dist.ReduceOp.SUM)

        return out_local
