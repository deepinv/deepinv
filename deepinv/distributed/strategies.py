r"""
Distributed signal processing strategies for the deepinv library.

This module provides abstract base classes and concrete implementations for
distributed signal processing, including splitting, batching, and reduction operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch

from deepinv.distributed.utils import (
    tiling_splitting_strategy,
    tiling_reduce_fn,
)

Index = tuple[slice, ...]


class DistributedSignalStrategy(ABC):
    r"""
    Abstract base class for distributed signal processing strategies.

    A strategy defines how to:
        1. Split a signal into patches for distributed processing
        2. Batch patches for efficient processing
        3. Reduce processed patches back into a complete signal

    This allows users to implement custom distributed processing strategies
    for different types of data and use cases.

    :param Sequence[int] img_size: shape of the complete signal tensor (e.g., `[B, C, H, W]`).
    """

    def __init__(self, img_size: Sequence[int], **kwargs):
        r"""
        Initialize the strategy.
        """
        self.img_size = torch.Size(img_size)

    @abstractmethod
    def get_local_patches(
        self, x: torch.Tensor, local_indices: list[int]
    ) -> list[tuple[int, torch.Tensor]]:
        r"""
        Extract and prepare local patches for processing.

        :param torch.Tensor X: the complete signal tensor.
        :param list[int] local_indices: global indices of patches assigned to this rank.
        :return: list of (global_index, prepared_patch) pairs ready for processing.
        """
        pass

    @abstractmethod
    def reduce_patches(
        self, out_tensor: torch.Tensor, local_pairs: list[tuple[int, torch.Tensor]]
    ) -> None:
        r"""
        Reduce processed patches into the output tensor.

        This operates in-place on `out_tensor`, placing each processed patch
        in its correct location within the complete signal.

        :param torch.Tensor out_tensor: output tensor to fill (should be initialized to zeros).
        :param list[tuple[int, torch.Tensor]] local_pairs: list of (global_index, processed_patch) pairs.
        """
        pass

    @abstractmethod
    def get_num_patches(self) -> int:
        r"""
        Get the total number of patches this strategy creates.

        :return: total number of patches.
        """
        pass

    def apply_batching(
        self, patches: list[torch.Tensor], max_batch_size: int | None = None
    ) -> list[torch.Tensor]:
        r"""
        Group patches into batches for efficient processing.

        The batching should preserve order: when the batched tensors are processed
        and then concatenated back, they should yield patches in the same order
        as the input.

        :param list[torch.Tensor] patches: list of prepared patches.
        :param int | None max_batch_size: maximum number of patches per batch. If `None`, all patches are batched together. If `1`, each patch is processed individually.
        :return: batched patches ready for processing. When processed results are concatenated, they should preserve the original patch order.
        """
        if not patches:
            return []

        # Verify all patches have the same shape
        expected_shape = patches[0].shape
        for i, patch in enumerate(patches):
            if patch.shape != expected_shape:
                raise RuntimeError(
                    f"Patch {i} has shape {patch.shape}, expected {expected_shape}"
                )

        # Store metadata for unpacking
        self._batching_metadata = {
            "num_patches": len(patches),
            "original_batch_size": patches[0].shape[0] if patches else 0,
            "patch_shape": expected_shape,
        }

        # If max_batch_size is None or >= num patches, batch all together
        if (
            max_batch_size is None
            or max_batch_size >= len(patches)
            or max_batch_size <= 0
        ):
            # Concatenate all patches along batch dimension
            batch = torch.cat(patches, dim=0)
            return [batch]

        # Otherwise, split into multiple batches
        batches = []
        for i in range(0, len(patches), max_batch_size):
            batch_patches = patches[i : i + max_batch_size]
            batch = torch.cat(batch_patches, dim=0)
            batches.append(batch)

        return batches

    def unpack_batched_results(
        self, processed_batches: list[torch.Tensor], num_patches: int
    ) -> list[torch.Tensor]:
        r"""
        Unpack processed batches back to individual patches.

        Default implementation: concatenate along batch dimension and split back.
        Uses stored metadata to determine original patch batch size.

        :param list[torch.Tensor] processed_batches: results from processing batched patches.
        :param int num_patches: expected number of individual patches.
        :return: list of individual processed patches in original order.
        """
        if len(processed_batches) == 0:
            return []

        # Use metadata if available to get original batch size per patch
        original_batch_size = 1
        if hasattr(self, "_batching_metadata"):
            original_batch_size = self._batching_metadata.get("original_batch_size", 1)

        # Concatenate all batches
        if len(processed_batches) == 1:
            all_batched = processed_batches[0]
        else:
            all_batched = torch.cat(processed_batches, dim=0)

        # Split back into individual patches
        # Each patch has original_batch_size elements in the batch dimension
        patches = []
        total_batch_size = all_batched.shape[0]
        expected_total = num_patches * original_batch_size

        if total_batch_size != expected_total:
            raise RuntimeError(
                f"Batch size mismatch: got {total_batch_size}, "
                f"expected {num_patches} patches × {original_batch_size} batch size = {expected_total}"
            )

        for i in range(num_patches):
            start = i * original_batch_size
            end = (i + 1) * original_batch_size
            patch = all_batched[start:end]
            patches.append(patch)

        return patches


class BasicStrategy(DistributedSignalStrategy):
    r"""
    Basic distributed strategy with naive splitting along specified dimensions.

    This strategy:
        - Splits the signal into blocks along specified dimensions
        - Processes patches individually (no batching)
        - Uses simple tensor assignment for reduction

    :param Sequence[int] img_size: shape of the complete signal tensor.
    :param int | tuple[int, ...] tiling_dims: dimensions along which to split. If `int`, splits the last `N` dimensions (default: `2` for last two dimensions).
    :param None, tuple[int, ...] num_splits: number of splits along each dimension. If `None`, automatically computed.
    """

    def __init__(
        self,
        img_size: Sequence[int],
        tiling_dims: int | tuple[int, ...] = 2,
        num_splits: tuple[int, ...] | None = None,
        **kwargs,
    ):
        r"""
        Initialize basic strategy.
        """
        super().__init__(img_size)

        # Normalize tiling_dims to tuple
        if isinstance(tiling_dims, int):
            # If tiling_dims is an int, interpret it as "split the last N dimensions"
            n = tiling_dims
            self.tiling_dims = tuple(range(-n, 0))
        elif isinstance(tiling_dims, tuple):
            self.tiling_dims = tiling_dims
        else:
            raise ValueError("tiling_dims must be an int or a tuple of ints")

        # Compute splits
        if num_splits is None:
            # Default: split into roughly square patches
            total_size = 1
            for dim in self.tiling_dims:
                total_size *= img_size[dim]
            target_patch_size = max(
                64, int(total_size ** (1 / len(self.tiling_dims)) / 2)
            )
            num_splits = tuple(
                max(1, img_size[dim] // target_patch_size) for dim in self.tiling_dims
            )

        self.num_splits_per_dim = num_splits
        self._compute_splits()

    def _compute_splits(self):
        """Compute all patch slices."""
        self._patch_slices = []
        self._patch_positions = []

        # Generate all combinations of splits
        ranges = []
        for i, dim in enumerate(self.tiling_dims):
            size = self.img_size[dim]
            n_splits = self.num_splits_per_dim[i]
            split_size = size // n_splits
            remainder = size % n_splits

            dim_ranges = []
            start = 0
            for j in range(n_splits):
                # Distribute remainder across first few splits
                current_size = split_size + (1 if j < remainder else 0)
                dim_ranges.append((start, start + current_size))
                start += current_size
            ranges.append(dim_ranges)

        # Generate all patch combinations
        import itertools

        for positions in itertools.product(*[range(len(r)) for r in ranges]):
            # Create slice tuple
            slices = [slice(None)] * len(self.img_size)
            for i, (dim, pos) in enumerate(zip(self.tiling_dims, positions)):
                start, end = ranges[i][pos]
                slices[dim] = slice(start, end)

            self._patch_slices.append(tuple(slices))
            self._patch_positions.append(positions)

    def get_local_patches(
        self, x: torch.Tensor, local_indices: list[int]
    ) -> list[tuple[int, torch.Tensor]]:
        r"""Extract local patches without any special processing."""
        patches = []
        for idx in local_indices:
            patch = x[self._patch_slices[idx]].clone()
            patches.append((idx, patch))
        return patches

    def reduce_patches(
        self, out_tensor: torch.Tensor, local_pairs: list[tuple[int, torch.Tensor]]
    ) -> None:
        r"""Simple assignment of patches to output tensor."""
        for idx, patch in local_pairs:
            out_tensor[self._patch_slices[idx]] = patch

    def get_num_patches(self) -> int:
        r"""Return total number of patches."""
        return len(self._patch_slices)


class SmartTilingStrategy(DistributedSignalStrategy):
    r"""
    Smart tiling strategy with padding for N-dimensional data.

    This strategy:
        - Creates uniform patches with receptive field padding
        - Batches patches for efficient processing
        - Uses optimized tensor operations for reduction

    :param Sequence[int] img_size: shape of the complete signal tensor.
    :param int | tuple[int, ...] | None tiling_dims: dimensions to tile.
    :param int | tuple[int, ...] patch_size: size of each patch, supports non-cuboid patch size.
    :param int | tuple[int, ...] receptive_field_size: padding radius around each patch, supports non-cuboid receptive field size.
    :param int | tuple[int, ...] | None stride: stride between patches. Default to the same value as `patch_size` for non-overlapping patches.
    :param str pad_mode: padding mode for edge patches.
    """

    def __init__(
        self,
        img_size: Sequence[int],
        tiling_dims: int | tuple[int, ...] | None = None,
        patch_size: int | tuple[int, ...] = 256,
        receptive_field_size: int | tuple[int, ...] = 32,
        stride: int | tuple[int, ...] | None = None,
        pad_mode: str = "reflect",
        **kwargs,
    ):
        super().__init__(img_size)
        self.patch_size = patch_size
        self.receptive_field_size = receptive_field_size
        self.stride = stride
        self.pad_mode = pad_mode
        self.tiling_dims = tiling_dims

        if self.tiling_dims is None and isinstance(patch_size, tuple):
            n = len(patch_size)
            self.tiling_dims = tuple(range(-n, 0))
        elif isinstance(self.tiling_dims, int):
            # If tiling_dims is an int, interpret it as "tile the last N dimensions"
            n = self.tiling_dims
            self.tiling_dims = tuple(range(-n, 0))
        elif not isinstance(self.tiling_dims, tuple):
            raise ValueError("tiling_dims must be an int or a tuple of ints")

        self._compute_tiling()

    def _compute_tiling(self):
        """Compute tiling layout using existing utils."""

        # At this point, tiling_dims is always a tuple (ensured by __init__)
        assert isinstance(
            self.tiling_dims, tuple
        ), "tiling_dims must be a tuple at this point"

        # Normalize patch_size and receptive_field_size to tuples
        ndim_tiled = len(self.tiling_dims)

        def to_tuple(val):
            if isinstance(val, int):
                return (val,) * ndim_tiled
            return val

        p_sizes = to_tuple(self.patch_size)
        rf_sizes = to_tuple(self.receptive_field_size)

        # Check dimensions
        shape = self.img_size

        # We might need to adjust patch sizes if they are too big
        new_p_sizes = list(p_sizes)
        new_rf_sizes = list(rf_sizes)
        modified = False

        for i, dim_idx in enumerate(self.tiling_dims):
            if dim_idx < 0:
                dim_idx += len(shape)
            D = shape[dim_idx]
            p = p_sizes[i]
            rf = rf_sizes[i]

            if p > D:
                safe_p = D - 2 * rf
                if safe_p <= 0:
                    safe_p = D
                    safe_rf = max(0, D // 8)
                    new_rf_sizes[i] = safe_rf
                else:
                    safe_rf = rf

                new_p_sizes[i] = safe_p
                modified = True

                if shape[0] == 1:  # Warning
                    print(
                        f"Warning: patch_size[{i}] ({p}) > dim {dim_idx} ({D}). Adjusted to {safe_p}, rf {safe_rf}"
                    )

        if modified:
            self.patch_size = (
                tuple(new_p_sizes)
                if isinstance(self.patch_size, tuple)
                else new_p_sizes[0]
            )
            self.receptive_field_size = (
                tuple(new_rf_sizes)
                if isinstance(self.receptive_field_size, tuple)
                else new_rf_sizes[0]
            )

        self._global_slices, self._metadata = tiling_splitting_strategy(
            self.img_size,
            patch_size=self.patch_size,
            receptive_field_size=self.receptive_field_size,
            stride=self.stride,
            tiling_dims=self.tiling_dims,
            pad_mode=self.pad_mode,
        )

    def get_local_patches(
        self, x: torch.Tensor, local_indices: list[int]
    ) -> list[tuple[int, torch.Tensor]]:
        r"""Extract and pad local patches."""

        # Apply global padding
        pad_specs = self._metadata.get("global_padding")
        pad_mode = self._metadata.get("pad_mode", self.pad_mode)

        if pad_specs and any(p > 0 for p in pad_specs):
            # Trim trailing zeros from pad_specs to avoid F.pad issues with reflect mode
            # pad_specs is (last_left, last_right, 2nd_last_left, ...)
            pads = list(pad_specs)
            while len(pads) >= 2 and pads[-1] == 0 and pads[-2] == 0:
                pads.pop()
                pads.pop()
            trimmed_pads = tuple(pads)

            try:
                x_pad = torch.nn.functional.pad(x, trimmed_pads, mode=pad_mode)
            except Exception:
                # Fallback to constant padding if reflect fails
                x_pad = torch.nn.functional.pad(x, pad_specs, mode="constant", value=0)
        else:
            x_pad = x

        patches = []
        for idx in local_indices:
            slc = self._global_slices[idx]
            patch = x_pad[slc]
            patches.append((idx, patch))
        return patches

    def reduce_patches(
        self, out_tensor: torch.Tensor, local_pairs: list[tuple[int, torch.Tensor]]
    ) -> None:
        r"""Reduce patches using tiling metadata."""
        tiling_reduce_fn(out_tensor, local_pairs, self._metadata)

    def get_num_patches(self) -> int:
        r"""Return total number of patches."""
        return len(self._global_slices)


def create_strategy(
    strategy_name: str, img_size: Sequence[int], n_dimension: int, **kwargs
) -> DistributedSignalStrategy:
    r"""
    Create a distributed signal strategy by name.

    :param str strategy_name: name of the strategy (`'basic'`, `'smart_tiling'`).
    :param Sequence[int] img_size: shape of the signal tensor.
    :param int n_dimension: number of dimensions of the signal (e.g., `2` for images, `3` for volumes).
    :return: the created strategy instance.
    """
    # Handle tiling_dims priority: kwargs > n_dimension
    if "tiling_dims" in kwargs and kwargs["tiling_dims"] is not None:
        tiling_dims = kwargs.pop("tiling_dims")
    else:
        tiling_dims = n_dimension
        if "tiling_dims" in kwargs:
            kwargs.pop("tiling_dims")

    if strategy_name == "basic":
        return BasicStrategy(img_size, tiling_dims=tiling_dims, **kwargs)
    elif strategy_name == "smart_tiling":
        return SmartTilingStrategy(img_size, tiling_dims=tiling_dims, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
