r"""
Distributed signal processing strategies for the deepinv library.

This module provides abstract base classes and concrete implementations for
distributed signal processing, including splitting, batching, and reduction operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import torch

from .utils import (
    extract_and_pad_patch,
    tiling_splitting_strategy,
    tiling2d_reduce_fn,
    tiling3d_splitting_strategy,
    tiling3d_reduce_fn,
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

    :param Sequence[int] signal_shape: shape of the complete signal tensor (e.g., [B, C, H, W]).
    """

    def __init__(self, signal_shape: Sequence[int], **kwargs):
        r"""
        Initialize the strategy.

        :param Sequence[int] signal_shape: shape of the complete signal tensor (e.g., [B, C, H, W]).
        """
        self.signal_shape = torch.Size(signal_shape)

    @abstractmethod
    def get_local_patches(
        self, X: torch.Tensor, local_indices: list[int]
    ) -> list[tuple[int, torch.Tensor]]:
        r"""
        Extract and prepare local patches for processing.

        :param torch.Tensor X: the complete signal tensor.
        :param list[int] local_indices: global indices of patches assigned to this rank.
        :return: (list) list of (global_index, prepared_patch) pairs ready for processing.
        """
        pass

    def apply_batching(self, patches: list[torch.Tensor], max_batch_size: Optional[int] = None) -> list[torch.Tensor]:
        r"""
        Group patches into batches for efficient processing.

        The batching should preserve order: when the batched tensors are processed
        and then concatenated back, they should yield patches in the same order
        as the input.

        :param list[torch.Tensor] patches: list of prepared patches.
        :param None, int max_batch_size: maximum number of patches per batch. If `None`, all patches are batched together. If 1, each patch is processed individually.
        :return: (list) batched patches ready for processing. When processed results are concatenated, they should preserve the original patch order.
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
        if max_batch_size is None or max_batch_size >= len(patches):
            # Concatenate all patches along batch dimension
            batch = torch.cat(patches, dim=0)
            return [batch]
        
        # Otherwise, split into multiple batches
        batches = []
        for i in range(0, len(patches), max_batch_size):
            batch_patches = patches[i:i + max_batch_size]
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
        :return: (list) individual processed patches in original order.
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

    @abstractmethod
    def reduce_patches(
        self, out_tensor: torch.Tensor, local_pairs: list[tuple[int, torch.Tensor]]
    ) -> None:
        r"""
        Reduce processed patches into the output tensor.

        This operates in-place on out_tensor, placing each processed patch
        in its correct location within the complete signal.

        :param torch.Tensor out_tensor: output tensor to fill (should be initialized to zeros).
        :param list[tuple[int, torch.Tensor]] local_pairs: list of (global_index, processed_patch) pairs.
        """
        pass

    @abstractmethod
    def get_num_patches(self) -> int:
        r"""
        Get the total number of patches this strategy creates.

        :return: (:class:`int`) total number of patches.
        """
        pass


class BasicStrategy(DistributedSignalStrategy):
    r"""
    Basic distributed strategy with naive splitting along specified dimensions.

    This strategy:
    - Splits the signal into blocks along specified dimensions
    - Processes patches individually (no batching)
    - Uses simple tensor assignment for reduction

    :param Sequence[int] signal_shape: shape of the complete signal tensor.
    :param tuple[int, ...] split_dims: dimensions along which to split (default: last two dimensions).
    :param None, tuple[int, ...] num_splits: number of splits along each dimension. If `None`, automatically computed.
    """

    def __init__(
        self,
        signal_shape: Sequence[int],
        split_dims: tuple[int, ...] = (-2, -1),
        num_splits: Optional[tuple[int, ...]] = None,
        **kwargs,
    ):
        r"""
        Initialize basic strategy.

        :param Sequence[int] signal_shape: shape of the complete signal tensor.
        :param tuple[int, ...] split_dims: dimensions along which to split (default: last two dimensions).
        :param None, tuple[int, ...] num_splits: number of splits along each dimension. If `None`, automatically computed.
        """
        super().__init__(signal_shape)
        self.split_dims = split_dims

        # Compute splits
        if num_splits is None:
            # Default: split into roughly square patches
            total_size = 1
            for dim in split_dims:
                total_size *= signal_shape[dim]
            target_patch_size = max(64, int(total_size ** (1 / len(split_dims)) / 2))
            num_splits = tuple(
                max(1, signal_shape[dim] // target_patch_size) for dim in split_dims
            )

        self.num_splits_per_dim = num_splits
        self._compute_splits()

    def _compute_splits(self):
        """Compute all patch slices."""
        self._patch_slices = []
        self._patch_positions = []

        # Generate all combinations of splits
        ranges = []
        for i, dim in enumerate(self.split_dims):
            size = self.signal_shape[dim]
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
            slices = [slice(None)] * len(self.signal_shape)
            for i, (dim, pos) in enumerate(
                zip(self.split_dims, positions, strict=False)
            ):
                start, end = ranges[i][pos]
                slices[dim] = slice(start, end)

            self._patch_slices.append(tuple(slices))
            self._patch_positions.append(positions)

    def get_local_patches(
        self, X: torch.Tensor, local_indices: list[int]
    ) -> list[tuple[int, torch.Tensor]]:
        r"""Extract local patches without any special processing."""
        patches = []
        for idx in local_indices:
            patch = X[self._patch_slices[idx]].clone()
            patches.append((idx, patch))
        return patches

    # Use base class apply_batching with max_batch_size=1 for individual processing

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
    Smart 2D tiling strategy with padding and efficient batching.

    This strategy:
    - Creates uniform patches with receptive field padding
    - Batches patches for efficient processing
    - Uses optimized tensor operations for reduction

    :param Sequence[int] signal_shape: shape of the complete signal tensor.
    :param int patch_size: size of each patch (assuming square patches).
    :param int receptive_field_size: padding radius around each patch.
    :param None, int stride: stride between patches (default: patch_size for non-overlapping).
    :param bool non_overlap: whether patches should be non-overlapping.
    :param str pad_mode: padding mode for edge patches.
    """

    def __init__(
        self,
        signal_shape: Sequence[int],
        patch_size: int = 256,
        receptive_field_size: int = 32,
        stride: Optional[int] = None,
        non_overlap: bool = True,
        pad_mode: str = "reflect",
        **kwargs,
    ):
        r"""
        Initialize smart tiling strategy.

        :param Sequence[int] signal_shape: shape of the complete signal tensor.
        :param int patch_size: size of each patch (assuming square patches).
        :param int receptive_field_radius: padding radius around each patch.
        :param None, int stride: stride between patches (default: patch_size for non-overlapping).
        :param bool non_overlap: whether patches should be non-overlapping.
        :param str pad_mode: padding mode for edge patches.
        """
        super().__init__(signal_shape)
        self.patch_size = patch_size
        self.receptive_field_size = receptive_field_size
        self.stride = stride or patch_size
        self.non_overlap = non_overlap
        self.pad_mode = pad_mode

        # Assume 2D tiling on last two dimensions
        self.hw_dims = (-2, -1)
        self._compute_tiling()

    def _compute_tiling(self):
        """Compute tiling layout using existing utils."""

        # Get image dimensions (assume 2D tiling on last two dimensions)
        H = self.signal_shape[self.hw_dims[0]]
        W = self.signal_shape[self.hw_dims[1]]

        # Check if patch size is larger than image dimensions
        if self.patch_size >= max(H, W):
            # Handle oversized patch case
            max_dim = max(H, W)
            min_dim = min(H, W)

            # Reduce patch size to fit the image with some margin for receptive field
            safe_patch_size = min_dim - 2 * self.receptive_field_size

            if safe_patch_size <= 0:
                # If even this doesn't work, use the whole image as a single patch
                # and reduce receptive field radius
                safe_patch_size = min_dim
                safe_receptive_field = max(
                    0, min_dim // 8
                )  # Use 12.5% of min dimension as padding

                if self.signal_shape[0] == 1:  # Only warn once per batch
                    print(
                        f"Warning: patch_size ({self.patch_size}) >= image size ({H}x{W}). "
                        f"Using single patch mode with patch_size={safe_patch_size}, "
                        f"receptive_field_radius={safe_receptive_field}"
                    )

                self.patch_size = safe_patch_size
                self.receptive_field_size = safe_receptive_field
            else:
                if self.signal_shape[0] == 1:  # Only warn once per batch
                    print(
                        f"Warning: patch_size ({self.patch_size}) >= image size ({H}x{W}). "
                        f"Reducing patch_size to {safe_patch_size}"
                    )

                self.patch_size = safe_patch_size

        kwargs = {
            "patch_size": self.patch_size,
            "receptive_field_size": self.receptive_field_size,
            "stride": (self.stride, self.stride) if not self.non_overlap else None,
            "hw_dims": self.hw_dims,
            "non_overlap": self.non_overlap,
            "pad_mode": self.pad_mode,
        }

        try:
            self._global_slices, self._metadata = tiling_splitting_strategy(
                self.signal_shape, **kwargs
            )
        except Exception as e:
            # Final fallback: use the whole image as a single patch with minimal padding
            H = self.signal_shape[self.hw_dims[0]]
            W = self.signal_shape[self.hw_dims[1]]

            print(
                f"Warning: Tiling strategy failed ({e}). Using whole image as single patch."
            )

            # Create a single patch that covers the whole image
            ndim = len(self.signal_shape)
            global_slice = tuple(
                slice(None) if i not in self.hw_dims else slice(0, self.signal_shape[i])
                for i in range(ndim)
            )

            self._global_slices = [global_slice]

            # Create minimal metadata for single patch
            self._metadata = {
                "pad_specs": [(0, 0, 0, 0)],  # No padding
                "crop_slices": [tuple(slice(None) for _ in range(ndim))],
                "target_slices": [global_slice],
                "window_shape": self.signal_shape,
                "original_shape": self.signal_shape,
            }

    def get_local_patches(
        self, X: torch.Tensor, local_indices: list[int]
    ) -> list[tuple[int, torch.Tensor]]:
        r"""Extract and pad local patches."""

        patches = []
        for idx in local_indices:
            patch = extract_and_pad_patch(X, idx, self._global_slices, self._metadata)
            patches.append((idx, patch))
        return patches

    # Use base class apply_batching and unpack_batched_results implementations

    def reduce_patches(
        self, out_tensor: torch.Tensor, local_pairs: list[tuple[int, torch.Tensor]]
    ) -> None:
        r"""Reduce patches using tiling metadata."""
        tiling2d_reduce_fn(out_tensor, local_pairs, self._metadata)

    def get_num_patches(self) -> int:
        r"""Return total number of patches."""
        return len(self._global_slices)


class SmartTiling3DStrategy(DistributedSignalStrategy):
    r"""
    Smart 3D tiling strategy with padding and efficient batching for volumetric data.

    This strategy:
    - Creates uniform cubic patches with receptive field padding
    - Batches patches for efficient processing
    - Uses optimized tensor operations for reduction

    Designed for 3D volumetric data with shape (B, C, D, H, W).

    :param Sequence[int] signal_shape: shape of the complete signal tensor (e.g., [B, C, D, H, W]).
    :param int patch_size: size of each cubic patch (assuming cubic patches).
    :param int receptive_field_size: padding radius around each patch.
    :param None, int stride: stride between patches (default: patch_size for non-overlapping).
    :param bool non_overlap: whether patches should be non-overlapping.
    :param str pad_mode: padding mode for edge patches.

    |sep|

    :Examples:

        Create and use a 3D tiling strategy:

        >>> from deepinv.distrib.distribution_strategies.strategies import SmartTiling3DStrategy
        >>> signal_shape = (1, 1, 64, 64, 64)
        >>> strategy = SmartTiling3DStrategy(
        ...     signal_shape,
        ...     patch_size=32,
        ...     receptive_field_size=8
        ... )
        >>> X = torch.randn(*signal_shape)
        >>> patches = strategy.get_local_patches(X, [0, 1])
        >>> print(f"Number of patches: {strategy.get_num_patches()}")
    """

    def __init__(
        self,
        signal_shape: Sequence[int],
        patch_size: int = 64,
        receptive_field_size: int = 8,
        stride: Optional[int] = None,
        non_overlap: bool = True,
        pad_mode: str = "reflect",
        **kwargs,
    ):
        r"""
        Initialize smart 3D tiling strategy.

        :param Sequence[int] signal_shape: shape of the complete signal tensor (e.g., [B, C, D, H, W]).
        :param int patch_size: size of each cubic patch (assuming cubic patches).
        :param int receptive_field_size: padding radius around each patch.
        :param None, int stride: stride between patches (default: patch_size for non-overlapping).
        :param bool non_overlap: whether patches should be non-overlapping.
        :param str pad_mode: padding mode for edge patches.
        """
        super().__init__(signal_shape)
        self.patch_size = patch_size
        self.receptive_field_size = receptive_field_size
        self.stride = stride or patch_size
        self.non_overlap = non_overlap
        self.pad_mode = pad_mode

        # Assume 3D tiling on last three dimensions (D, H, W)
        self.dhw_dims = (-3, -2, -1)
        self._compute_tiling()

    def _compute_tiling(self):
        """Compute 3D tiling layout using existing utils."""

        # Get volume dimensions (assume 3D tiling on last three dimensions)
        D = self.signal_shape[self.dhw_dims[0]]
        H = self.signal_shape[self.dhw_dims[1]]
        W = self.signal_shape[self.dhw_dims[2]]

        # Check if patch size is larger than volume dimensions
        max_dim = max(D, H, W)
        min_dim = min(D, H, W)

        if self.patch_size >= max_dim:
            # Handle oversized patch case
            safe_patch_size = min_dim - 2 * self.receptive_field_size

            if safe_patch_size <= 0:
                # Use the whole volume as a single patch
                safe_patch_size = min_dim
                safe_receptive_field = max(0, min_dim // 8)

                if self.signal_shape[0] == 1:  # Only warn once per batch
                    print(
                        f"Warning: patch_size ({self.patch_size}) >= volume size ({D}x{H}x{W}). "
                        f"Using single patch mode with patch_size={safe_patch_size}, "
                        f"receptive_field_radius={safe_receptive_field}"
                    )

                self.patch_size = safe_patch_size
                self.receptive_field_size = safe_receptive_field
            else:
                if self.signal_shape[0] == 1:  # Only warn once per batch
                    print(
                        f"Warning: patch_size ({self.patch_size}) >= volume size ({D}x{H}x{W}). "
                        f"Reducing patch_size to {safe_patch_size}"
                    )

                self.patch_size = safe_patch_size

        kwargs = {
            "patch_size": self.patch_size,
            "receptive_field_radius": self.receptive_field_size,
            "stride": (self.stride, self.stride, self.stride)
            if not self.non_overlap
            else None,
            "dhw_dims": self.dhw_dims,
            "non_overlap": self.non_overlap,
            "pad_mode": self.pad_mode,
        }

        try:
            self._global_slices, self._metadata = tiling3d_splitting_strategy(
                self.signal_shape, **kwargs
            )
        except Exception as e:
            # Final fallback: use the whole volume as a single patch
            D = self.signal_shape[self.dhw_dims[0]]
            H = self.signal_shape[self.dhw_dims[1]]
            W = self.signal_shape[self.dhw_dims[2]]

            print(
                f"Warning: 3D tiling strategy failed ({e}). Using whole volume as single patch."
            )

            # Create a single patch that covers the whole volume
            ndim = len(self.signal_shape)
            global_slice = tuple(
                slice(None)
                if i not in self.dhw_dims
                else slice(0, self.signal_shape[i])
                for i in range(ndim)
            )

            self._global_slices = [global_slice]

            # Create minimal metadata for single patch
            self._metadata = {
                "pad_specs": [(0, 0, 0, 0, 0, 0)],  # No padding for 3D
                "crop_slices": [tuple(slice(None) for _ in range(ndim))],
                "target_slices": [global_slice],
                "window_shape": self.signal_shape,
                "original_shape": self.signal_shape,
            }

    def get_local_patches(
        self, X: torch.Tensor, local_indices: list[int]
    ) -> list[tuple[int, torch.Tensor]]:
        r"""Extract and pad local 3D patches."""

        patches = []
        for idx in local_indices:
            patch = extract_and_pad_patch(X, idx, self._global_slices, self._metadata)
            patches.append((idx, patch))
        return patches

    # Use base class apply_batching and unpack_batched_results implementations

    def reduce_patches(
        self, out_tensor: torch.Tensor, local_pairs: list[tuple[int, torch.Tensor]]
    ) -> None:
        r"""Reduce 3D patches using tiling metadata."""
        tiling3d_reduce_fn(out_tensor, local_pairs, self._metadata)

    def get_num_patches(self) -> int:
        r"""Return total number of 3D patches."""
        return len(self._global_slices)


def create_strategy(
    strategy_name: str, signal_shape: Sequence[int], **kwargs
) -> DistributedSignalStrategy:
    r"""
    Create a distributed signal strategy by name.

    :param str strategy_name: name of the strategy (`'basic'`, `'smart_tiling'`, `'smart_tiling_3d'`).
    :param Sequence[int] signal_shape: shape of the signal tensor.
    :return: (:class:`DistributedSignalStrategy`) the created strategy instance.
    """
    if strategy_name == "basic":
        return BasicStrategy(signal_shape, **kwargs)
    elif strategy_name == "smart_tiling":
        return SmartTilingStrategy(signal_shape, **kwargs)
    elif strategy_name == "smart_tiling_3d":
        return SmartTiling3DStrategy(signal_shape, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
