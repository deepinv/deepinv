"""
Distributed signal processing strategies for the deepinv library.

This module provides abstract base classes and concrete implementations for
distributed signal processing, including splitting, batching, and reduction operations.
"""

from abc import ABC, abstractmethod
from typing import Optional, Sequence
import torch
from .utils import extract_and_pad_patch, tiling_splitting_strategy, tiling2d_reduce_fn
Index = tuple[slice, ...]


class DistributedSignalStrategy(ABC):
    """
    Abstract base class for distributed signal processing strategies.

    A strategy defines how to:
    1. Split a signal into patches for distributed processing
    2. Batch patches for efficient processing
    3. Reduce processed patches back into a complete signal

    This allows users to implement custom distributed processing strategies
    for different types of data and use cases.
    """

    def __init__(self, signal_shape: Sequence[int], **kwargs):
        """
        Initialize the strategy.

        Parameters
        ----------
        signal_shape : Sequence[int]
            Shape of the complete signal tensor (e.g., [B, C, H, W])
        **kwargs
            Strategy-specific parameters
        """
        self.signal_shape = torch.Size(signal_shape)

    @abstractmethod
    def get_local_patches(
        self, X: torch.Tensor, local_indices: list[int]
    ) -> list[tuple[int, torch.Tensor]]:
        """
        Extract and prepare local patches for processing.

        Parameters
        ----------
        X : torch.Tensor
            The complete signal tensor
        local_indices : List[int]
            Global indices of patches assigned to this rank

        Returns
        -------
        List[Tuple[int, torch.Tensor]]
            List of (global_index, prepared_patch) pairs ready for processing
        """
        pass

    @abstractmethod
    def apply_batching(self, patches: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Group patches into batches for efficient processing.

        The batching should preserve order: when the batched tensors are processed
        and then concatenated back, they should yield patches in the same order
        as the input.

        Parameters
        ----------
        patches : List[torch.Tensor]
            List of prepared patches

        Returns
        -------
        List[torch.Tensor]
            Batched patches ready for processing. When processed results are
            concatenated, they should preserve the original patch order.
        """
        pass

    def unpack_batched_results(
        self, processed_batches: list[torch.Tensor], num_patches: int
    ) -> list[torch.Tensor]:
        """
        Unpack processed batches back to individual patches.

        Default implementation: concatenate along batch dimension and split back.
        Strategies can override this if they use different batching logic.

        Parameters
        ----------
        processed_batches : List[torch.Tensor]
            Results from processing batched patches
        num_patches : int
            Expected number of individual patches

        Returns
        -------
        List[torch.Tensor]
            Individual processed patches in original order
        """
        if len(processed_batches) == 0:
            return []

        # Default: concatenate and split back
        if len(processed_batches) == 1:
            # Single batch - split along batch dimension
            return list(torch.unbind(processed_batches[0], dim=0))
        else:
            # Multiple batches - concatenate then split
            all_batched = torch.cat(processed_batches, dim=0)
            return list(torch.unbind(all_batched, dim=0))

    @abstractmethod
    def reduce_patches(
        self, out_tensor: torch.Tensor, local_pairs: list[tuple[int, torch.Tensor]]
    ) -> None:
        """
        Reduce processed patches into the output tensor.

        This operates in-place on out_tensor, placing each processed patch
        in its correct location within the complete signal.

        Parameters
        ----------
        out_tensor : torch.Tensor
            Output tensor to fill (should be initialized to zeros)
        local_pairs : List[Tuple[int, torch.Tensor]]
            List of (global_index, processed_patch) pairs
        """
        pass

    @abstractmethod
    def get_num_patches(self) -> int:
        """
        Get the total number of patches this strategy creates.

        Returns
        -------
        int
            Total number of patches
        """
        pass


class BasicStrategy(DistributedSignalStrategy):
    """
    Basic distributed strategy with naive splitting along specified dimensions.

    This strategy:
    - Splits the signal into blocks along specified dimensions
    - Processes patches individually (no batching)
    - Uses simple tensor assignment for reduction
    """

    def __init__(
        self,
        signal_shape: Sequence[int],
        split_dims: tuple[int, ...] = (-2, -1),
        num_splits: tuple[int, ...] = None,
        **kwargs,
    ):
        """
        Initialize basic strategy.

        Parameters
        ----------
        signal_shape : Sequence[int]
            Shape of the complete signal tensor
        split_dims : Tuple[int, ...]
            Dimensions along which to split (default: last two dimensions)
        num_splits : Tuple[int, ...]
            Number of splits along each dimension. If None, automatically computed.
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
            for i, (dim, pos) in enumerate(zip(self.split_dims, positions, strict=False)):
                start, end = ranges[i][pos]
                slices[dim] = slice(start, end)

            self._patch_slices.append(tuple(slices))
            self._patch_positions.append(positions)

    def get_local_patches(
        self, X: torch.Tensor, local_indices: list[int]
    ) -> list[tuple[int, torch.Tensor]]:
        """Extract local patches without any special processing."""
        patches = []
        for idx in local_indices:
            patch = X[self._patch_slices[idx]].clone()
            patches.append((idx, patch))
        return patches

    def apply_batching(self, patches: list[torch.Tensor]) -> list[torch.Tensor]:
        """No batching - process each patch individually."""
        return patches

    def reduce_patches(
        self, out_tensor: torch.Tensor, local_pairs: list[tuple[int, torch.Tensor]]
    ) -> None:
        """Simple assignment of patches to output tensor."""
        for idx, patch in local_pairs:
            out_tensor[self._patch_slices[idx]] = patch

    def get_num_patches(self) -> int:
        """Return total number of patches."""
        return len(self._patch_slices)


class SmartTilingStrategy(DistributedSignalStrategy):
    """
    Smart 2D tiling strategy with padding and efficient batching.

    This strategy:
    - Creates uniform patches with receptive field padding
    - Batches patches for efficient processing
    - Uses optimized tensor operations for reduction
    """

    def __init__(
        self,
        signal_shape: Sequence[int],
        patch_size: int = 256,
        receptive_field_radius: int = 32,
        stride: Optional[int] = None,
        non_overlap: bool = True,
        pad_mode: str = "reflect",
        **kwargs,
    ):
        """
        Initialize smart tiling strategy.

        Parameters
        ----------
        signal_shape : Sequence[int]
            Shape of the complete signal tensor
        patch_size : int
            Size of each patch (assuming square patches)
        receptive_field_radius : int
            Padding radius around each patch
        stride : int, optional
            Stride between patches (default: patch_size for non-overlapping)
        non_overlap : bool
            Whether patches should be non-overlapping
        pad_mode : str
            Padding mode for edge patches
        """
        super().__init__(signal_shape)
        self.patch_size = patch_size
        self.receptive_field_radius = receptive_field_radius
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
            safe_patch_size = min_dim - 2 * self.receptive_field_radius
            
            if safe_patch_size <= 0:
                # If even this doesn't work, use the whole image as a single patch
                # and reduce receptive field radius
                safe_patch_size = min_dim
                safe_receptive_field = max(0, min_dim // 8)  # Use 12.5% of min dimension as padding
                
                if self.signal_shape[0] == 1:  # Only warn once per batch
                    print(f"Warning: patch_size ({self.patch_size}) >= image size ({H}x{W}). "
                          f"Using single patch mode with patch_size={safe_patch_size}, "
                          f"receptive_field_radius={safe_receptive_field}")
                
                self.patch_size = safe_patch_size
                self.receptive_field_radius = safe_receptive_field
            else:
                if self.signal_shape[0] == 1:  # Only warn once per batch
                    print(f"Warning: patch_size ({self.patch_size}) >= image size ({H}x{W}). "
                          f"Reducing patch_size to {safe_patch_size}")
                
                self.patch_size = safe_patch_size

        kwargs = {
            "patch_size": self.patch_size,
            "receptive_field_radius": self.receptive_field_radius,
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
            
            print(f"Warning: Tiling strategy failed ({e}). Using whole image as single patch.")
            
            # Create a single patch that covers the whole image
            ndim = len(self.signal_shape)
            global_slice = tuple(slice(None) if i not in self.hw_dims 
                                else slice(0, self.signal_shape[i]) 
                                for i in range(ndim))
            
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
        """Extract and pad local patches."""

        patches = []
        for idx in local_indices:
            patch = extract_and_pad_patch(X, idx, self._global_slices, self._metadata)
            patches.append((idx, patch))
        return patches

    def apply_batching(self, patches: list[torch.Tensor]) -> list[torch.Tensor]:
        """Batch patches for efficient processing."""
        if not patches:
            return []

        # Verify all patches have the same shape (they should after padding)
        expected_shape = patches[0].shape
        for i, patch in enumerate(patches):
            if patch.shape != expected_shape:
                raise RuntimeError(
                    f"Patch {i} has shape {patch.shape}, expected {expected_shape}"
                )

        # For patches with batch dimension, we need to flatten the patch dimension
        # into the batch dimension for processing, then reshape back
        # Input: list of [batch_size, C, H, W] tensors
        # Output: single [total_batch_size, C, H, W] tensor for processing
        
        # Store metadata for unpacking
        self._batching_metadata = {
            "num_patches": len(patches),
            "original_batch_size": patches[0].shape[0] if patches else 0,
            "patch_shape": expected_shape
        }
        
        # Concatenate along batch dimension
        batch = torch.cat(patches, dim=0)
        return [batch]

    def unpack_batched_results(
        self, processed_batches: list[torch.Tensor], num_patches: int
    ) -> list[torch.Tensor]:
        """
        Unpack processed batches back to individual patches.

        For SmartTilingStrategy, we expect exactly one batch that was concatenated along
        the batch dimension. We use stored metadata to split it back correctly.
        """
        if len(processed_batches) != 1:
            raise RuntimeError(
                f"SmartTilingStrategy expects exactly 1 batch result, got {len(processed_batches)}"
            )

        result_batch = processed_batches[0]
        
        # Use stored metadata if available
        if hasattr(self, '_batching_metadata'):
            metadata = self._batching_metadata
            expected_num_patches = metadata["num_patches"]
            original_batch_size = metadata["original_batch_size"]
            
            if expected_num_patches != num_patches:
                raise RuntimeError(
                    f"Metadata mismatch: expected {expected_num_patches} patches, got {num_patches}"
                )
            
            # Calculate expected total batch size
            expected_total_batch = original_batch_size * num_patches
            
            if result_batch.shape[0] != expected_total_batch:
                # Handle special cases
                if num_patches == 1:
                    return [result_batch]
                else:
                    raise RuntimeError(
                        f"Result batch size {result_batch.shape[0]} != expected {expected_total_batch}. "
                        f"Expected {num_patches} patches × {original_batch_size} batch size each."
                    )
            
            # Split back into patches
            patches = []
            for i in range(num_patches):
                start_idx = i * original_batch_size
                end_idx = (i + 1) * original_batch_size
                patch = result_batch[start_idx:end_idx]
                patches.append(patch)
            
            return patches
        else:
            # Fallback for cases without metadata (single patch, etc.)
            if num_patches == 1:
                return [result_batch]
            else:
                # Try to split evenly
                if result_batch.shape[0] % num_patches == 0:
                    patch_batch_size = result_batch.shape[0] // num_patches
                    patches = []
                    for i in range(num_patches):
                        start_idx = i * patch_batch_size
                        end_idx = (i + 1) * patch_batch_size
                        patches.append(result_batch[start_idx:end_idx])
                    return patches
                else:
                    raise RuntimeError(
                        f"Cannot split batch of size {result_batch.shape[0]} into {num_patches} patches evenly"
                    )

    def reduce_patches(
        self, out_tensor: torch.Tensor, local_pairs: list[tuple[int, torch.Tensor]]
    ) -> None:
        """Reduce patches using tiling metadata."""
        tiling2d_reduce_fn(out_tensor, local_pairs, self._metadata)

    def get_num_patches(self) -> int:
        """Return total number of patches."""
        return len(self._global_slices)


def create_strategy(
    strategy_name: str, signal_shape: Sequence[int], **kwargs
) -> DistributedSignalStrategy:
    """
    Create a distributed signal strategy by name.

    Parameters
    ----------
    strategy_name : str
        Name of the strategy ('basic', 'smart_tiling')
    signal_shape : Sequence[int]
        Shape of the signal tensor
    **kwargs
        Strategy-specific parameters

    Returns
    -------
    DistributedSignalStrategy
        The created strategy instance
    """
    if strategy_name == "basic":
        return BasicStrategy(signal_shape, **kwargs)
    elif strategy_name == "smart_tiling":
        return SmartTilingStrategy(signal_shape, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
