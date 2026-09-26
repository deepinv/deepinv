# TilingStrategy

### *class* deepinv.distributed.strategies.distributed_strategies.TilingStrategy(img_size, tiling_dims=None, patch_size=256, overlap=32, stride=None, pad_mode='reflect', \*\*kwargs)

Bases: [`DistributedSignalStrategy`](https://deepinv.org/api/stubs/deepinv.distributed.strategies.DistributedSignalStrategy.html.md#deepinv.distributed.strategies.DistributedSignalStrategy)

Smart tiling strategy with padding for N-dimensional data.

This strategy:
: - Creates uniform patches with receptive field padding
  - Batches patches for efficient processing
  - Uses optimized tensor operations for reduction

* **Parameters:**
  * **img_size** (*Sequence* *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – full shape of the signal tensor, including batch and channel dimensions (e.g., `(B, C, H, W)`).
  * **tiling_dims** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,*  *...* *]*  *|* *None*) – dimensions to tile.
    -   If `None`, defaults to last N dimensions where N is `len(patch_size)` if `patch_size` is `tuple`, else `2`.
    -   If `int`, tiles only that dimension.
    -   If `tuple`, tiles specified dimensions.
  * **patch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,*  *...* *]*) – size of each patch, supports non-cuboid patch size.
  * **overlap** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,*  *...* *]*) – padding radius around each patch, supports non-cuboid receptive field size.
  * **stride** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,*  *...* *]*  *|* *None*) – stride between patches. Default to the same value as `patch_size` for non-overlapping patches.
  * **pad_mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – padding mode for edge patches.

#### get_local_patches(x, local_indices)

Extract and pad local patches.

#### get_num_patches()

Return total number of patches.

#### reduce_patches(out_tensor, local_pairs)

Reduce patches using tiling metadata.
