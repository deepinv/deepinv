# DistributedSignalStrategy

### *class* deepinv.distributed.strategies.DistributedSignalStrategy(img_size, \*\*kwargs)

Bases: [`ABC`](https://docs.python.org/3.9/library/abc.html#abc.ABC)

Abstract base class for distributed signal processing strategies.

A strategy defines how to:
: 1. Split a signal into patches for distributed processing
  2. Batch patches for efficient processing
  3. Reduce processed patches back into a complete signal

This allows users to implement custom distributed processing strategies
for different types of data and use cases.

* **Parameters:**
  **img_size** (*Sequence* *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – shape of the complete signal tensor (e.g., `[B, C, H, W]`).

#### apply_batching(patches, max_batch_size=None)

Group patches into batches for efficient processing.

The batching should preserve order: when the batched tensors are processed
and then concatenated back, they should yield patches in the same order
as the input.

* **Parameters:**
  * **patches** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – list of prepared patches.
  * **max_batch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* *None*) – maximum number of patches per batch. If `None`, all patches are batched together. If `1`, each patch is processed individually.
* **Returns:**
  batched patches ready for processing. When processed results are concatenated, they should preserve the original patch order.
* **Return type:**
  [list](https://docs.python.org/3.9/library/stdtypes.html#list)[[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]

#### *abstractmethod* get_local_patches(x, local_indices)

Extract and prepare local patches for processing.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the complete signal tensor.
  * **local_indices** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – global indices of patches assigned to this rank.
* **Returns:**
  list of (global_index, prepared_patch) pairs ready for processing.
* **Return type:**
  [list](https://docs.python.org/3.9/library/stdtypes.html#list)[[tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[int](https://docs.python.org/3.9/library/functions.html#int), [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]]

#### *abstractmethod* get_num_patches()

Get the total number of patches this strategy creates.

* **Returns:**
  total number of patches.
* **Return type:**
  [int](https://docs.python.org/3.9/library/functions.html#int)

#### *abstractmethod* reduce_patches(out_tensor, local_pairs)

Reduce processed patches into the output tensor.

This operates in-place on `out_tensor`, placing each processed patch
in its correct location within the complete signal.

* **Parameters:**
  * **out_tensor** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – output tensor to fill (should be initialized to zeros).
  * **local_pairs** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]* *]*) – list of (global_index, processed_patch) pairs.

#### unpack_batched_results(processed_batches, num_patches)

Unpack processed batches back to individual patches.

Default implementation: concatenate along batch dimension and split back.
Uses stored metadata to determine original patch batch size.

* **Parameters:**
  * **processed_batches** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – results from processing batched patches.
  * **num_patches** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – expected number of individual patches.
* **Returns:**
  list of individual processed patches in original order.
* **Return type:**
  [list](https://docs.python.org/3.9/library/stdtypes.html#list)[[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]
