# get_subset_tensor

### deepinv.physics.functional.tomography_subsets.get_subset_tensor(tensor, num_subsets)

Return indices that interleave a tensor into equal subsets.

* **Parameters:**
  * **tensor** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – tensor containing angles or geometry vectors to split along its first dimension.
  * **num_subsets** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of subsets.
* **Returns:**
  list of index tensors.
* **Return type:**
  [list](https://docs.python.org/3.9/library/stdtypes.html#list)[[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]
