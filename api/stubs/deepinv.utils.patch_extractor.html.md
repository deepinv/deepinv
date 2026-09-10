# patch_extractor

### deepinv.utils.patch_extractor(imgs, n_patches, patch_size, duplicates=False, position_inds_linear=None)

This function takes a `B x C x H x W` tensor as input and extracts `n_patches` random patches
of size `C x patch_size x patch_size` from each `C x H x W` image.
Hence, the output is of shape `B x n_patches x C x patch_size x patch_size`.

It returns a tuple of the extracted patches and the linear indices of the patches in the original image.

* **Parameters:**
  * **imgs** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Images for cutting out patches. Shape batch size x channels x height x width
  * **patch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – size of the patches. The patches are square, so this is the height and width of the patch.
  * **n_patches** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of patches to cut out from each image. If -1, all possible patches are cut out.
  * **duplicates** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – determines if a patch can appear twice.
  * **position_inds_linear** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – allows it to cut patches with specific indices (required for the EPLL reconstruction).
    dtype of the tensor should be torch.long.
* **Returns:**
  tuple of (patches, linear_indices)
* **Return type:**
  [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]
