# patches_to_image

### deepinv.utils.patches_to_image(patches, stride, img_size=None, reduce_overlap='mean')

Reconstruct images from overlapping 2D patches.

* **Parameters:**
  * **patches** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Patches tensor of shape `(B, C, n_rows, n_cols, patch_h, patch_w)`.
  * **stride** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – Stride between adjacent patches as `(stride_h, stride_w)`.
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – Optional target output image size `(height, width)`. If provided, output is cropped to this size.
  * **reduce_overlap** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – How to reduce overlapping areas, `"sum"` or `"mean"` (default).
* **Returns:**
  Reconstructed images of shape `(B, C, H, W)`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### NOTE
If `reduce_overlap` is set to `"mean"` and the patch size, stride and image size are compatible, this function is the exact inverse of [`deepinv.utils.image_to_patches()`](https://deepinv.org/api/stubs/deepinv.utils.image_to_patches.html.md#deepinv.utils.image_to_patches). If they are not compatible, an extra cropping is needed to get the exact inverse, which can be achieved by providing the `img_size` argument.
