# TiledMixin2d

### *class* deepinv.utils.TiledMixin2d(patch_size, stride=None, pad_if_needed=True, \*args, \*\*kwargs)

Bases: [`object`](https://docs.python.org/3.9/library/functions.html#object)

Mixin base class for 2D tiled patch extraction and reconstruction.
Provides methods to extract overlapping patches from images and reconstruct images from patches.

It also handles padding if necessary to ensure all patches have the same size.
The patch extraction and reconstruction are implemented using PyTorch’s unfold and fold operations for efficiency.

* **Parameters:**
  * **patch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – Size of each patch (height, width) or single `int` for square patches.
  * **stride** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – Stride between adjacent patches (height, width). If a single `int` is provided, it is used for both dimensions. Defaults to half the patch size.
  * **pad_if_needed** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, the image will be padded if necessary to ensure all patches have the same size. Defaults to `True`.

<hr />

The following example demonstrates how to use the `TiledMixin2d` to extract patches from an image and reconstruct the image from those patches.

* **Examples:**
  ```pycon
  >>> import torch
  >>> from deepinv.utils.mixins import TiledMixin2d
  >>> # Create an image of shape (B, C, H, W)
  >>> B, C, H, W = 1, 1, 5, 5
  >>> image = torch.arange(B * C * H * W, dtype=torch.float32).reshape(B, C, H, W)
  >>> print(image)
  tensor([[[[ 0.,  1.,  2.,  3.,  4.],
            [ 5.,  6.,  7.,  8.,  9.],
            [10., 11., 12., 13., 14.],
            [15., 16., 17., 18., 19.],
            [20., 21., 22., 23., 24.]]]])
  ```

  ```pycon
  >>> # Initialize the TiledMixin2d with patch size and stride
  >>> patch_size = (3, 3)
  >>> stride = (2, 2)
  >>> tiled_mixin = TiledMixin2d(patch_size=patch_size, stride=stride)
  ```

  ```pycon
  >>> # Extract patches from the image
  >>> patches = tiled_mixin.image_to_patches(image)
  >>> print("Extracted Patches Shape:", patches.shape)
  Extracted Patches Shape: torch.Size([1, 1, 2, 2, 3, 3])
  >>> print(patches[..., 0, 0, :, :]) # Print the first patch for verification
  tensor([[[[ 0.,  1.,  2.],
            [ 5.,  6.,  7.],
            [10., 11., 12.]]]])
  ```

  ```pycon
  >>> # Reconstruct the image from the patches
  >>> reconstructed_image = tiled_mixin.patches_to_image(patches, img_size=(H, W))
  >>> print("Reconstructed Image Shape:", reconstructed_image.shape)
  Reconstructed Image Shape: torch.Size([1, 1, 5, 5])
  >>> print(reconstructed_image)
  tensor([[[[ 0.,  1.,  4.,  3.,  4.],
            [ 5.,  6., 14.,  8.,  9.],
            [20., 22., 48., 26., 28.],
            [15., 16., 34., 18., 19.],
            [20., 21., 44., 23., 24.]]]])
  >>> # Note that by default, the reconstructed image is not necessarily equal to the original image due to overlapping regions being summed. Setting `reduce_overlap="mean"` in `patches_to_image` will average the overlapping regions instead of summing, which can give a closer reconstruction to the original image.
  >>> reconstructed_image_mean = tiled_mixin.patches_to_image(patches, img_size=(H, W), reduce_overlap="mean")
  >>> print(reconstructed_image_mean)
  tensor([[[[ 0.,  1.,  2.,  3.,  4.],
            [ 5.,  6.,  7.,  8.,  9.],
            [10., 11., 12., 13., 14.],
            [15., 16., 17., 18., 19.],
            [20., 21., 22., 23., 24.]]]])
  ```

#### get_compatible_img_size(img_size)

Get compatible image size for patch extraction.

* **Parameters:**
  **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – Original image size (height, width).
* **Returns:**
  Compatible image size (height, width).
* **Return type:**
  [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[int](https://docs.python.org/3.9/library/functions.html#int), [int](https://docs.python.org/3.9/library/functions.html#int)]

#### get_needed_pad(img_size)

Get required padding.

* **Parameters:**
  **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – Original image size (height, width).
* **Returns:**
  Tuple of (compatible_size, padding).
* **Return type:**
  [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[int](https://docs.python.org/3.9/library/functions.html#int), [int](https://docs.python.org/3.9/library/functions.html#int)]

#### get_num_patches(img_size)

Get number of patches along height and width.
: - If `pad_if_needed` is `True`, this will return the number of patches that can be extracted after padding the image to a compatible size.
  - If `pad_if_needed` is `False`, this will return the number of patches that can be extracted without padding, which may not cover the whole image.

* **Parameters:**
  **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – Image size (height, width).
* **Returns:**
  Number of patches (n_h, n_w).
* **Return type:**
  [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[int](https://docs.python.org/3.9/library/functions.html#int), [int](https://docs.python.org/3.9/library/functions.html#int)]

#### image_to_patches(image, pad=(0, 0, 0, 0))

Split an image into overlapping patches.

The image will be padded if necessary to ensure all patches have the same size.

* **Parameters:**
  * **image** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input image tensor of shape `(B, C, H, W)`.
  * **pad** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – Optional, if provided, the patch size will be increased by this padding on each side. Can be a single int for symmetric padding or a tuple of 4 ints for (left, right, top, bottom) padding. Defaults to `(0, 0, 0, 0)` for no additional padding. If provided, this padding is added on top of any padding that may be needed to ensure compatible patch extraction. So the effective patch size will become `patch_size + pad`.
* **Returns:**
  Patches tensor of shape `(B, C, n_rows, n_cols, patch_h, patch_w)`.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### NOTE
The `pad` argument allows you to specify additional padding to be added to the patch size. This can be useful if you want to include some context around each patch. For example, if you have a patch size of (3, 3) and you set `pad=1`, then the effective patch size will become (5, 5). This is useful when you want perform operations that require context around the patch, such as convolutional operations.

#### patches_to_image(patches, img_size=None, reduce_overlap='sum')

Reconstruct an image from overlapping patches.

This is the inverse operation of `image_to_patches`. Note that overlapping
regions are summed. So the reconstructed image is not necessarily equal to the original image.

* **Parameters:**
  * **patches** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Patches tensor of shape `(B, C, n_rows, n_cols, patch_h, patch_w)`.
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*  *|* *None*) – Target output size (height, width). If provided, output is cropped to this size from the top-left corner.
  * **reduce_overlap** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – How to handle overlapping regions. Options are `"sum"` or `"mean"`. If `"sum"`, overlapping regions are summed (default). If `"mean"`, overlapping regions are averaged, which can give an identical reconstruction to the original image.
* **Returns:**
  Reconstructed image tensor of shape `(B, C, H, W)`.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-utils-tiledmixin2d"></a>

## Examples using `TiledMixin2d`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In this example we use patch priors for limited angle computed tomography. More precisely, we consider the inverse problem y = \\mathrm{noisy}(Ax), where A is the discretized Radon transform with 100 equispace angles between 20 and 160 degrees. For the reconstruction, we minimize the variational problem">  <div class="sphx-glr-thumbnail-title">Patch priors for limited-angle computed tomography</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of 2D blur operators in DeepInverse. In particular, we show how to use DiffractionBlurs (Fresnel diffraction), motion blurs and space varying blurs.">  <div class="sphx-glr-thumbnail-title">Tour of blur operators</div>
</div>
<!-- thumbnail-parent-div-close --></div>
