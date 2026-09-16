# image_to_patches

### deepinv.utils.image_to_patches(image, patch_size, stride=None, pad_if_needed=True, pad=0)

Split a batch of images into overlapping 2D patches.

The behavior mirrors [`deepinv.utils.TiledMixin2d`](https://deepinv.org/api/stubs/deepinv.utils.TiledMixin2d.html.md#deepinv.utils.TiledMixin2d) while exposing a functional API.

* **Parameters:**
  * **image** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input image tensor of shape `(B, C, H, W)`.
  * **patch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – Patch size `(patch_h, patch_w)`.
    If an `int` is provided, the same value is used for both dimensions.
  * **stride** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – Stride between adjacent patches as
    `(stride_h, stride_w)`. If `None`, defaults to half the patch size.
  * **pad_if_needed** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, adds extra right/bottom padding so that the patches cover the entire image. Default is `True`.
  * **pad** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – Optional, if provided, the patch size will be increased by this padding on each side. Can be a single int for symmetric padding or a tuple of 4 ints for (left, right, top, bottom) padding. Defaults to `0` for no additional padding. If provided, this padding is added on top of any padding that may be needed to ensure compatible patch extraction. So the effective patch size will become `patch_size + pad`.
* **Returns:**
  Patches of shape `(B, C, n_rows, n_cols, patch_h, patch_w)`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### NOTE
The `pad` argument allows you to specify additional padding to be added to the patch size. This can be useful if you want to include some context around each patch. For example, if you have a patch size of (3, 3) and you set `pad=1`, then the effective patch size will become (5, 5). This is useful when you want perform operations that require context around the patch, such as convolutional operations.

#### NOTE
If `pad_if_needed` is `False` and the image size is not compatible with the patch size and stride, the patches will only cover the top-left portion of the image, and the right and bottom borders will be ignored.

<hr />

* **Examples:**

```pycon
>>> import deepinv as dinv
>>> from torchvision.utils import make_grid
>>> x = dinv.utils.load_example('butterfly.png')
>>> patches = dinv.utils.image_to_patches(x, patch_size=64, stride=32)
>>> print(f"Input shape: {x.shape}, patchified shape: {patches.shape}")
Input shape: torch.Size([1, 3, 256, 256]), patchified shape: torch.Size([1, 3, 7, 7, 64, 64])
>>> list_patch = [patches[0,:, i, j, ...] for i in range(patches.shape[2]) for j in range(patches.shape[3])]
>>> dinv.utils.plot([x, make_grid(list_patch, nrow=patches.shape[2])], titles=["Original", "Overlapping patches"])
```

![image](../../_build/plot_directive/api/stubs/deepinv-utils-image_to_patches-1.*)
