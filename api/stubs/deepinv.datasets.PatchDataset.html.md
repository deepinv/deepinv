# PatchDataset

### *class* deepinv.datasets.PatchDataset(imgs, patch_size=6, stride=1, transform=None, shape=(-1,), use_dict_output=False)

Bases: [`TiledMixin2d`](https://deepinv.org/api/stubs/deepinv.utils.TiledMixin2d.html.md#deepinv.utils.TiledMixin2d), [`ImageDataset`](https://deepinv.org/api/stubs/deepinv.datasets.ImageDataset.html.md#deepinv.datasets.ImageDataset)

Builds the dataset of all patches from a tensor of images.

* **Parameters:**
  * **imgs** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Tensor of images of shape `(B, C, H, W)`.
  * **patch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – size of patches to extract. If `int`, the same value is used for height and width.
  * **stride** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – stride between patches. If `int`, the same value is used for height and width.
  * **transform** (*Callable*) – data augmentation. A callable object, set to `None` for no augmentation.
  * **shape** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – shape of the returned tensor. If `None`, returns `(C, h, w)` where `h` and `w` are height and width of the patch.
    The default shape is `(-1,)` (flatten).
  * **use_dict_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return output as dict with key “x” instead of a bare tensor (default `False`).

<a id="sphx-glr-backref-deepinv-datasets-patchdataset"></a>

## Examples using `PatchDataset`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In this example we use patch priors for limited angle computed tomography. More precisely, we consider the inverse problem y = \\mathrm{noisy}(Ax), where A is the discretized Radon transform with 100 equispace angles between 20 and 160 degrees. For the reconstruction, we minimize the variational problem">  <div class="sphx-glr-thumbnail-title">Patch priors for limited-angle computed tomography</div>
</div>
<!-- thumbnail-parent-div-close --></div>
