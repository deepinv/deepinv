# load_raster

### deepinv.utils.load_raster(fname, patch=False, patch_start=(0, 0), transform=None, \*\*kwargs)

Load a raster image and return patches as tensors using `rasterio`.

This function allows you to stream patches from large rasters e.g. satellite imagery, SAR etc.
and supports all file formats supported by `rasterio`.

This function requires `rasterio`, and should not rely on external GDAL dependencies. Install it with `pip install rasterio`.

* **Parameters:**
  * **fname** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Path to the raster file, such as `.geotiff`, `.tiff`, `.cos` etc., or buffer.
  * **patch** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]* *,*) – Patch extraction mode. If `False` (default), return the entire image as a
    [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) of shape `(C, H, W)` where C are bands.
    If `True`, yield patches based on the raster’s internal block windows
    (if no block windows are available, raises error; if any block has a dimension of 1 (strip layout), raise warning).
    If `int` or `(int, int)`, yield patches of the manually specified size `h, w`.
  * **patch_start** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – h and w indices from which to start taking patches. Defaults to `0,0`.
  * **transform** (*Callable* *,* *None*) – Optional transform applied to each patch.
* **Returns:**
  Either (where C is the band dimension)
  \* a full image [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) of shape `(C, H, W)`, if `patch=False`, or
  \* an iterator of torch tensors over patches of shape `(C, h, w)`, if `patch=True` or a size is specified, where `h,w` is the patch size.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [*Iterator*](https://docs.python.org/3.9/library/typing.html#typing.Iterator)[[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]

<hr />

* **Examples:**

```pycon
>>> from deepinv.utils.io import load_raster, load_url
>>> file = load_url("https://download.osgeo.org/geotiff/samples/spot/chicago/SP27GTIF.TIF")
>>> x = load_raster(file, patch=False) # Load whole image
>>> x.shape
torch.Size([1, 929, 699])
>>> x = load_raster(file, patch=True) # Patch via internal block size
>>> next(x).shape
torch.Size([1, 11, 699])
>>> all_patches = list(x) # Load all patches into memory
>>> len(all_patches)
84
>>> from torch.utils.data import DataLoader
>>> dataloader = DataLoader(all_patches, batch_size=2) # You can use this for training
>>>
>>> x = load_raster(file, patch=128, patch_start=(200, 200)) # Patch via manual size, pick away from origin
>>> next(x).shape
torch.Size([1, 128, 128])
```

## Examples using `load_raster`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various input/output functions provided by DeepInverse for handling medical and scientific imaging formats. We demonstrate loading and plotting from DICOM, NIfTI, ISMRMRD, PyTorch, NumPy and raster data sources.">  <div class="sphx-glr-thumbnail-title">Loading scientific images</div>
</div>
<!-- thumbnail-parent-div-close --></div>
