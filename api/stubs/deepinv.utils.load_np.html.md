# load_np

### deepinv.utils.load_np(fname, as_memmap=False, dtype=np.float32, \*\*kwargs)

Load numpy array from file as torch tensor.

* **Parameters:**
  * **fname** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – file to load.
  * **as_memmap** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool) *,*) – open this file as a memmap, which does not load the entire array into memory. This is useful when extracting patches from large arrays or to quickly infer dtype and shape.
  * **dtype** ([*numpy.dtype*](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – data type to use when loading the numpy array. This is ignored if `as_memmap` is `True`.
* **Returns:**
  [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) containing loaded numpy array. If `as_memmap` is `True`, returns a numpy `memmap` object instead.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)

## Examples using `load_np`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various input/output functions provided by DeepInverse for handling medical and scientific imaging formats. We demonstrate loading and plotting from DICOM, NIfTI, ISMRMRD, PyTorch, NumPy and raster data sources.">  <div class="sphx-glr-thumbnail-title">Loading scientific images</div>
</div>
<!-- thumbnail-parent-div-close --></div>
