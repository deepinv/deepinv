# load_nifti

### deepinv.utils.load_nifti(fname, as_memmap=False, dtype=np.float32, \*\*kwargs)

Load volume from nifti file as torch tensor.

We assume that the data contains a channel dimension. If not, unsqueeze the output to
add a channel dimension `x = load_nifti(...).unsqueeze(0)`.

#### WARNING
When loading zipped nifti files (e.g., .nii.gz), it is recommended to install indexed_gzip (`pip install indexed-gzip`) to speed up loading times.

<!-- warning:

Set the `dtype` correctly to load double or complex data.
You can also inspect the `nibabel` image object headers (result of `nib.load`) prior to calling `get_fdata` or `dataobj`,
to get the intended `dtype` and other metadata. -->
* **Parameters:**
  * **fname** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – file to load.
  * **as_memmap** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool) *,*) – open this file as a proxy array, which does not eagerly load the entire array into memory. This is useful when extracting patches from large arrays or to quickly infer dtype and shape.
  * **dtype** ([*numpy.dtype*](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – data type to use when loading the nifti file. This is ignored if `as_memmap` is `True`.
* **Returns:**
  [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) containing nifti image. If `as_memmap` is `True`, returns a proxy array instead.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | nib.arrayproxy.ArrayProxy

## Examples using `load_nifti`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various input/output functions provided by DeepInverse for handling medical and scientific imaging formats. We demonstrate loading and plotting from DICOM, NIfTI, ISMRMRD, PyTorch, NumPy and raster data sources.">  <div class="sphx-glr-thumbnail-title">Loading scientific images</div>
</div>
<!-- thumbnail-parent-div-close --></div>
