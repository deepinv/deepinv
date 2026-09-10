# load_ismrmd

### deepinv.utils.load_ismrmd(fname, data_name='kspace', data_slice=None, \*\*kwargs)

Load complex MRI data from ISMRMD format.

Uses `h5py` to load data specified by `data_name` key. The data is assumed to be stored in complex type.

#### NOTE
To speed up loading, slice/index the data before converting to tensor.

* **Parameters:**
  * **fname** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – file to load.
  * **data_name** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – key of data in file, defaults to “kspace”.
  * **data_slice** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – slice or index to use before converting to tensor, such as `int`, `slice` or `tuple` of these.
* **Returns:**
  data loaded in [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) of shape `(2, ...)` containing real and imaginary parts,
  where `...` are dimensions of the raw data.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

## Examples using `load_ismrmd`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various input/output functions provided by DeepInverse for handling medical and scientific imaging formats. We demonstrate loading and plotting from DICOM, NIfTI, ISMRMRD, PyTorch, NumPy and raster data sources.">  <div class="sphx-glr-thumbnail-title">Loading scientific images</div>
</div>
<!-- thumbnail-parent-div-close --></div>
