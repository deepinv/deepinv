# load_dicom

### deepinv.utils.load_dicom(fname, as_tensor=True, apply_rescale=False, dtype=np.float32, \*\*kwargs)

Load image from DICOM file.

Requires `pydicom` to be installed. Install it with `pip install pydicom`.

* **Parameters:**
  * **fname** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – path to DICOM file or buffer.
  * **as_tensor** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, return as torch tensor (default), otherwise return as numpy array.
  * **apply_rescale** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, map the stored values (SV) to output values according to the pydicom `apply_rescale`, default False.
    See [pydicom docs](https://pydicom.github.io/pydicom/3.0/tutorials/pixel_data/introduction.html)
    and [apply_rescale](https://pydicom.github.io/pydicom/2.4/reference/generated/pydicom.pixel_data_handlers.apply_rescale.html) for details.
    Note this is only useful when the appropriate dicom tags are present, for example in CT for [`deepinv.datasets.LidcIdriSliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.LidcIdriSliceDataset.html.md#deepinv.datasets.LidcIdriSliceDataset)
    for converting to Hounsfield Units. For other applications such as SUV/PET, we recommend applying the rescaling yourself.
  * **dtype** ([*numpy.dtype*](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – data type to use when loading the nifti file.
* **Returns:**
  either numpy array of shape of raw data `(...)`, or torch float tensor of shape `(1, ...)` where `...` are the DICOM image dimensions.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)

## Examples using `load_dicom`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various input/output functions provided by DeepInverse for handling medical and scientific imaging formats. We demonstrate loading and plotting from DICOM, NIfTI, ISMRMRD, PyTorch, NumPy and raster data sources.">  <div class="sphx-glr-thumbnail-title">Loading scientific images</div>
</div>
<!-- thumbnail-parent-div-close --></div>
