# load_tiff

### deepinv.utils.load_tiff(fname, dtype=None)

Load image or volume from a TIFF file as a torch tensor.

Integer images are normalized to the range `[0, 1]` by dividing by the
maximum value representable by their dtype; floating point images are
cast as-is (values are preserved, but the dtype is recast to float64).
2D images of shape `(H, W)` are loaded with a single channel,
and 3D arrays of shape `(H, W, C)` are converted to channel-first `(C, H, W)`.
In both cases a leading batch dimension is added.

#### WARNING
Requires `tifffile` to be installed. Install it with `pip install tifffile`.

* **Parameters:**
  * **fname** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – path to TIFF file or buffer.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – if not `None`, cast the output tensor to this dtype. If `None`
    (default), the tensor is returned as `torch.float64` regardless of the TIFF’s
    original dtype (e.g. a `float32` TIFF is upcast); pass `dtype=torch.float32`
    to match the dtype commonly used in the library and PyTorch.
* **Returns:**
  [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) of shape `(1, C, H, W)` and dtype `torch.float64` unless
  `dtype` is specified.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

## Examples using `load_tiff`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to denoise low-intensity STED fluorescence microscopy images of live-cell mitochondria using the pretrained foundation model deepinv.models.RAM. We load real Abberior STED microscopy data from osunavargas2025denoising, process it in batches, and visualize the results both with deepinv.utils.plot and with the interactive 3D viewer deepinv.utils.plot_napari.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_microscopy_denoising_thumb.png)

[Low-intensity STED fluorescence microscopy denoising](https://deepinv.org/auto_examples/external-libraries/demo_microscopy_denoising.md)

  <div class="sphx-glr-thumbnail-title">Low-intensity STED fluorescence microscopy denoising</div>
</div>
<!-- thumbnail-parent-div-close --></div>
