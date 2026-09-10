# plot_napari

### deepinv.utils.plot_napari(\*x, screenshot=False, rescale_mode='min_max', vmin=None, vmax=None)

View 2D images or 3D volumes in napari.

Opens an interactive [napari](https://napari.org) viewer displaying the
provided tensors side by side in a grid. This is useful for visualising 3D data e.g.
volumes or stacks of images.

#### WARNING
Requires `napari` to be installed. Install it with `pip install "napari[all]"`.

#### NOTE
This function opens an interactive window and therefore requires a
display. Pass `screenshot=True` to render off-screen and return a
`PIL.Image.Image` instead.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – tensors passed as args, accepts 1 to 6.
    Each must be either a 2D image of shape `(1, 1, H, W)` or a 3D volume of
    shape `(1, 1, D, H, W)`. Batch dim and channel dim must both be 1.
    All tensors must have the same number of dimensions but can be different sizes.
  * **screenshot** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, capture a screenshot after rendering, close the viewer, and return a `PIL.Image.Image`.
  * **rescale_mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – rescale mode, either `'min_max'` (images are linearly rescaled between 0 and 1 using
    their min and max values) or `'clip'` (images are clipped between 0 and 1).
  * **vmin** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* *None*) – optional minimum value for clipping when using ‘clip’ rescaling.
  * **vmax** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* *None*) – optional maximum value for clipping when using ‘clip’ rescaling.
* **Returns:**
  `None`, or a `PIL.Image.Image` if `screenshot` is `True`.
* **Return type:**
  None

## Examples using `plot_napari`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to denoise low-intensity STED fluorescence microscopy images of live-cell mitochondria using the pretrained foundation model deepinv.models.RAM. We load real Abberior STED microscopy data from :footciteosunavargas2025denoising, process it in batches, and visualize the results both with deepinv.utils.plot and with the interactive 3D viewer deepinv.utils.plot_napari.">  <div class="sphx-glr-thumbnail-title">Low-intensity STED fluorescence microscopy denoising</div>
</div>
<!-- thumbnail-parent-div-close --></div>
