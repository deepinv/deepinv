# plot_inset

### deepinv.utils.plot_inset(img_list, titles=None, save_fn=None, save_dir=None, tight=True, max_imgs=4, rescale_mode='min_max', show=True, figsize=None, suptitle=None, subtitles=None, cmap='gray', fontsize=17, interpolation='none', cbar=False, dpi=1200, fig=None, axs=None, labels=(), label_loc=(0.03, 0.03), extract_loc=(0.0, 0.0), extract_size=0.2, inset_loc=(0.0, 0.5), inset_size=0.4, return_fig=False, return_axs=False)

Plots a list of images with zoomed-in insets extracted from the images.

#### Deprecated
Deprecated since version This: function is deprecated and will be removed in a future version.
Use [`plot()`](https://deepinv.org/api/stubs/deepinv.utils.plot.html.md#deepinv.utils.plot) with `plot_inset=True` instead, which has the same functionality.

The inset taken from extract_loc and shown at inset_loc. The coordinates extract_loc, inset_loc, and label_loc correspond to their top left corners taken at (horizontal, vertical) from the image’s top left.

Each loc can either be a tuple (float, float) which uses the same loc for all images across the batch dimension, or a list of these whose length must equal the batch dimension.

Coordinates are fractions from 0-1, (0, 0) is the top left corner and (1, 1) is the bottom right corner.

* **Parameters:**
  * **img_list** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]* *,* [*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – list of images, single image,
    or dict of titles: images to plot.
  * **titles** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *]* *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *None*) – list of titles for each image, has to be same length as img_list.
  * **save_fn** (*None* *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – path to save the plot as a single image (i.e. side-by-side).
  * **save_dir** (*None* *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – path to save the plots as individual images.
  * **tight** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – use tight layout.
  * **max_imgs** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of images to plot.
  * **rescale_mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – rescale mode, either `'min_max'` (images are linearly rescaled between 0 and 1 using
    their min and max values) or `'clip'` (images are clipped between 0 and 1).
  * **show** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – show the image plot. Under the hood, this calls the `plt.show()` function.
  * **figsize** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – size of the figure. If `None`, calculated from the size of `img_list`.
  * **subtitles** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *]* *]* *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *]* *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *None*) – list of subtitles for each image, can be either the same length or the same shape as img_list.
  * **suptitle** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – title of the figure.
  * **cmap** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – colormap to use for the images. Default: gray
  * **fontsize** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – fontsize for the plot. Default: 17
  * **interpolation** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – interpolation to use for the images.
    See [https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html](https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html) for more details.
    Default: none
  * **cbar** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to add colorbar to the images.
  * **dpi** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – DPI to save images.
  * **fig** (*None* *,* [*matplotlib.figure.Figure*](https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure)) – matplotlib Figure object to plot on. If None, create new Figure. Defaults to None.
  * **axs** (*None* *,* [*matplotlib.axes.Axes*](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes)) – matplotlib Axes object to plot on. If None, create new Axes. Defaults to None.
  * **labels** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *]*) – list of overlaid labels for each image, has to be same length as img_list.
  * **label_loc** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – location or locations for label to be plotted on image, defaults to (.03, .03)
  * **extract_loc** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – image location or locations for extract to be taken from, defaults to (0., 0.)
  * **extract_size** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – size of extract to be taken from image, defaults to 0.2
  * **inset_loc** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – location or locations for inset to be plotted on image, defaults to (0., 0.5)
  * **inset_size** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – size of inset to be plotted on image, defaults to 0.4
  * **return_fig** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – return the figure object.
  * **return_axs** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – return the axs object.

## Examples using `plot_inset`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example focuses on blind image Gaussian denoising, i.e. the problem">  <div class="sphx-glr-thumbnail-title">Blind denoising with noise level estimation</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example, we investigate a simple 2D Radio Interferometry (RI) imaging task with deepinverse. The following example and data are taken from :footciteaghabiglou2024r2d2. If you are interested in RI imaging problem and would like to see more examples or try the state-of-the-art algorithms, please check BASPLib.">  <div class="sphx-glr-thumbnail-title">Radio interferometric imaging with deepinverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of the denoisers in DeepInverse. A denoiser is a model that takes in a noisy image and outputs a denoised version of it. Basically, it solves the following problem:">  <div class="sphx-glr-thumbnail-title">Benchmarking pretrained denoisers</div>
</div>
<!-- thumbnail-parent-div-close --></div>
