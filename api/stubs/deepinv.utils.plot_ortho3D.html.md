# plot_ortho3D

### deepinv.utils.plot_ortho3D(img_list, titles=None, save_dir=None, tight=True, max_imgs=4, rescale_mode='min_max', show=True, return_fig=False, figsize=None, suptitle=None, cmap='gray', fontsize=None, interpolation='nearest')

Plots an orthogonal view of 3D images.

The images should be of shape [B, C, D, H, W] or [C, D, H, W], where B is the batch size, C is the number of channels,
D is the depth, H is the height and W is the width. The images are plotted in a grid, where the number of rows is B
and the number of columns is the length of the list. If the B is bigger than max_imgs, only the first
batches are plotted.

#### WARNING
If the number of channels is 2, the magnitude of the complex images is plotted.
If the number of channels is bigger than 3, only the first 3 channels are plotted.

Example usage:

```pycon
import torch
from deepinv.utils import plot_ortho3D
img = torch.rand(2, 3, 8, 16, 16)
plot_ortho3D(img)
```

* **Parameters:**
  * **img_list** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – list of images to plot or single image.
  * **titles** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *]*) – list of titles for each image, has to be same length as img_list.
  * **save_dir** (*None* *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – path to save the plot.
  * **tight** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – use tight layout.
  * **max_imgs** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of images to plot.
  * **rescale_mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – rescale mode, either ‘min_max’ (images are linearly rescaled between 0 and 1 using their min and max values) or ‘clip’ (images are clipped between 0 and 1).
  * **show** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – show the image plot.
  * **return_fig** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – return the figure object.
  * **figsize** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – size of the figure.
  * **suptitle** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – title of the figure.
  * **cmap** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – colormap to use for the images. Default: gray
  * **fontsize** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – fontsize for the titles. Default: 17
  * **interpolation** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – interpolation to use for the images. See [https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html](https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html) for more details. Default: none

## Examples using `plot_ortho3D`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of 3D blur operators in the library. In particular, we show how to use Diffraction Blurs (Fresnel diffraction) to simulate fluorescence microscopes.">  <div class="sphx-glr-thumbnail-title">3D diffraction PSF</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div>
<!-- thumbnail-parent-div-close --></div>
