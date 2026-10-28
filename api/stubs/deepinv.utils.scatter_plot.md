# scatter_plot

### deepinv.utils.scatter_plot(xy_list, titles=None, save_dir=None, tight=True, show=True, return_fig=False, figsize=None, subtitles=None, suptitle=None, cmap='gray', fontsize=None, s=0.1, linewidths=1.5, color='b')

Plots a list of scatter plots.

Example usage:

```pycon
import torch
from deepinv.utils import scatter_plot
xy = torch.randn(10, 2)
scatter_plot([xy, xy], titles=["scatter1", "scatter2"], save_dir="test.png")
```

* **Parameters:**
  * **xy_list** ([*list*](https://docs.python.org/3.12/library/stdtypes.html#list) *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – list of scatter plots data, or single scatter plot data.
  * **subtitles** ([*list*](https://docs.python.org/3.12/library/stdtypes.html#list) *[*[*list*](https://docs.python.org/3.12/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3.12/library/stdtypes.html#str) *]* *]* *,* [*list*](https://docs.python.org/3.12/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3.12/library/stdtypes.html#str) *]* *,* [*str*](https://docs.python.org/3.12/library/stdtypes.html#str) *,* *None*) – list of subtitles for each image, can be either the same length or the same shape as img_list.
  * **titles** ([*list*](https://docs.python.org/3.12/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3.12/library/stdtypes.html#str) *]*) – list of titles for each image, has to be same length as img_list.
  * **save_dir** (*None* *,* [*str*](https://docs.python.org/3.12/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.12/library/pathlib.html#pathlib.Path)) – path to save the plot.
  * **tight** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – use tight layout.
  * **show** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – show the image plot.
  * **return_fig** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – return the figure object.
  * **figsize** ([*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.12/library/functions.html#int) *]*) – size of the figure.
  * **suptitle** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str)) – title of the figure.
  * **cmap** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str)) – colormap to use for the images. Default: gray
  * **fontsize** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – fontsize for the plot. Default: 17
  * **s** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – size of the scattered points. Default: 0.1
  * **linewidths** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – width of the lines. Default: 1.5
  * **color** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str)) – color of the points. Default: blue

## Examples using `scatter_plot`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In this example, we investigate a simple 2D Radio Interferometry (RI) imaging task with deepinverse. The following example and data are taken from aghabiglou2024r2d2. If you are interested in RI imaging problem and would like to see more examples or try the state-of-the-art algorithms, please check BASPLib.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_ri_basic_thumb.png)

[Radio interferometric imaging with deepinverse](https://deepinv.org/auto_examples/external-libraries/demo_ri_basic.md)

  <div class="sphx-glr-thumbnail-title">Radio interferometric imaging with deepinverse</div>
</div>
<!-- thumbnail-parent-div-close --></div>
