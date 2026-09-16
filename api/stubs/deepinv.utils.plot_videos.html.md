# plot_videos

### deepinv.utils.plot_videos(vid_list, titles=None, time_dim=2, rescale_mode='min_max', display=False, figsize=None, dpi=None, save_fn=None, return_anim=False, anim_writer=None, anim_kwargs=MappingProxyType({}), \*\*plot_kwargs)

Plots and animates a list of image sequences.

Plots videos as sequence of side-by-side frames, and saves animation (e.g. GIF) or displays as interactive HTML in notebook.
This is useful for e.g. time-varying inverse problems. Individual frames are plotted with [`deepinv.utils.plot()`](https://deepinv.org/api/stubs/deepinv.utils.plot.html.md#deepinv.utils.plot)

Plots videos as sequence of side-by-side frames, and saves animation (e.g. GIF) or displays as interactive HTML in notebook. This is useful for e.g. time-varying inverse problems. Individual frames are plotted with [`deepinv.utils.plot()`](https://deepinv.org/api/stubs/deepinv.utils.plot.html.md#deepinv.utils.plot)
vid_list can either be a video or a list of them. A video is defined as images of shape `(B,C,H,W)` augmented with a time dimension specified by `time_dim`, e.g. of shape `(B,C,T,H,W)` and `time_dim=2`. All videos must be same time-length.

Per frame of the videos, this function calls [`deepinv.utils.plot()`](https://deepinv.org/api/stubs/deepinv.utils.plot.html.md#deepinv.utils.plot), see its params to see how the frames are plotted.

To display an interactive HTML video in an IPython notebook, use `display=True`. Note IPython must be installed for this.
Per frame of the videos, this function calls [`deepinv.utils.plot()`](https://deepinv.org/api/stubs/deepinv.utils.plot.html.md#deepinv.utils.plot), see its params to see how the frames are plotted.
To display an interactive HTML video in an IPython notebook, use `display=True`. Note IPython must be installed for this.

<hr />

* **Examples:**
  Display list of image sequences live in a notebook:
  ```default
  from deepinv.utils import plot_videos
  x = torch.rand((1, 3, 5, 8, 8)) # B,C,T,H,W image sequence
  y = torch.rand((1, 3, 5, 16, 16))

  plot_videos([x, y], display=True) # Display interactive view in notebook (requires IPython)
  plot_videos([x, y], save_fn="vid.gif") # Save video as GIF
  ```
* **Parameters:**
  * **vid_list** (*Union* *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]* *]*) – video or list of videos as defined above.
  * **titles** (*Union* *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *]* *]*) – titles of images in frame, defaults to `None`.
  * **time_dim** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – time dimension of the videos. All videos should have same length in this dimension, or length 1.
    After indexing this dimension, the resulting images should be of shape `(B,C,H,W)`. Defaults to 2.
  * **rescale_mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – rescaling mode for [`deepinv.utils.plot()`](https://deepinv.org/api/stubs/deepinv.utils.plot.html.md#deepinv.utils.plot), defaults to `"min_max"`
  * **vid_list** – video or list of videos as defined above
  * **titles** – titles of images in frame, defaults to `None`
  * **time_dim** – time dimension of the videos. All videos should have same length in this dimension, or length 1. After indexing this dimension, the resulting images should be of shape `(B,C,H,W)`. Defaults to 2
  * **rescale_mode** – rescaling mode for [`deepinv.utils.plot()`](https://deepinv.org/api/stubs/deepinv.utils.plot.html.md#deepinv.utils.plot), defaults to `"min_max"`
  * **display** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – display an interactive HTML video in an IPython notebook, defaults to False
  * **figsize** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]* *,* *None*) – size of the figure. If `None`, calculated from size of img list.
  * **save_fn** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – if not `None`, save the animation to this filename.
    File extension must be provided, note `anim_writer` might have to be specified. Defaults to `None`
  * **save_fn** – if not `None`, save the animation to this filename. File extension must be provided, note `anim_writer` might have to be specified. Defaults to `None`
  * **anim_writer** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – animation writer, see [https://matplotlib.org/stable/users/explain/animations/animations.html#animation-writers](https://matplotlib.org/stable/users/explain/animations/animations.html#animation-writers), defaults to `None`
  * **return_anim** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – return matplotlib animation object, defaults to `False`
  * **dpi** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – DPI of saved videos.
  * **anim_kwargs** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – keyword args for matplotlib FuncAnimation init
  * **plot_kwargs** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – kwargs to pass to [`deepinv.utils.plot()`](https://deepinv.org/api/stubs/deepinv.utils.plot.html.md#deepinv.utils.plot)

## Examples using `plot_videos`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.sampling.PosteriorDiffusion to perform posterior sampling. It also can be used to perform unconditional image generation with arbitrary denoisers, if the data fidelity term is not specified.">  <div class="sphx-glr-thumbnail-title">Building your diffusion posterior sampling method using SDEs</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we will go over the steps in the Diffusion Posterior Sampling (DPS) algorithm introduced in :footcitechung2022diffusion. The full algorithm is implemented in deepinv.sampling.DPS.">  <div class="sphx-glr-thumbnail-title">DPS -- Posterior Sampling with Diffusion Models</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the self-supervised Artifact2Artifact loss for solving an undersampled sequential MRI reconstruction problem without ground truth.">  <div class="sphx-glr-thumbnail-title">Self-supervised MRI reconstruction with Artifact2Artifact</div>
</div>
<!-- thumbnail-parent-div-close --></div>
