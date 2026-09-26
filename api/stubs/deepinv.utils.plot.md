# plot

### deepinv.utils.plot(img_list, titles=None, save_fn=None, save_dir=None, tight=True, max_imgs=4, rescale_mode='min_max', show=True, close=False, figsize=None, subtitles=None, suptitle=None, cmap='gray', fontsize=None, interpolation='none', cbar=False, dpi=1200, fig=None, axs=None, return_fig=False, return_axs=False, vmin=None, vmax=None, labels=None, label_loc=(0.03, 0.03), plot_inset=False, extract_loc=(0.0, 0.0), extract_size=0.2, inset_loc=(0.0, 0.5), inset_size=0.4, norm=None, \*\*imshow_kwargs)

Plots a list of images.

The images should be of shape [B,C,H,W] or [C, H, W], where B is the batch size, C is the number of channels,
H is the height and W is the width. The images are plotted in a grid, where the number of rows is B
and the number of columns is the length of the list. If the B is bigger than max_imgs, only the first
batches are plotted.

#### WARNING
If the number of channels is 2, the magnitude of the complex images is plotted.
If the number of channels is bigger than 3, only the first 3 channels are plotted.

We provide flexibility to save plots either side-by-side using `save_fn` or as individual images using `save_dir`.

#### NOTE
Zoomed-in insets can be plotted by setting `plot_inset=True`. See [`plot_inset()`](https://deepinv.org/api/stubs/deepinv.utils.plot_inset.md#deepinv.utils.plot_inset) for the original implementation.

Example usage:

```pycon
import torch
from deepinv.utils import plot
img = torch.rand(4, 3, 256, 256)
plot([img, img, img], titles=["img1", "img2", "img3"], save_dir="test.png")
```

#### NOTE
Using `show=True` calls `plt.show()` with blocking (outside notebook environments).
If this is undesired simply use `fig = plot(..., show=False, return_fig=True)`
and plot at your desired location using `fig.show()`.

* **Parameters:**
  * **img_list** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]* *,* [*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – list of images, single image,
    or dict of titles: images to plot.
  * **titles** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *]* *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *None*) – list of titles for each image, has to be same length as img_list.
  * **save_fn** (*None* *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – path to save the plot as a single image (i.e. side-by-side).
  * **save_dir** (*None* *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – path to save the plots as individual images.
  * **tight** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – use tight layout.
  * **max_imgs** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of images to plot.
  * **rescale_mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *None*) – rescale mode, either `'min_max'` (images are linearly rescaled between 0 and 1 using
    their minimum and maximum values), `'clip'` (images are clipped to `[vmin, vmax]` and then rescaled between
    0 and 1), or `None` (images are clipped to `[vmin, vmax]` without rescaling and displayed using `norm` or,
    by default, if norm is also `None`, a fixed range of `[0, 1]`).
  * **show** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – show the image plot. Under the hood, this calls the `plt.show()` function.
  * **close** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – close the image plot. Under the hood, this calls the `plt.close()` function.
  * **figsize** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – size of the figure. If `None`, calculated from the size of `img_list`.
  * **subtitles** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *]* *]* *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *]* *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *None*) – list of subtitles for each image, can be either the same length or the same shape as img_list.
  * **suptitle** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – title of the figure.
  * **cmap** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – colormap to use for the images. Default: gray
  * **interpolation** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – interpolation to use for the images.
    See [https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html](https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html) for more details.
    Default: none
  * **dpi** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – DPI to save images.
  * **fig** (*None* *,* [*matplotlib.figure.Figure*](https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure)) – matplotlib Figure object to plot on. If None, create new Figure. Defaults to None.
  * **axs** (*None* *,* [*matplotlib.axes.Axes*](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes)) – matplotlib Axes object to plot on. If None, create new Axes. Defaults to None.
  * **return_fig** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – return the figure object.
  * **return_axs** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – return the axs object.
  * **vmin** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* *None*) – minimum value for clipping when using ‘clip’ rescaling.
  * **vmax** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* *None*) – maximum value for clipping when using ‘clip’ rescaling.
  * **labels** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *]* *,* *None*) – list of overlaid labels for each image, has to be same length as img_list.
  * **label_loc** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list)) – location or locations for label to be plotted on image. Defaults to (.03, .03).
    Coordinates are fractions from 0-1, where (0, 0) is the top left corner and (1, 1) is the bottom right corner.
  * **plot_inset** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to plot zoomed-in insets. Defaults to False.
  * **extract_loc** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list)) – image location or locations for extract to be taken from. Defaults to (0., 0.).
    Coordinates are fractions from 0-1, where (0, 0) is the top left corner and (1, 1) is the bottom right corner.
  * **extract_size** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – size of extract to be taken from image. Defaults to 0.2.
  * **inset_loc** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list)) – location or locations for inset to be plotted on image. Defaults to (0., 0.5).
    Coordinates are fractions from 0-1, where (0, 0) is the top left corner and (1, 1) is the bottom right corner.
  * **inset_size** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – size of inset to be plotted on image. Defaults to 0.4.
  * **imshow_kwargs** – keyword args to pass to the matplotlib `imshow` calls. See
    [imshow docs](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html) for possible kwargs.
  * **norm** ([*matplotlib.colors.Normalize*](https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Normalize.html#matplotlib.colors.Normalize) *,* *None*) – Map the colormap from norm object passed as argument. When `rescale_mode=None` and `norm=None`, defaults to `Normalize(0.0, 1.0, clip=True)`.

## Examples using `plot`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use DeepInverse with your own dataset.">![](auto_examples/basics/images/thumb/sphx_glr_demo_custom_dataset_thumb.png)

[Bring your own dataset](https://deepinv.org/auto_examples/basics/demo_custom_dataset.md)

  <div class="sphx-glr-thumbnail-title">Bring your own dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using an iterative algorithm.">![](auto_examples/basics/images/thumb/sphx_glr_demo_custom_optim_thumb.png)

[Use iterative reconstruction algorithms](https://deepinv.org/auto_examples/basics/demo_custom_optim.md)

  <div class="sphx-glr-thumbnail-title">Use iterative reconstruction algorithms</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This examples shows you how to use DeepInverse with your own physics.">![](auto_examples/basics/images/thumb/sphx_glr_demo_custom_physics_thumb.png)

[Bring your own physics](https://deepinv.org/auto_examples/basics/demo_custom_physics.md)

  <div class="sphx-glr-thumbnail-title">Bring your own physics</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using a pretrained model in one line.">![](auto_examples/basics/images/thumb/sphx_glr_demo_pretrained_model_thumb.png)

[Use a pretrained model](https://deepinv.org/auto_examples/basics/demo_pretrained_model.md)

  <div class="sphx-glr-thumbnail-title">Use a pretrained model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to get started with DeepInverse in under 5 minutes.">![](auto_examples/basics/images/thumb/sphx_glr_demo_quickstart_thumb.png)

[5 minute quickstart tutorial](https://deepinv.org/auto_examples/basics/demo_quickstart.md)

  <div class="sphx-glr-thumbnail-title">5 minute quickstart tutorial</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates blind image deblurring using the pretrained kernel estimation network from the paper carbajal2023blind. The network estimates spatially-varying blur kernels from a blurred image, which are then used in a space-varying blur physics model to reconstruct the sharp image using a non-blind deblurring algorithm.">![](auto_examples/blind-inverse-problems/images/thumb/sphx_glr_demo_blind_deblurring_thumb.png)

[Blind deblurring with kernel estimation network](https://deepinv.org/auto_examples/blind-inverse-problems/demo_blind_deblurring.md)

  <div class="sphx-glr-thumbnail-title">Blind deblurring with kernel estimation network</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example focuses on blind image Gaussian denoising, i.e. the problem">![](auto_examples/blind-inverse-problems/images/thumb/sphx_glr_demo_blind_denoising_thumb.png)

[Blind denoising with noise level estimation](https://deepinv.org/auto_examples/blind-inverse-problems/demo_blind_denoising.md)

  <div class="sphx-glr-thumbnail-title">Blind denoising with noise level estimation</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example introduces the Richardson-Lucy algorithm for deconvolution in the blind setting, where both the underlying clean image and the blur kernel are unknown.">![](auto_examples/blind-inverse-problems/images/thumb/sphx_glr_demo_blind_richardsonlucy_thumb.png)

[Blind Richardson-Lucy deconvolution](https://deepinv.org/auto_examples/blind-inverse-problems/demo_blind_richardsonlucy.md)

  <div class="sphx-glr-thumbnail-title">Blind Richardson-Lucy deconvolution</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.physics.Physics together with automatic differentiation to estimate unknown parameters of your forward operator.">![](auto_examples/blind-inverse-problems/images/thumb/sphx_glr_demo_optimizing_physics_parameter_thumb.png)

[Calibrating physics operators](https://deepinv.org/auto_examples/blind-inverse-problems/demo_optimizing_physics_parameter.md)

  <div class="sphx-glr-thumbnail-title">Calibrating physics operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In many imaging problems, the data to be processed can be very large, making it challenging to fit the denoising process into the memory of a single device. For instance, medical imaging or satellite imagery often involves processing gigapixel images that cannot be processed as a whole.">![](auto_examples/distributed/images/thumb/sphx_glr_demo_denoiser_distributed_thumb.png)

[Distributed Denoiser with Image Tiling](https://deepinv.org/auto_examples/distributed/demo_denoiser_distributed.md)

  <div class="sphx-glr-thumbnail-title">Distributed Denoiser with Image Tiling</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Many large-scale imaging problems involve operators that can be naturally decomposed as a stack of multiple sub-operators:">![](auto_examples/distributed/images/thumb/sphx_glr_demo_physics_distributed_thumb.png)

[Distributed Physics Operators](https://deepinv.org/auto_examples/distributed/demo_physics_distributed.md)

  <div class="sphx-glr-thumbnail-title">Distributed Physics Operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Many large-scale imaging problems involve operators that can be naturally decomposed as a stack of multiple sub-operators:">![](auto_examples/distributed/images/thumb/sphx_glr_demo_pnp_distributed_thumb.png)

[Distributed Plug-and-Play (PnP) Reconstruction](https://deepinv.org/auto_examples/distributed/demo_pnp_distributed.md)

  <div class="sphx-glr-thumbnail-title">Distributed Plug-and-Play (PnP) Reconstruction</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In many large-scale imaging problems, the size of the image/volume to reconstruct is very large, making it impossible to train reconstruction networks (in this example, unfolded networks) with a single GPU. The deepinv.distributed framework enables training a model on multiple GPUs, by carefully parallelizing the data fidelity and denoising steps inside the network.">![](auto_examples/distributed/images/thumb/sphx_glr_demo_unrolled_distributed_thumb.png)

[Distributed Training of Unfolded Networks](https://deepinv.org/auto_examples/distributed/demo_unrolled_distributed.md)

  <div class="sphx-glr-thumbnail-title">Distributed Training of Unfolded Networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="The data is taken from the 2DeteCT benchmark kiss2025benchmarking and dataset kiss20232detect, which is an industrial CT dataset of various materials acquired using a proprietary scanner from CWI (i.e. sinogram-to-image). The setup is matched exactly to kiss2025benchmarking, such that you can compare DeepInverse image reconstruction methods with the values reported in kiss2025benchmarking.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_astra_2detect_thumb.png)

[Reconstruct real CT sinograms with the 2DeteCT benchmark](https://deepinv.org/auto_examples/external-libraries/demo_astra_2detect.md)

  <div class="sphx-glr-thumbnail-title">Reconstruct real CT sinograms with the 2DeteCT benchmark</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the Astra tomography toolbox with deepinv, a popular toolbox for tomography with GPU acceleration.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_astra_tomography_thumb.png)

[Low-dose CT with ASTRA backend and Total-Variation (TV) prior](https://deepinv.org/auto_examples/external-libraries/demo_astra_tomography.md)

  <div class="sphx-glr-thumbnail-title">Low-dose CT with ASTRA backend and Total-Variation (TV) prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use Spyrit linear models and measurements with DeepInverse. Here we use the HadamSplit2d linear model from Spyrit.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_connect_spyrit_thumb.png)

[Single-pixel imaging with Spyrit](https://deepinv.org/auto_examples/external-libraries/demo_connect_spyrit.md)

  <div class="sphx-glr-thumbnail-title">Single-pixel imaging with Spyrit</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to load and prepare properly a HuggingFace dataset using the datasets library.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_hf_dataset_thumb.png)

[Using HuggingFace datasets](https://deepinv.org/auto_examples/external-libraries/demo_hf_dataset.md)

  <div class="sphx-glr-thumbnail-title">Using HuggingFace datasets</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various input/output functions provided by DeepInverse for handling medical and scientific imaging formats. We demonstrate loading and plotting from DICOM, NIfTI, ISMRMRD, PyTorch, NumPy and raster data sources.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_io_thumb.png)

[Loading scientific images](https://deepinv.org/auto_examples/external-libraries/demo_io.md)

  <div class="sphx-glr-thumbnail-title">Loading scientific images</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to denoise low-intensity STED fluorescence microscopy images of live-cell mitochondria using the pretrained foundation model deepinv.models.RAM. We load real Abberior STED microscopy data from osunavargas2025denoising, process it in batches, and visualize the results both with deepinv.utils.plot and with the interactive 3D viewer deepinv.utils.plot_napari.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_microscopy_denoising_thumb.png)

[Low-intensity STED fluorescence microscopy denoising](https://deepinv.org/auto_examples/external-libraries/demo_microscopy_denoising.md)

  <div class="sphx-glr-thumbnail-title">Low-intensity STED fluorescence microscopy denoising</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs raw non-Cartesian multicoil kspace data from the FastMRI breast dataset solomonFastMRI2025, for mammography.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_mrinufft_breast_thumb.png)

[Reconstruct accelerated non-Cartesian breast MRI acquisition data](https://deepinv.org/auto_examples/external-libraries/demo_mrinufft_breast.md)

  <div class="sphx-glr-thumbnail-title">Reconstruct accelerated non-Cartesian breast MRI acquisition data</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example, we investigate a simple 2D Radio Interferometry (RI) imaging task with deepinverse. The following example and data are taken from aghabiglou2024r2d2. If you are interested in RI imaging problem and would like to see more examples or try the state-of-the-art algorithms, please check BASPLib.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_ri_basic_thumb.png)

[Radio interferometric imaging with deepinverse](https://deepinv.org/auto_examples/external-libraries/demo_ri_basic.md)

  <div class="sphx-glr-thumbnail-title">Radio interferometric imaging with deepinverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to fit deepinv.loss.metric.NIQE on a new dataset, and use it to evaluate denoiser performance.">![](auto_examples/metrics/images/thumb/sphx_glr_demo_custom_niqe_thumb.png)

[Fitting NIQE on a custom dataset](https://deepinv.org/auto_examples/metrics/demo_custom_niqe.md)

  <div class="sphx-glr-thumbnail-title">Fitting NIQE on a custom dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In blind inverse problems, some parameters of the physics are unknown at test time. Running non-blind models requires knowing these parameters, which are often hard to estimate. For example, a denoiser needs the noise level \\sigma, a deblurring model needs the blur kernel.">![](auto_examples/metrics/images/thumb/sphx_glr_demo_test_time_tuning_thumb.png)

[Blind inverse problems with no reference metrics](https://deepinv.org/auto_examples/metrics/demo_test_time_tuning.md)

  <div class="sphx-glr-thumbnail-title">Blind inverse problems with no reference metrics</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model terris2025reconstruct to solve inverse problems.">![](auto_examples/models/images/thumb/sphx_glr_demo_foundation_model_thumb.png)

[Inference and fine-tune a foundation model](https://deepinv.org/auto_examples/models/demo_foundation_model.md)

  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo reconstructs undersampled k-space data for 2D cardiac and brain MRI on:">![](auto_examples/models/images/thumb/sphx_glr_demo_mri_pretrained_thumb.png)

[Reconstruct undersampled k-space for cardiac and brain MRI](https://deepinv.org/auto_examples/models/demo_mri_pretrained.md)

  <div class="sphx-glr-thumbnail-title">Reconstruct undersampled k-space for cardiac and brain MRI</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs real prospectively undersampled multicoil brain k-space from yu2022validation.">![](auto_examples/models/images/thumb/sphx_glr_demo_prospective_mri_thumb.png)

[Reconstruct prospectively-undersampled raw multicoil MRI](https://deepinv.org/auto_examples/models/demo_prospective_mri.md)

  <div class="sphx-glr-thumbnail-title">Reconstruct prospectively-undersampled raw multicoil MRI</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Single-image super-resolution (SISR) is the inverse problem of recovering a high-resolution (HR) image x from a low-resolution (LR) observation y = \\downarrow_s(x), where \\downarrow_s denotes downsampling by factor s.">![](auto_examples/models/images/thumb/sphx_glr_demo_super_resolution_thumb.png)

[Super-resolution with SRResNet](https://deepinv.org/auto_examples/models/demo_super_resolution.md)

  <div class="sphx-glr-thumbnail-title">Super-resolution with SRResNet</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a very simple quick start introduction to training reconstruction networks with DeepInverse for solving imaging inverse problems.">![](auto_examples/models/images/thumb/sphx_glr_demo_training_thumb.png)

[Training a reconstruction model](https://deepinv.org/auto_examples/models/demo_training.md)

  <div class="sphx-glr-thumbnail-title">Training a reconstruction model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use variational 3D denoisers for denoising a 3D image. We first apply a standard soft-thresholding wavelet denoiser to a 3D brain MRI volume, as well as a 3D TV denoiser. We then extend the wavelet denoiser objective to a redundant dictionary of wavelet bases, which does not admit a closed-form solution. We solve the denoising problem using the Dykstra-like algorithm.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_3D_denoising_thumb.png)

[3D denoising of brain MRI with wavelet and TV priors](https://deepinv.org/auto_examples/optimization/demo_3D_denoising.md)

  <div class="sphx-glr-thumbnail-title">3D denoising of brain MRI with wavelet and TV priors</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard TV prior for image deblurring. The problem writes as y = Ax + \\epsilon where A is a convolutional operator and \\epsilon is the realization of some Gaussian noise. The goal is to recover the original image x from the blurred and noisy image y. The TV prior is used to regularize the problem.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_TV_minimisation_thumb.png)

[Image deblurring with Total-Variation (TV) prior](https://deepinv.org/auto_examples/optimization/demo_TV_minimisation.md)

  <div class="sphx-glr-thumbnail-title">Image deblurring with Total-Variation (TV) prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to reconstruct a noisy and incomplete image using the deep image prior.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_dip_thumb.png)

[Reconstructing an image using the deep image prior.](https://deepinv.org/auto_examples/optimization/demo_dip.md)

  <div class="sphx-glr-thumbnail-title">Reconstructing an image using the deep image prior.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we use the expected patch log likelihood (EPLL) prior zoran2011learning. for denoising and inpainting of natural images. To this end, we consider the inverse problem y = Ax+\\epsilon, where A is either the identity (for denoising) or a masking operator (for inpainting) and \\epsilon\\sim\\mathcal{N}(0,\\sigma^2 I) is white Gaussian noise with standard deviation \\sigma.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_epll_thumb.png)

[Expected Patch Log Likelihood (EPLL) for Denoising and Inpainting](https://deepinv.org/auto_examples/optimization/demo_epll.md)

  <div class="sphx-glr-thumbnail-title">Expected Patch Log Likelihood (EPLL) for Denoising and Inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a full-resolution multispectral image from raw snapshot mosaiced measurements using various reconstruction algorithms.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_multispectral_demosaicing_thumb.png)

[Multispectral demosaicing from raw sensor data](https://deepinv.org/auto_examples/optimization/demo_multispectral_demosaicing.md)

  <div class="sphx-glr-thumbnail-title">Multispectral demosaicing from raw sensor data</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we use patch priors for limited angle computed tomography. More precisely, we consider the inverse problem y = \\mathrm{noisy}(Ax), where A is the discretized Radon transform with 100 equispace angles between 20 and 160 degrees. For the reconstruction, we minimize the variational problem">![](auto_examples/optimization/images/thumb/sphx_glr_demo_patch_priors_CT_thumb.png)

[Patch priors for limited-angle computed tomography](https://deepinv.org/auto_examples/optimization/demo_patch_priors_CT.md)

  <div class="sphx-glr-thumbnail-title">Patch priors for limited-angle computed tomography</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to solve Poisson inverse problems using the Maximum-Likelihood Expectation-Maximization (MLEM) algorithm sheppMaximumLikelihoodReconstruction1982, also known as the Richardson-Lucy algorithm in the deconvolution setting richardsonBayesianBasedIterativeMethod1972,lucyIterativeTechniqueRectification1974.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_poisson_mlem_thumb.png)

[Poisson Inverse Problems with Maximum-Likelihood Expectation-Maximization (MLEM)](https://deepinv.org/auto_examples/optimization/demo_poisson_mlem.md)

  <div class="sphx-glr-thumbnail-title">Poisson Inverse Problems with Maximum-Likelihood Expectation-Maximization (MLEM)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard wavelet prior for image inpainting. The problem writes as y = Ax + \\epsilon where A is a mask and \\epsilon is the realization of some Gaussian noise. The goal is to recover the original image x from the blurred and noisy image y. The wavelet prior is used to regularize the problem.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_wavelet_prior_thumb.png)

[Image inpainting with wavelet prior](https://deepinv.org/auto_examples/optimization/demo_wavelet_prior.md)

  <div class="sphx-glr-thumbnail-title">Image inpainting with wavelet prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to denoise images corrupted by Poisson-Gaussian noise using the Generalized Anscombe Transform (GAT), which converts any Gaussian denoiser into a Poisson-Gaussian denoiser makitalo2012optimal.">![](auto_examples/physics/images/thumb/sphx_glr_demo_anscombe_thumb.png)

[Poisson-Gaussian Denoising with the Generalized Anscombe Transform](https://deepinv.org/auto_examples/physics/demo_anscombe.md)

  <div class="sphx-glr-thumbnail-title">Poisson-Gaussian Denoising with the Generalized Anscombe Transform</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of 2D blur operators in DeepInverse. In particular, we show how to use DiffractionBlurs (Fresnel diffraction), motion blurs and space varying blurs.">![](auto_examples/physics/images/thumb/sphx_glr_demo_blur_tour_thumb.png)

[Tour of blur operators](https://deepinv.org/auto_examples/physics/demo_blur_tour.md)

  <div class="sphx-glr-thumbnail-title">Tour of blur operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Real-world blurry images have decorrelated opposite boundaries, unlike images synthetically blurred using circular filters. This makes the use of spectral deconvolution methods (inverse filtering, Wiener filtering) impractical and prone to ringing artifacts. Liu-Jia padding liu2008reducing is a pre-processing step that pads the input image to make it have smooth circular boundaries, while preserving the original spectral content as much as possible.">![](auto_examples/physics/images/thumb/sphx_glr_demo_liu_jia_padding_thumb.png)

[Spectral Methods for Non-Circular Deblurring with Liu-Jia Padding](https://deepinv.org/auto_examples/physics/demo_liu_jia_padding.md)

  <div class="sphx-glr-thumbnail-title">Spectral Methods for Non-Circular Deblurring with Liu-Jia Padding</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of 3D blur operators in the library. In particular, we show how to use Diffraction Blurs (Fresnel diffraction) to simulate fluorescence microscopes.">![](auto_examples/physics/images/thumb/sphx_glr_demo_microscopy_3d_thumb.png)

[3D diffraction PSF](https://deepinv.org/auto_examples/physics/demo_microscopy_3d.md)

  <div class="sphx-glr-thumbnail-title">3D diffraction PSF</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">![](auto_examples/physics/images/thumb/sphx_glr_demo_mri_tour_thumb.png)

[Tour of MRI functionality in DeepInverse](https://deepinv.org/auto_examples/physics/demo_mri_tour.md)

  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct an image from them.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet2d_thumb.png)

[Positron emission tomography (PET) in 2D](https://deepinv.org/auto_examples/physics/demo_pet2d.md)

  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 2D</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct a volume from them.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet3d_thumb.png)

[Positron emission tomography (PET) in 3D](https://deepinv.org/auto_examples/physics/demo_pet3d.md)

  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 3D</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a 2D slice from the BrainWeb \`&lt;https://github.com/casperdcl/brainweb&gt;\`_ positron emission tomography (PET) dataset. The slice contains five hot lesions, we simulate a sinogram with deepinv.physics.PET and we compare standard PET reconstruction algorithms with methods that support penalized objective functions.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet_brainweb_2d_thumb.png)

[2D PET reconstruction with the Brainweb dataset](https://deepinv.org/auto_examples/physics/demo_pet_brainweb_2d.md)

  <div class="sphx-glr-thumbnail-title">2D PET reconstruction with the Brainweb dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a volume from the BrainWeb \`&lt;https://github.com/casperdcl/brainweb&gt;\`_ positron emission tomography (PET) dataset. We compare standard PET reconstruction algorithms with methods that support penalized objective functions.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet_brainweb_3d_thumb.png)

[3D PET reconstruction with the Brainweb dataset](https://deepinv.org/auto_examples/physics/demo_pet_brainweb_3d.md)

  <div class="sphx-glr-thumbnail-title">3D PET reconstruction with the Brainweb dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to create a random phase retrieval operator and generate phaseless measurements from a given image. The example showcases 4 different reconstruction methods to recover the image from the phaseless measurements:">![](auto_examples/physics/images/thumb/sphx_glr_demo_phase_retrieval_thumb.png)

[Random phase retrieval and reconstruction methods.](https://deepinv.org/auto_examples/physics/demo_phase_retrieval.md)

  <div class="sphx-glr-thumbnail-title">Random phase retrieval and reconstruction methods.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in here.">![](auto_examples/physics/images/thumb/sphx_glr_demo_physics_tour_thumb.png)

[Tour of forward sensing operators](https://deepinv.org/auto_examples/physics/demo_physics_tour.md)

  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to create a Ptychography phase retrieval operator and generate phaseless measurements from a given image.">![](auto_examples/physics/images/thumb/sphx_glr_demo_ptychography_thumb.png)

[Ptychography phase retrieval](https://deepinv.org/auto_examples/physics/demo_ptychography.md)

  <div class="sphx-glr-thumbnail-title">Ptychography phase retrieval</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we demonstrate remote sensing inverse problems for multispectral satellite imaging.">![](auto_examples/physics/images/thumb/sphx_glr_demo_remote_sensing_thumb.png)

[Remote sensing with satellite images](https://deepinv.org/auto_examples/physics/demo_remote_sensing.md)

  <div class="sphx-glr-thumbnail-title">Remote sensing with satellite images</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we show how to use the deepinv.physics.Scattering forward model.">![](auto_examples/physics/images/thumb/sphx_glr_demo_scattering_thumb.png)

[Inverse scattering problem](https://deepinv.org/auto_examples/physics/demo_scattering.md)

  <div class="sphx-glr-thumbnail-title">Inverse scattering problem</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows the use of the deepinv.physics.SpatialUnwrapping forward model and the deepinv.optim.ItohFidelity for unwrapping problems, which occur in modulo imaging, interferometry SAR and other imaging applications. It shows how to generate a wrapped phase image, apply blur and noise, and reconstruct the original phase using both DCT inversion and ADMM optimization.">![](auto_examples/physics/images/thumb/sphx_glr_demo_spatial_unwrapping_thumb.png)

[Spatial unwrapping and modulo imaging](https://deepinv.org/auto_examples/physics/demo_spatial_unwrapping.md)

  <div class="sphx-glr-thumbnail-title">Spatial unwrapping and modulo imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo illustrates the impact of different Hadamard pattern ordering algorithms in the Single Pixel Camera (SPC), a computational imaging system that uses a single photodetector to capture images by projecting the scene onto a series of patterns. The SPC is implemented in the DeepInverse library.">![](auto_examples/physics/images/thumb/sphx_glr_demo_spc_thumb.png)

[Pattern Ordering in a Compressive Single Pixel Camera](https://deepinv.org/auto_examples/physics/demo_spc.md)

  <div class="sphx-glr-thumbnail-title">Pattern Ordering in a Compressive Single Pixel Camera</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the plane-wave ultrafast ultrasound forward physics (deepinv.physics.UltrasoundPlaneWave) available in DeepInverse for pulse-echo imaging problems.">![](auto_examples/physics/images/thumb/sphx_glr_demo_ultrasound_tour_thumb.png)

[Tour of ultrafast ultrasound in DeepInverse](https://deepinv.org/auto_examples/physics/demo_ultrasound_tour.md)

  <div class="sphx-glr-thumbnail-title">Tour of ultrafast ultrasound in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to define your own optimization algorithm. For example, here, we implement the Primal-Dual Condat-Vu (CV) algorithm, and apply it for Single Pixel Camera reconstruction.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_PnP_custom_optim_thumb.png)

[PnP with custom optimization algorithm (Primal-Dual Condat-Vu)](https://deepinv.org/auto_examples/plug-and-play/demo_PnP_custom_optim.md)

  <div class="sphx-glr-thumbnail-title">PnP with custom optimization algorithm (Primal-Dual Condat-Vu)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use a mirror descent algorithm for solving an inverse problem with Poisson noise. See hurault2023convergent for more details.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_PnP_mirror_descent_thumb.png)

[Plug-and-Play algorithm with Mirror Descent for Poisson noise inverse problems.](https://deepinv.org/auto_examples/plug-and-play/demo_PnP_mirror_descent.md)

  <div class="sphx-glr-thumbnail-title">Plug-and-Play algorithm with Mirror Descent for Poisson noise inverse problems.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Plug-and-Play (PnP) is known to be challenging to apply to certain inverse problems like inpainting. One way to overcome this is to use a multi-scale approach instead of the standard single-scale approach.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_PnP_multiscale_thumb.png)

[Multi-scale Plug-and-Play for Inpainting](https://deepinv.org/auto_examples/plug-and-play/demo_PnP_multiscale.md)

  <div class="sphx-glr-thumbnail-title">Multi-scale Plug-and-Play for Inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use deepinv.physics.UltrasoundPlaneWave to reconstructs an in-vivo carotid acquisition of the EPFL LTS5 ultrafast ultrasound dataset from raw RF ultrasound data.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_ultrasound_invivo_PnP_thumb.png)

[In-vivo ultrafast ultrasound reconstruction with Plug-and-Play](https://deepinv.org/auto_examples/plug-and-play/demo_ultrasound_invivo_PnP.md)

  <div class="sphx-glr-thumbnail-title">In-vivo ultrafast ultrasound reconstruction with Plug-and-Play</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard PnP algorithm with DnCNN denoiser for computed tomography.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_vanilla_PnP_thumb.png)

[Vanilla PnP for computed tomography (CT).](https://deepinv.org/auto_examples/plug-and-play/demo_vanilla_PnP.md)

  <div class="sphx-glr-thumbnail-title">Vanilla PnP for computed tomography (CT).</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to build your custom sampling kernel. Here we build a preconditioned Unadjusted Langevin Algorithm (PreconULA) that takes advantage of the singular value decomposition of the forward operator to accelerate the sampling.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_custom_kernel_thumb.png)

[Building your custom MCMC sampling algorithm.](https://deepinv.org/auto_examples/sampling/demo_custom_kernel.md)

  <div class="sphx-glr-thumbnail-title">Building your custom MCMC sampling algorithm.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows you how to use the DDRM diffusion algorithm kawar2022denoising to reconstruct images and also compute the uncertainty of a reconstruction from incomplete and noisy measurements.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_ddrm_thumb.png)

[Image reconstruction with a diffusion model](https://deepinv.org/auto_examples/sampling/demo_ddrm.md)

  <div class="sphx-glr-thumbnail-title">Image reconstruction with a diffusion model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we revisit the implementation of the DiffPIR diffusion algorithm for image reconstruction from zhu2023denoising. The full algorithm is implemented in deepinv.sampling.DiffPIR.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_diffpir_thumb.png)

[Implementing DiffPIR](https://deepinv.org/auto_examples/sampling/demo_diffpir.md)

  <div class="sphx-glr-thumbnail-title">Implementing DiffPIR</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use our wrapper deepinv.models.DiffusersDenoiserWrapper to turn any SOTA models from the HuggingFace Hub to an image denoiser. It also can be used to perform unconditional image generation or for posterior sampling.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_diffusers_thumb.png)

[Using state-of-the-art diffusion models from HuggingFace Diffusers with DeepInverse](https://deepinv.org/auto_examples/sampling/demo_diffusers.md)

  <div class="sphx-glr-thumbnail-title">Using state-of-the-art diffusion models from HuggingFace Diffusers with DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.sampling.PosteriorDiffusion to perform posterior sampling. It also can be used to perform unconditional image generation with arbitrary denoisers, if the data fidelity term is not specified.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_diffusion_sde_thumb.png)

[Building your diffusion posterior sampling method using SDEs](https://deepinv.org/auto_examples/sampling/demo_diffusion_sde.md)

  <div class="sphx-glr-thumbnail-title">Building your diffusion posterior sampling method using SDEs</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we will go over the steps in the Diffusion Posterior Sampling (DPS) algorithm introduced in chung2022diffusion. The full algorithm is implemented in deepinv.sampling.DPS.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_dps_thumb.png)

[DPS – Posterior Sampling with Diffusion Models](https://deepinv.org/auto_examples/sampling/demo_dps.md)

  <div class="sphx-glr-thumbnail-title">DPS -- Posterior Sampling with Diffusion Models</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to perform unconditional image generation and posterior sampling using Flow Matching (FM).">![](auto_examples/sampling/images/thumb/sphx_glr_demo_flow_matching_thumb.png)

[Flow-Matching for posterior sampling and unconditional generation](https://deepinv.org/auto_examples/sampling/demo_flow_matching.md)

  <div class="sphx-glr-thumbnail-title">Flow-Matching for posterior sampling and unconditional generation</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example compares six approximations of the measurement-matching term used by diffusion posterior samplers.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_noisy_data_fidelity_thumb.png)

[Noisy data-fidelity terms for diffusion posterior sampling](https://deepinv.org/auto_examples/sampling/demo_noisy_data_fidelity.md)

  <div class="sphx-glr-thumbnail-title">Noisy data-fidelity terms for diffusion posterior sampling</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows you how to use sampling algorithms to quantify uncertainty of a reconstruction from incomplete and noisy measurements.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_sampling_thumb.png)

[Uncertainty quantification with PnP-ULA.](https://deepinv.org/auto_examples/sampling/demo_sampling.md)

  <div class="sphx-glr-thumbnail-title">Uncertainty quantification with PnP-ULA.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the self-supervised Artifact2Artifact loss for solving an undersampled sequential MRI reconstruction problem without ground truth.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_artifact2artifact_thumb.png)

[Self-supervised MRI reconstruction with Artifact2Artifact](https://deepinv.org/auto_examples/self-supervised-learning/demo_artifact2artifact.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised MRI reconstruction with Artifact2Artifact</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates various geometric image transformations implemented in deepinv that can be used in Equivariant Imaging (EI) for self-supervised learning:">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_ei_transforms_thumb.png)

[Image transformations for Equivariant Imaging](https://deepinv.org/auto_examples/self-supervised-learning/demo_ei_transforms.md)

  <div class="sphx-glr-thumbnail-title">Image transformations for Equivariant Imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Equivariant splitting consists in minimizing a self-supervised loss to train a reconstruction model using measurement data only sechaud26Equivariant.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_equivariant_splitting_thumb.png)

[Self-supervised learning with Equivariant Splitting](https://deepinv.org/auto_examples/self-supervised-learning/demo_equivariant_splitting.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised (blind) denoising of a low-field MRI scan without ground truth data.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_lowfieldmri_thumb.png)

[Low-field MRI denoising without ground truth](https://deepinv.org/auto_examples/self-supervised-learning/demo_lowfieldmri.md)

  <div class="sphx-glr-thumbnail-title">Low-field MRI denoising without ground truth</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to restore a single image corrupted by Poisson noise using Poisson2Sparse, without requiring external training or knowledge of the noise level.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_poisson2sparse_thumb.png)

[Poisson denoising using Poisson2Sparse](https://deepinv.org/auto_examples/self-supervised-learning/demo_poisson2sparse.md)

  <div class="sphx-glr-thumbnail-title">Poisson denoising using Poisson2Sparse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate scan-specific self-supervised learning, that is, learning to reconstruct MRI scans from a single accelerated sample without ground truth.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_scan_specific_thumb.png)

[Scan-specific zero-shot SSDU for MRI](https://deepinv.org/auto_examples/self-supervised-learning/demo_scan_specific.md)

  <div class="sphx-glr-thumbnail-title">Scan-specific zero-shot SSDU for MRI</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example despeckles real clinical ultrasound B-mode images with Speckle2Self li2025speckle2self, a model pretrained without clean reference images.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_ultrasound_despeckling_thumb.png)

[Ultrasound despeckling from B-mode images](https://deepinv.org/auto_examples/self-supervised-learning/demo_ultrasound_despeckling.md)

  <div class="sphx-glr-thumbnail-title">Ultrasound despeckling from B-mode images</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the use of our deepinv.transform module for use in solving imaging problems. These can be used for:">![](auto_examples/transforms-equivariance/images/thumb/sphx_glr_demo_transforms_thumb.png)

[Image transforms for equivariance & augmentations](https://deepinv.org/auto_examples/transforms-equivariance/demo_transforms.md)

  <div class="sphx-glr-thumbnail-title">Image transforms for equivariance & augmentations</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This a toy example to show you how to use DEQ to solve a deblurring problem. Note that this is a small dataset for training. For optimal results, use a larger dataset.">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_DEQ_thumb.png)

[Deep Equilibrium (DEQ) algorithms for image deblurring](https://deepinv.org/auto_examples/unfolded/demo_DEQ.md)

  <div class="sphx-glr-thumbnail-title">Deep Equilibrium (DEQ) algorithms for image deblurring</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to implement the LISTA algorithm gregor2010learning, for a compressed sensing problem. In a nutshell, LISTA is an unfolded proximal gradient algorithm involving a soft-thresholding proximal operator with learnable thresholding parameters.">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_LISTA_thumb.png)

[Learned Iterative Soft-Thresholding Algorithm (LISTA) for compressed sensing](https://deepinv.org/auto_examples/unfolded/demo_LISTA.md)

  <div class="sphx-glr-thumbnail-title">Learned Iterative Soft-Thresholding Algorithm (LISTA) for compressed sensing</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to implement a learned unrolled proximal gradient descent algorithm with a custom prior function. The custom prior in use is The algorithm is trained on a dataset of compressed sensing measurements of MNIST images.">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_custom_prior_unfolded_thumb.png)

[Learned iterative custom prior](https://deepinv.org/auto_examples/unfolded/demo_custom_prior_unfolded.md)

  <div class="sphx-glr-thumbnail-title">Learned iterative custom prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the Deep Equilibrium Attention Least Squares (DEAL) model in DeepInverse for both denoising and a simple reconstruction settings.">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_deal_thumb.png)

[DEAL denoising and reconstruction](https://deepinv.org/auto_examples/unfolded/demo_deal.md)

  <div class="sphx-glr-thumbnail-title">DEAL denoising and reconstruction</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="where both the data fidelity and the prior are learned modules, distinct for each iterations.">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_learned_primal_dual_thumb.png)

[Learned Primal-Dual algorithm for CT scan.](https://deepinv.org/auto_examples/unfolded/demo_learned_primal_dual.md)

  <div class="sphx-glr-thumbnail-title">Learned Primal-Dual algorithm for CT scan.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Image inpainting consists in solving y = Ax where A is a mask operator. This problem can be reformulated as the following minimization problem:">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_unfolded_constrained_LISTA_thumb.png)

[Unfolded Chambolle-Pock for constrained image inpainting](https://deepinv.org/auto_examples/unfolded/demo_unfolded_constrained_LISTA.md)

  <div class="sphx-glr-thumbnail-title">Unfolded Chambolle-Pock for constrained image inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use vanilla unfolded Plug-and-Play. The DnCNN denoiser and the algorithm parameters (stepsize, regularization parameters) are trained jointly. For simplicity, we show how to train the algorithm on a  small dataset. For optimal results, use a larger dataset.">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_vanilla_unfolded_thumb.png)

[Vanilla Unfolded algorithm for super-resolution](https://deepinv.org/auto_examples/unfolded/demo_vanilla_unfolded.md)

  <div class="sphx-glr-thumbnail-title">Vanilla Unfolded algorithm for super-resolution</div>
</div>
<!-- thumbnail-parent-div-close --></div>
