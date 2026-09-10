# deepinv.utils

This module provides various plotting and utility functions.
Please refer to the [user guide](https://deepinv.org/user_guide/other/utils.html.md#utils) for more information.

## Plotting

**User Guide:** refer to [Plotting](https://deepinv.org/user_guide/other/utils.html.md#plotting) for more information.

| [`deepinv.utils.plot`](https://deepinv.org/api/stubs/deepinv.utils.plot.html.md#deepinv.utils.plot)                                             | Plots a list of images.                                                 |
|------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| [`deepinv.utils.plot_curves`](https://deepinv.org/api/stubs/deepinv.utils.plot_curves.html.md#deepinv.utils.plot_curves)                               | Plots the metrics of a Plug-and-Play algorithm.                         |
| [`deepinv.utils.plot_parameters`](https://deepinv.org/api/stubs/deepinv.utils.plot_parameters.html.md#deepinv.utils.plot_parameters)                       | Plot the parameters of the model before and after training.             |
| [`deepinv.utils.plot_inset`](https://deepinv.org/api/stubs/deepinv.utils.plot_inset.html.md#deepinv.utils.plot_inset)                                 | Plots a list of images with zoomed-in insets extracted from the images. |
| [`deepinv.utils.plot_videos`](https://deepinv.org/api/stubs/deepinv.utils.plot_videos.html.md#deepinv.utils.plot_videos)                               | Plots and animates a list of image sequences.                           |
| [`deepinv.utils.save_videos`](https://deepinv.org/api/stubs/deepinv.utils.save_videos.html.md#deepinv.utils.save_videos)                               | Saves an animation of a list of image sequences.                        |
| [`deepinv.utils.plot_ortho3D`](https://deepinv.org/api/stubs/deepinv.utils.plot_ortho3D.html.md#deepinv.utils.plot_ortho3D)                             | Plots an orthogonal view of 3D images.                                  |
| [`deepinv.utils.plot_napari`](https://deepinv.org/api/stubs/deepinv.utils.plot_napari.html.md#deepinv.utils.plot_napari)                               | View 2D images or 3D volumes in napari.                                 |
| [`deepinv.utils.disable_tex`](https://deepinv.org/api/stubs/deepinv.utils.disable_tex.html.md#deepinv.utils.disable_tex)                               | Globally disable LaTeX                                                  |
| [`deepinv.utils.enable_tex`](https://deepinv.org/api/stubs/deepinv.utils.enable_tex.html.md#deepinv.utils.enable_tex)                                 | Globally enable LaTeX                                                   |
| [`deepinv.utils.normalize_signal`](https://deepinv.org/api/stubs/deepinv.utils.normalize_signal.html.md#deepinv.utils.normalize_signal)                     | Normalize a batch of signals between zero and one.                      |
| [`deepinv.utils.plotting.config_matplotlib`](https://deepinv.org/api/stubs/deepinv.utils.plotting.config_matplotlib.html.md#deepinv.utils.plotting.config_matplotlib) | Config matplotlib for nice plots in the examples.                       |

## TensorList

**User Guide:** refer to [TensorList](https://deepinv.org/user_guide/other/utils.html.md#tensorlist) for more information.

| [`deepinv.utils.TensorList`](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList)   | Represents a list of [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) with different shapes.   |
|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|

| [`deepinv.utils.zeros_like`](https://deepinv.org/api/stubs/deepinv.utils.zeros_like.html.md#deepinv.utils.zeros_like)   | Returns a [`deepinv.utils.TensorList`](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) or [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) with the same type as x, filled with zeros.                           |
|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`deepinv.utils.ones_like`](https://deepinv.org/api/stubs/deepinv.utils.ones_like.html.md#deepinv.utils.ones_like)     | Returns a [`deepinv.utils.TensorList`](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) or [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) with the same type as x, filled with ones.                            |
| [`deepinv.utils.randn_like`](https://deepinv.org/api/stubs/deepinv.utils.randn_like.html.md#deepinv.utils.randn_like)   | Returns a [`deepinv.utils.TensorList`](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) or [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) with the same type as x, filled with standard gaussian numbers.       |
| [`deepinv.utils.rand_like`](https://deepinv.org/api/stubs/deepinv.utils.rand_like.html.md#deepinv.utils.rand_like)     | Returns a [`deepinv.utils.TensorList`](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) or [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) with the same type as x, filled with random uniform numbers in [0,1]. |

## Logging

**User Guide:** refer to [Logging](https://deepinv.org/user_guide/other/utils.html.md#logging) for more information.

| [`deepinv.utils.AverageMeter`](https://deepinv.org/api/stubs/deepinv.utils.AverageMeter.html.md#deepinv.utils.AverageMeter)   | Compute and store aggregates online from a stream of scalar values   |
|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| [`deepinv.utils.ProgressMeter`](https://deepinv.org/api/stubs/deepinv.utils.ProgressMeter.html.md#deepinv.utils.ProgressMeter) |                                                                      |
| [`deepinv.utils.get_timestamp`](https://deepinv.org/api/stubs/deepinv.utils.get_timestamp.html.md#deepinv.utils.get_timestamp) | Get current timestamp string.                                        |

## Mixins

**User Guide:** refer to [Mixins](https://deepinv.org/user_guide/other/utils.html.md#mixin) for more information.

| [`deepinv.utils.MRIMixin`](https://deepinv.org/api/stubs/deepinv.utils.MRIMixin.html.md#deepinv.utils.MRIMixin)         | Mixin base class for MRI functionality.                            |
|--------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| [`deepinv.utils.TimeMixin`](https://deepinv.org/api/stubs/deepinv.utils.TimeMixin.html.md#deepinv.utils.TimeMixin)       | Base class for temporal capabilities for physics and models.       |
| [`deepinv.utils.TiledMixin2d`](https://deepinv.org/api/stubs/deepinv.utils.TiledMixin2d.html.md#deepinv.utils.TiledMixin2d) | Mixin base class for 2D tiled patch extraction and reconstruction. |

## Image Loading

**User Guide:** refer to [Image Loading](https://deepinv.org/user_guide/other/utils.html.md#io-utils) for more information.

| [`deepinv.utils.load_dicom`](https://deepinv.org/api/stubs/deepinv.utils.load_dicom.html.md#deepinv.utils.load_dicom)   | Load image from DICOM file.                                         |
|------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------|
| [`deepinv.utils.load_nifti`](https://deepinv.org/api/stubs/deepinv.utils.load_nifti.html.md#deepinv.utils.load_nifti)   | Load volume from nifti file as torch tensor.                        |
| [`deepinv.utils.load_tiff`](https://deepinv.org/api/stubs/deepinv.utils.load_tiff.html.md#deepinv.utils.load_tiff)     | Load image or volume from a TIFF file as a torch tensor.            |
| [`deepinv.utils.load_url`](https://deepinv.org/api/stubs/deepinv.utils.load_url.html.md#deepinv.utils.load_url)       | Load URL to a buffer.                                               |
| [`deepinv.utils.load_np`](https://deepinv.org/api/stubs/deepinv.utils.load_np.html.md#deepinv.utils.load_np)         | Load numpy array from file as torch tensor.                         |
| [`deepinv.utils.load_torch`](https://deepinv.org/api/stubs/deepinv.utils.load_torch.html.md#deepinv.utils.load_torch)   | Load torch tensor from file.                                        |
| [`deepinv.utils.load_mat`](https://deepinv.org/api/stubs/deepinv.utils.load_mat.html.md#deepinv.utils.load_mat)       | Load MATLAB array from file.                                        |
| [`deepinv.utils.load_raster`](https://deepinv.org/api/stubs/deepinv.utils.load_raster.html.md#deepinv.utils.load_raster) | Load a raster image and return patches as tensors using `rasterio`. |
| [`deepinv.utils.load_ismrmd`](https://deepinv.org/api/stubs/deepinv.utils.load_ismrmd.html.md#deepinv.utils.load_ismrmd) | Load complex MRI data from ISMRMD format.                           |

| [`deepinv.utils.DownloadError`](https://deepinv.org/api/stubs/deepinv.utils.DownloadError.html.md#deepinv.utils.DownloadError)   | Raised when a network download initiated by deepinv fails.   |
|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|

## Demo Utils

**User Guide:** refer to [Demo Utils](https://deepinv.org/user_guide/other/utils.html.md#demo-utils) for more information.

| [`deepinv.utils.load_image`](https://deepinv.org/api/stubs/deepinv.utils.load_image.html.md#deepinv.utils.load_image)                   | Load an image from a file and return a torch.Tensor with a batch dimension.                                   |
|----------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| [`deepinv.utils.load_url_image`](https://deepinv.org/api/stubs/deepinv.utils.load_url_image.html.md#deepinv.utils.load_url_image)           | Load an image from a URL and return a torch.Tensor with a batch dimension.                                    |
| [`deepinv.utils.load_np_url`](https://deepinv.org/api/stubs/deepinv.utils.load_np_url.html.md#deepinv.utils.load_np_url)                 | Load a numpy array from url and convert to tensor.                                                            |
| [`deepinv.utils.load_torch_url`](https://deepinv.org/api/stubs/deepinv.utils.load_torch_url.html.md#deepinv.utils.load_torch_url)           | Load an array from url and read it by torch.load.                                                             |
| [`deepinv.utils.load_example`](https://deepinv.org/api/stubs/deepinv.utils.load_example.html.md#deepinv.utils.load_example)               | Load example image from the [DeepInverse HuggingFace](https://huggingface.co/datasets/deepinv/images).        |
| [`deepinv.utils.download_example`](https://deepinv.org/api/stubs/deepinv.utils.download_example.html.md#deepinv.utils.download_example)       | Download an image from the [DeepInverse HuggingFace](https://huggingface.co/datasets/deepinv/images) to file. |
| [`deepinv.utils.get_cache_home`](https://deepinv.org/api/stubs/deepinv.utils.get_cache_home.html.md#deepinv.utils.get_cache_home)           | Return a folder to store deepinv cache (datasets, models, etc.).                                              |
| [`deepinv.utils.get_image_url`](https://deepinv.org/api/stubs/deepinv.utils.get_image_url.html.md#deepinv.utils.get_image_url)             | Get URL for image from DeepInverse HuggingFace repository.                                                    |
| [`deepinv.utils.get_degradation_url`](https://deepinv.org/api/stubs/deepinv.utils.get_degradation_url.html.md#deepinv.utils.get_degradation_url) | Get URL for degradation from DeepInverse HuggingFace repository.                                              |
| [`deepinv.utils.load_dataset`](https://deepinv.org/api/stubs/deepinv.utils.load_dataset.html.md#deepinv.utils.load_dataset)               | Loads an ImageFolder dataset from DeepInverse HuggingFace repository.                                         |
| [`deepinv.utils.load_degradation`](https://deepinv.org/api/stubs/deepinv.utils.load_degradation.html.md#deepinv.utils.load_degradation)       | Loads a degradation tensor from DeepInverse HuggingFace repository.                                           |

## Phantoms

| [`deepinv.utils.phantoms.generate_shepp_logan`](https://deepinv.org/api/stubs/deepinv.utils.phantoms.generate_shepp_logan.html.md#deepinv.utils.phantoms.generate_shepp_logan)       | Generate a Shepp-Logan phantom approximation in PyTorch.                    |
|------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| [`deepinv.utils.phantoms.generate_random_phantom`](https://deepinv.org/api/stubs/deepinv.utils.phantoms.generate_random_phantom.html.md#deepinv.utils.phantoms.generate_random_phantom) | Generate a random ellipsoid phantom directly using torch.                   |
| [`deepinv.utils.phantoms.generate_pet_phantom`](https://deepinv.org/api/stubs/deepinv.utils.phantoms.generate_pet_phantom.html.md#deepinv.utils.phantoms.generate_pet_phantom)       | Generate a 2D or 3D PET-like phantom and its corresponding attenuation map. |

| [`deepinv.utils.phantoms.SheppLoganDataset`](https://deepinv.org/api/stubs/deepinv.utils.phantoms.SheppLoganDataset.html.md#deepinv.utils.phantoms.SheppLoganDataset)       | Dataset for the single Shepp-Logan phantom.                |
|------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| [`deepinv.utils.phantoms.RandomPhantomDataset`](https://deepinv.org/api/stubs/deepinv.utils.phantoms.RandomPhantomDataset.html.md#deepinv.utils.phantoms.RandomPhantomDataset) | Dataset of random ellipsoid phantoms generated on the fly. |

## Tiling / Untiling (Patching and Unpatching)

**User Guide:** refer to [Tiling / Untiling (Patching and Unpatching)](https://deepinv.org/user_guide/other/utils.html.md#tiling-utils) for more information.

| [`deepinv.utils.patch_extractor`](https://deepinv.org/api/stubs/deepinv.utils.patch_extractor.html.md#deepinv.utils.patch_extractor)   | This function takes a `B x C x H x W` tensor as input and extracts `n_patches` random patches of size `C x patch_size x patch_size` from each `C x H x W` image.   |
|----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`deepinv.utils.image_to_patches`](https://deepinv.org/api/stubs/deepinv.utils.image_to_patches.html.md#deepinv.utils.image_to_patches) | Split a batch of images into overlapping 2D patches.                                                                                                               |
| [`deepinv.utils.patches_to_image`](https://deepinv.org/api/stubs/deepinv.utils.patches_to_image.html.md#deepinv.utils.patches_to_image) | Reconstruct images from overlapping 2D patches.                                                                                                                    |
| [`deepinv.utils.patchify`](https://deepinv.org/api/stubs/deepinv.utils.patchify.html.md#deepinv.utils.patchify)                 | Alias of [`deepinv.utils.image_to_patches()`](https://deepinv.org/api/stubs/deepinv.utils.image_to_patches.html.md#deepinv.utils.image_to_patches).                                         |

## Other

**User Guide:** refer to [Other](https://deepinv.org/user_guide/other/utils.html.md#other-utils) for more information.

| [`deepinv.utils.get_freer_gpu`](https://deepinv.org/api/stubs/deepinv.utils.get_freer_gpu.html.md#deepinv.utils.get_freer_gpu)     | Returns the GPU device with the most free memory.                                                                                                                                                                                                                     |
|--------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`deepinv.utils.dirac`](https://deepinv.org/api/stubs/deepinv.utils.dirac.html.md#deepinv.utils.dirac)                     | Returns a [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) with a Dirac delta in 2D at the center.                                                                                                                                    |
| [`deepinv.utils.dirac_like`](https://deepinv.org/api/stubs/deepinv.utils.dirac_like.html.md#deepinv.utils.dirac_like)           | Returns a [`deepinv.utils.TensorList`](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) or [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) with the same type as x, filled with zeros.                          |
| [`deepinv.utils.dirac_comb`](https://deepinv.org/api/stubs/deepinv.utils.dirac_comb.html.md#deepinv.utils.dirac_comb)           | Returns a [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) with a Dirac comb in 2D (impulse train) at the given step.                                                                                                                 |
| [`deepinv.utils.dirac_comb_like`](https://deepinv.org/api/stubs/deepinv.utils.dirac_comb_like.html.md#deepinv.utils.dirac_comb_like) | Returns a [`deepinv.utils.TensorList`](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) or [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) with the same type as x, filled with a Dirac comb at the given step. |
