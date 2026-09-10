<a id="utils"></a>

# Utils

<a id="plotting"></a>

## Plotting

We provide some plotting functions that are adapted to inverse problems.
The main plotting function is [`deepinv.utils.plot`](https://deepinv.org/api/stubs/deepinv.utils.plot.html.md#deepinv.utils.plot),
which can be used to quickly plot a list of tensor images.

```pycon
>>> from deepinv.utils import plot
>>> import torch
>>> x1 = torch.rand(4, 3, 16, 16)
>>> x2 = torch.rand(4, 3, 16, 16)
>>> plot([x1, x2], titles=['x1', 'x2'])
```

#### HINT
Do you get a matplotlib LaTeX error when plotting? Disable LaTeX using `dinv.utils.disable_tex()`

We provide other plotting functions that are useful for inverse problems:

#### Utility Functions and Descriptions

| **Function**                                                                                                   | **Description**                                                                            |
|----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| [`deepinv.utils.plot()`](https://deepinv.org/api/stubs/deepinv.utils.plot.html.md#deepinv.utils.plot)                       | Plots a list of tensor images with optional titles.                                        |
| [`deepinv.utils.plot_curves()`](https://deepinv.org/api/stubs/deepinv.utils.plot_curves.html.md#deepinv.utils.plot_curves)         | Plots curves for visualizing metrics over optimization iterations.                         |
| [`deepinv.utils.plot_parameters()`](https://deepinv.org/api/stubs/deepinv.utils.plot_parameters.html.md#deepinv.utils.plot_parameters) | Visualizes model parameters over optimization iterations.                                  |
| [`deepinv.utils.plot_inset()`](https://deepinv.org/api/stubs/deepinv.utils.plot_inset.html.md#deepinv.utils.plot_inset)           | Plots a list of images with zoomed-in insets extracted from the images.                    |
| [`deepinv.utils.plot_videos()`](https://deepinv.org/api/stubs/deepinv.utils.plot_videos.html.md#deepinv.utils.plot_videos)         | Plots and animates a list of image sequences.                                              |
| [`deepinv.utils.save_videos()`](https://deepinv.org/api/stubs/deepinv.utils.save_videos.html.md#deepinv.utils.save_videos)         | Save a list of image sequences.                                                            |
| [`deepinv.utils.plot_ortho3D()`](https://deepinv.org/api/stubs/deepinv.utils.plot_ortho3D.html.md#deepinv.utils.plot_ortho3D)       | Plots 3D orthographic projections for analyzing data or model outputs in three dimensions. |
| [`deepinv.utils.plot_napari()`](https://deepinv.org/api/stubs/deepinv.utils.plot_napari.html.md#deepinv.utils.plot_napari)         | Opens an interactive napari viewer to inspect 2D images or 3D volumes/stacks.              |
| [`deepinv.utils.disable_tex()`](https://deepinv.org/api/stubs/deepinv.utils.disable_tex.html.md#deepinv.utils.disable_tex)         | Globally force disable LaTeX for matplotlib plotting.                                      |
| [`deepinv.utils.enable_tex()`](https://deepinv.org/api/stubs/deepinv.utils.enable_tex.html.md#deepinv.utils.enable_tex)           | Globally force enable LaTeX for matplotlib plotting.                                       |

<a id="logging"></a>

## Logging

#### Logging functionality

| **Function/class**                                                                                         | **Description**                                 |
|------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| [`deepinv.utils.AverageMeter()`](https://deepinv.org/api/stubs/deepinv.utils.AverageMeter.html.md#deepinv.utils.AverageMeter)   | Store values and keep track of average and std. |
| [`deepinv.utils.get_timestamp()`](https://deepinv.org/api/stubs/deepinv.utils.get_timestamp.html.md#deepinv.utils.get_timestamp) | Get current timestamp string.                   |

<a id="io-utils"></a>

## Image Loading

We provide utilities for loading images and data from various sources:

#### Image Loading Functions

| **Function**                                                                                           | **Description**                                                                          |
|--------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| [`deepinv.utils.load_dicom()`](https://deepinv.org/api/stubs/deepinv.utils.load_dicom.html.md#deepinv.utils.load_dicom)   | Load images as tensors from DICOM files.                                                 |
| [`deepinv.utils.load_tiff()`](https://deepinv.org/api/stubs/deepinv.utils.load_tiff.html.md#deepinv.utils.load_tiff)     | Load images or volumes as tensors from TIFF files.                                       |
| [`deepinv.utils.load_url()`](https://deepinv.org/api/stubs/deepinv.utils.load_url.html.md#deepinv.utils.load_url)       | Load a file into a buffer directly from a URL.                                           |
| [`deepinv.utils.load_np()`](https://deepinv.org/api/stubs/deepinv.utils.load_np.html.md#deepinv.utils.load_np)         | Load NumPy arrays to tensors from disk.                                                  |
| [`deepinv.utils.load_torch()`](https://deepinv.org/api/stubs/deepinv.utils.load_torch.html.md#deepinv.utils.load_torch)   | Load PyTorch tensors from disk.                                                          |
| [`deepinv.utils.load_mat()`](https://deepinv.org/api/stubs/deepinv.utils.load_mat.html.md#deepinv.utils.load_mat)       | Load MATLAB `.mat` files from disk.                                                      |
| [`deepinv.utils.load_raster()`](https://deepinv.org/api/stubs/deepinv.utils.load_raster.html.md#deepinv.utils.load_raster) | Load raster image formats (e.g. satellite images `.tif`, `.geotiff`, SAR images `.cos`). |
| [`deepinv.utils.load_ismrmd()`](https://deepinv.org/api/stubs/deepinv.utils.load_ismrmd.html.md#deepinv.utils.load_ismrmd) | Load raw MRI data in ISMRMD format using `h5py`.                                         |

<a id="tiling-utils"></a>

## Tiling / Untiling (Patching and Unpatching)

We provide utilities for extracting and merging tiles (patches) from 2D images:

#### Tiling and Untiling Functions

| **Function**                                                                                                     | **Description**                                                                                                                                           |
|------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`deepinv.utils.patch_extractor()`](https://deepinv.org/api/stubs/deepinv.utils.patch_extractor.html.md#deepinv.utils.patch_extractor)   | Extracts random patches from a 2D image tensor.                                                                                                           |
| [`deepinv.utils.image_to_patches()`](https://deepinv.org/api/stubs/deepinv.utils.image_to_patches.html.md#deepinv.utils.image_to_patches) | Splits a 2D image tensor into overlapping patches.                                                                                                        |
| [`deepinv.utils.patches_to_image()`](https://deepinv.org/api/stubs/deepinv.utils.patches_to_image.html.md#deepinv.utils.patches_to_image) | Merges overlapping patches back into a 2D image tensor.                                                                                                   |
| [`deepinv.utils.patchify()`](https://deepinv.org/api/stubs/deepinv.utils.patchify.html.md#deepinv.utils.patchify)                 | An alias for [`deepinv.utils.image_to_patches()`](https://deepinv.org/api/stubs/deepinv.utils.image_to_patches.html.md#deepinv.utils.image_to_patches) for backward compatibility. |

<a id="demo-utils"></a>

## Demo Utils

These functions make it easy to fetch demo data and resources for experiments:

#### Demo Utility Functions

| **Function**                                                                                                           | **Description**                                                                           |
|------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| [`deepinv.utils.load_image()`](https://deepinv.org/api/stubs/deepinv.utils.load_image.html.md#deepinv.utils.load_image)                   | Loads a local image file for experiments or demos.                                        |
| [`deepinv.utils.load_url_image()`](https://deepinv.org/api/stubs/deepinv.utils.load_url_image.html.md#deepinv.utils.load_url_image)           | Loads an image directly from a URL for experiments or demos.                              |
| [`deepinv.utils.load_np_url()`](https://deepinv.org/api/stubs/deepinv.utils.load_np_url.html.md#deepinv.utils.load_np_url)                 | Loads a NumPy array into a tensor directly from a URL.                                    |
| [`deepinv.utils.load_torch_url()`](https://deepinv.org/api/stubs/deepinv.utils.load_torch_url.html.md#deepinv.utils.load_torch_url)           | Loads a PyTorch tensor directly from a URL.                                               |
| [`deepinv.utils.load_example()`](https://deepinv.org/api/stubs/deepinv.utils.load_example.html.md#deepinv.utils.load_example)               | Loads an image directly from DeepInverse HuggingFace repository for experiments or demos. |
| [`deepinv.utils.download_example()`](https://deepinv.org/api/stubs/deepinv.utils.download_example.html.md#deepinv.utils.download_example)       | Downloads an image from DeepInverse HuggingFace repository to file.                       |
| [`deepinv.utils.get_cache_home()`](https://deepinv.org/api/stubs/deepinv.utils.get_cache_home.html.md#deepinv.utils.get_cache_home)           | Get the path to the default directory for storing cached files.                           |
| [`deepinv.utils.get_image_url()`](https://deepinv.org/api/stubs/deepinv.utils.get_image_url.html.md#deepinv.utils.get_image_url)             | Get URL for an image from DeepInverse HuggingFace repository.                             |
| [`deepinv.utils.get_degradation_url()`](https://deepinv.org/api/stubs/deepinv.utils.get_degradation_url.html.md#deepinv.utils.get_degradation_url) | Get URL for a degradation from DeepInverse HuggingFace repository.                        |
| [`deepinv.utils.load_dataset()`](https://deepinv.org/api/stubs/deepinv.utils.load_dataset.html.md#deepinv.utils.load_dataset)               | Loads an ImageFolder dataset from DeepInverse HuggingFace repository.                     |
| [`deepinv.utils.load_degradation()`](https://deepinv.org/api/stubs/deepinv.utils.load_degradation.html.md#deepinv.utils.load_degradation)       | Loads a degradation tensor from DeepInverse HuggingFace repository.                       |

<a id="other-utils"></a>

## Other

Other miscellaneous utility functions:

#### Other Utility Functions

| **Function**                                                                                                   | **Description**                                                     |
|----------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------|
| [`deepinv.utils.get_freer_gpu()`](https://deepinv.org/api/stubs/deepinv.utils.get_freer_gpu.html.md#deepinv.utils.get_freer_gpu)     | Finds the GPU with the most available memory.                       |
| [`deepinv.utils.dirac()`](https://deepinv.org/api/stubs/deepinv.utils.dirac.html.md#deepinv.utils.dirac)                     | Creates a Dirac delta tensor.                                       |
| [`deepinv.utils.dirac_like()`](https://deepinv.org/api/stubs/deepinv.utils.dirac_like.html.md#deepinv.utils.dirac_like)           | Creates a Dirac delta tensor with the same shape as the input.      |
| [`deepinv.utils.dirac_comb()`](https://deepinv.org/api/stubs/deepinv.utils.dirac_comb.html.md#deepinv.utils.dirac_comb)           | Creates a Dirac delta comb tensor.                                  |
| [`deepinv.utils.dirac_comb_like()`](https://deepinv.org/api/stubs/deepinv.utils.dirac_comb_like.html.md#deepinv.utils.dirac_comb_like) | Creates a Dirac delta comb tensor with the same shape as the input. |

<a id="tensorlist"></a>

## TensorList

The [`deepinv.utils.TensorList`](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) class is a wrapper around a list of tensors. It allows performing
elementary operations on the list of tensors, such as sum, multiplication, etc.:

```pycon
>>> from deepinv.utils import TensorList
>>> import torch
>>> x1 = torch.ones(2, 3, 2, 2)
>>> x2 = torch.ones(2, 1, 3, 3)
>>> t1 = TensorList([x1, x2])
>>> t2 = TensorList([x1*2, x2/2])
>>> t3 = t1 + t2
```

<a id="mixin"></a>

### Mixins

DeepInverse maximizes code reuse via inheritance.
We provide mixin classes to provide specialized methods for certain physics, models, datasets and losses,
such as temporal or MRI functionality.

#### Mixins

| **Mixin**                                                                                              | **Description**                                                             |
|--------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| [`deepinv.utils.MRIMixin`](https://deepinv.org/api/stubs/deepinv.utils.MRIMixin.html.md#deepinv.utils.MRIMixin)         | Utility methods for MRI physics.                                            |
| [`deepinv.utils.TimeMixin`](https://deepinv.org/api/stubs/deepinv.utils.TimeMixin.html.md#deepinv.utils.TimeMixin)       | Methods for expanding and flattening time dimension for dynamic/video data. |
| [`deepinv.utils.TiledMixin2d`](https://deepinv.org/api/stubs/deepinv.utils.TiledMixin2d.html.md#deepinv.utils.TiledMixin2d) | Methods for extracting and merging tiles for 2D images.                     |
