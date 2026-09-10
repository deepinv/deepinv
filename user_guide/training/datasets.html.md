<a id="datasets"></a>

# Datasets

The datasets module lets you use datasets with DeepInverse, for testing and training.

Datasets can be either:

* Loaded in from your own data, see [base datasets](#base-datasets) for desired format;
* Paired dataset generated synthetically from ground-truth, see [generating datasets](#generating-datasets);
* A public dataset, such as DIV2K or FastMRI, see [predefined datasets](#predefined-datasets).

<a id="base-datasets"></a>

## Base Datasets

Datasets can return optionally ground-truth images `x`, measurements `y`, or [physics parameters](https://deepinv.org/user_guide/physics/intro.html.md#parameter-dependent-operators) `params`,
or any combination of these, in one of the following ways:

* `x` i.e a dataset that returns only ground truth;
* `(x, y)` i.e. a dataset that returns pairs of ground truth and measurement. `x` can be equal to `torch.nan` if your dataset is ground-truth-free.
* `(x, params)` i.e. a dataset of ground truth and dict of [physics parameters](https://deepinv.org/user_guide/physics/intro.html.md#physics-generators). Useful for training with online measurements.
* `(x, y, params)` i.e. a dataset that returns ground truth, measurements and dict of params.

#### TIP
If you have a dataset of measurements only `(y)` or `(y, params)` you should modify it such that it returns `(torch.nan, y)` or `(torch.nan, y, params)`

Alternatively, set `use_dict_output=True` (default False) in the dataset which makes them return a dict of the format `{"x": x, "y": y, "params": params}` with any key omitted if not applicable. This dict format is recommended over tuple for better readability and flexibility.

```pycon
>>> import torch
>>> from deepinv.datasets import TensorDataset
>>> x, y = torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8)
>>> dataset = TensorDataset(x=x, y=y, use_dict_output=True)
>>> dataset[0].keys()
['x', 'y']
```

If you have your own dataset (e.g. a PyTorch `Dataset`), check that it is compatible using the function [`deepinv.datasets.check_dataset()`](https://deepinv.org/api/stubs/deepinv.datasets.check_dataset.html.md#deepinv.datasets.check_dataset)
(e.g. to be used with [`deepinv.Trainer`](https://deepinv.org/api/stubs/deepinv.Trainer.html.md#deepinv.Trainer) or [`deepinv.test`](https://deepinv.org/api/stubs/deepinv.test.html.md#deepinv.test)).

#### SEE ALSO
See [Bring your own dataset](https://deepinv.org/auto_examples/basics/demo_custom_dataset.html.md#sphx-glr-auto-examples-basics-demo-custom-dataset-py) for a simple example of how to use DeepInverse with your own dataset.

We provide dataset classes for you to easily load in your own data:

#### Base Datasets Overview

| **Dataset**                                                                                                              | **Description**                                                                                                                                                                                                                          |
|--------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`deepinv.datasets.ImageDataset`](https://deepinv.org/api/stubs/deepinv.datasets.ImageDataset.html.md#deepinv.datasets.ImageDataset)             | Base abstract dataset class                                                                                                                                                                                                              |
| [`deepinv.datasets.ImageFolder`](https://deepinv.org/api/stubs/deepinv.datasets.ImageFolder.html.md#deepinv.datasets.ImageFolder)               | Dataset that loads images (ground-truth, measurements or both) from a folder                                                                                                                                                             |
| [`deepinv.datasets.TensorDataset`](https://deepinv.org/api/stubs/deepinv.datasets.TensorDataset.html.md#deepinv.datasets.TensorDataset)           | Dataset that returns tensor(s) passed in at input: either tensor(s) for a single observation or a whole dataset of them                                                                                                                  |
| [`deepinv.datasets.HDF5Dataset`](https://deepinv.org/api/stubs/deepinv.datasets.HDF5Dataset.html.md#deepinv.datasets.HDF5Dataset)               | Dataset of measurements generated using [`deepinv.datasets.generate_dataset()`](https://deepinv.org/api/stubs/deepinv.datasets.generate_dataset.html.md#deepinv.datasets.generate_dataset), see [below](#generating-datasets) for how to use. |
| [`deepinv.datasets.RandomPatchSampler`](https://deepinv.org/api/stubs/deepinv.datasets.RandomPatchSampler.html.md#deepinv.datasets.RandomPatchSampler) | Dataset that randomly samples a patch from a larger nD image at each iteration, accepts a ground-truth directory or measurement directory. If both are provided, filenames and shapes must match for each pair.                          |

<a id="generating-datasets"></a>

## Generating Datasets

You can generate a dataset associated with a certain forward operator using [`deepinv.datasets.generate_dataset()`](https://deepinv.org/api/stubs/deepinv.datasets.generate_dataset.html.md#deepinv.datasets.generate_dataset)
using a base dataset.
Your base dataset can be any dataset that returns ground truth, i.e. either one of our [predefined datasets](#predefined-datasets),
your own data in the format provided by one of our [base datasets](#base-datasets),
or other external datasets.

For example, here we generate a dataset of inpainting measurements from the [`deepinv.datasets.Set14HR`](https://deepinv.org/api/stubs/deepinv.datasets.Set14HR.html.md#deepinv.datasets.Set14HR) dataset:

#### NOTE
We support all data types supported by `h5py`, including complex numbers.

```pycon
>>> import deepinv as dinv
>>> from torchvision.transforms import ToTensor, Compose, CenterCrop
>>> save_dir = dinv.utils.get_cache_home() / 'set14'
>>>
>>> # Define base train dataset
>>> dataset = dinv.datasets.Set14HR(save_dir, download=True, transform=Compose([CenterCrop(128), ToTensor()]))
>>>
>>> # Define forward operator
>>> physics = dinv.physics.Inpainting(img_size=(3, 128, 128), mask=0.8, noise_model=dinv.physics.GaussianNoise(sigma=.05))
>>>
>>> # Generate paired dataset
>>> path = dinv.datasets.generate_dataset(dataset, physics, save_dir=save_dir, verbose=False)
```

The datasets are saved in `.h5` (HDF5) format, and can be easily loaded to PyTorch’s standard
[`torch.utils.data.DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader):

```pycon
>>> from torch.utils.data import DataLoader
>>>
>>> train_dataset = dinv.datasets.HDF5Dataset(path)
>>> dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
>>> x, y = next(iter(dataloader))
>>> x.shape, y.shape
(torch.Size([4, 3, 128, 128]), torch.Size([4, 3, 128, 128]))
>>> train_dataset.close()
```

We can also use physics generators to randomly generate physics `params` for data,
and save and load the physics `params` into the dataset:

```pycon
>>> physics_generator = dinv.physics.generator.SigmaGenerator()
>>> path = dinv.datasets.generate_dataset(dataset, physics, physics_generator=physics_generator, save_dir=save_dir, verbose=False)
>>> train_dataset = dinv.datasets.HDF5Dataset(path, load_physics_generator_params=True)
>>> dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
>>> x, y, params = next(iter(dataloader))
>>> print(params['sigma'].shape)
torch.Size([4])
```

<a id="predefined-datasets"></a>

## Predefined Datasets

Multiple popular easy-to-download datasets are available, which all either return
ground-truth only `x`, paired data `(x, y)` or with params (e.g. mask) `(x, y, params)`.

All these datasets inherit from [`deepinv.datasets.ImageDataset`](https://deepinv.org/api/stubs/deepinv.datasets.ImageDataset.html.md#deepinv.datasets.ImageDataset).

#### Datasets Overview

| **Dataset**                                                                                                           | **Returns**                                                  | **Dataset Size**                    | **Tensor Sizes**                                                | **Description**                                                                                                                                                                                                  |
|-----------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|-------------------------------------|-----------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`DIV2K`](https://deepinv.org/api/stubs/deepinv.datasets.DIV2K.html.md#deepinv.datasets.DIV2K)                                         | `x`                                                          | 800 (train) + 100 (val) images      | RGB, up to 2040x2040 pixels (variable)                          | A widely-used dataset for natural image restoration.                                                                                                                                                             |
| [`Urban100HR`](https://deepinv.org/api/stubs/deepinv.datasets.Urban100HR.html.md#deepinv.datasets.Urban100HR)                               | `x`                                                          | 100 images                          | up to 1200x1280 pixels (variable)                               | Contains diverse high-resolution urban scenes, typically used for testing super-resolution algorithms.                                                                                                           |
| [`Set14HR`](https://deepinv.org/api/stubs/deepinv.datasets.Set14HR.html.md#deepinv.datasets.Set14HR)                                     | `x`                                                          | 14 high-resolution images           | RGB, 248×248 to 512×768 pixels.                                 | A small benchmark dataset for super-resolution tasks, containing a variety of natural images.                                                                                                                    |
| [`Set5HR`](https://deepinv.org/api/stubs/deepinv.datasets.Set5HR.html.md#deepinv.datasets.Set5HR)                                       | `x`                                                          | 5 high-resolution images            | RGB, 256×256 to 512×512 pixels.                                 | A very small benchmark dataset commonly used for super-resolution tasks.                                                                                                                                         |
| [`BSDS500`](https://deepinv.org/api/stubs/deepinv.datasets.BSDS500.html.md#deepinv.datasets.BSDS500)                                     | `x`                                                          | 400 (train) + 100 (test) images     | RGB, 481x321 or 321x481 pixels                                  | Color Berkeley Segmentation Dataset.                                                                                                                                                                             |
| [`BSD100HR`](https://deepinv.org/api/stubs/deepinv.datasets.BSD100HR.html.md#deepinv.datasets.BSD100HR)                                   | `x`                                                          | 100 high-resolution images          | RGB, 240×160 to 480×320 pixels.                                 | A benchmark subset of BSDS300/BSDS500 commonly used for super-resolution tasks.                                                                                                                                  |
| [`McMaster`](https://deepinv.org/api/stubs/deepinv.datasets.McMaster.html.md#deepinv.datasets.McMaster)                                   | `x`                                                          | 18 images                           | RGB, 500×500 pixels.                                            | A small benchmark dataset commonly used for testing color demosaicing algorithms.                                                                                                                                |
| [`Kodak24`](https://deepinv.org/api/stubs/deepinv.datasets.Kodak24.html.md#deepinv.datasets.Kodak24)                                     | `x`                                                          | 24 images                           | RGB, 768×512 or 512×768 pixels.                                 | A widely-used benchmark dataset for denoising, compression and demosaicing.                                                                                                                                      |
| [`CBSD68`](https://deepinv.org/api/stubs/deepinv.datasets.CBSD68.html.md#deepinv.datasets.CBSD68)                                       | `x`                                                          | 68 images                           | RGB, 481x321 or 321x481 pixels                                  | A subset of the color Berkeley Segmentation Dataset.                                                                                                                                                             |
| [`FastMRISliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.FastMRISliceDataset.html.md#deepinv.datasets.FastMRISliceDataset)             | `(x, y)` or `(x, y, {'mask': mask, 'coil_maps': coil_maps})` | Over 100,000 MRI slices             | Complex, varying shape approx. 640x320                          | Raw MRI knee and brain fully-sampled or undersampled k-space data and optional RSS targets from the FastMRI dataset.                                                                                             |
| [`SimpleFastMRISliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.SimpleFastMRISliceDataset.html.md#deepinv.datasets.SimpleFastMRISliceDataset) | `x`                                                          | 973 (knee) and 455 (brain) images   | 320x320 fully-sampled reconstructed slices                      | Easy-to-use in-memory prepared subset of 2D slices from the full FastMRI slice dataset for knees and brains, padded to standard size.                                                                            |
| [`CMRxReconSliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.CMRxReconSliceDataset.html.md#deepinv.datasets.CMRxReconSliceDataset)         | `(x, y)` or `(x, y, {'mask': mask})`                         | 300 patients, each with 8-13 slices | Padded to 512x256x12 time steps                                 | Dynamic MRI sequences of cardiac cine from short axis (5-10 slices) and long axis (3 views) split by patient, from the CMRxRecon challenge.                                                                      |
| [`SKMTEASliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.SKMTEASliceDataset.html.md#deepinv.datasets.SKMTEASliceDataset)               | `(x, y, {'mask': mask, 'coil_maps': coil_maps})`             | 25,000 slices from 155 patients     | Complex double-echo with 8 coils of shape 512x160.              | Raw MRI knee multicoil undersampled k-space data and fully-sampled ground truth from the Stanford SKM-TEA dataset, with precomputed Poisson disc masks from 4x to 16x acceleration, and pre-estimated coil maps. |
| [`LidcIdriSliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.LidcIdriSliceDataset.html.md#deepinv.datasets.LidcIdriSliceDataset)           | `x`                                                          | Over 200,000 CT scan slices         | Slices 512x512 voxels                                           | A comprehensive dataset of lung CT scans with annotations, used for medical image processing and lung cancer detection research.                                                                                 |
| [`Flickr2kHR`](https://deepinv.org/api/stubs/deepinv.datasets.Flickr2kHR.html.md#deepinv.datasets.Flickr2kHR)                               | `x`                                                          | 2,650 images                        | RGB, up to 2000x2000 pixels (variable)                          | A dataset from Flickr containing high-resolution images for tasks like super-resolution and image restoration.                                                                                                   |
| [`LsdirHR`](https://deepinv.org/api/stubs/deepinv.datasets.LsdirHR.html.md#deepinv.datasets.LsdirHR)                                     | `x`                                                          | 84499 (train) + 1000 (val) images   | RGB, up to 2160x2160 pixels (variable)                          | A dataset with high-resolution images, often used for training large reconstruction models.                                                                                                                      |
| [`FMD`](https://deepinv.org/api/stubs/deepinv.datasets.FMD.html.md#deepinv.datasets.FMD)                                             | `x`                                                          | 12000 images                        | 512x512 pixels                                                  | The Fluorescence Microscopy Dataset (FMD) is a dataset of real fluorescence microscopy images.                                                                                                                   |
| [`Kohler`](https://deepinv.org/api/stubs/deepinv.datasets.Kohler.html.md#deepinv.datasets.Kohler)                                       | `(x, y)`                                                     | 48 blurry + 9547 sharp images       | 800x800 RGB                                                     | A blind-deblurring dataset consists of blurry shots and sharp frames, each blurry shot being associated with about 200 sharp frames.                                                                             |
| [`NBUDataset`](https://deepinv.org/api/stubs/deepinv.datasets.NBUDataset.html.md#deepinv.datasets.NBUDataset)                               | `x` Tensor or TensorList                                     | 510 images across 6 satellites      | Cx256x256 multispectral (C=4 or 8) and 1x1024x1024 panchromatic | Multispectral satellite images of urban scenes from 6 different satellites.                                                                                                                                      |
| [`BrainWebPET`](https://deepinv.org/api/stubs/deepinv.datasets.BrainWebPET.html.md#deepinv.datasets.BrainWebPET)                             | `(x, params)`                                                | 20 synthetic brain volumes          | 1x127x344x344 voxels                                            | Synthetic PET emission volumes with attenuation maps and optional MRI contrasts from the BrainWeb dataset.                                                                                                       |
| [`BrainWebMRI`](https://deepinv.org/api/stubs/deepinv.datasets.BrainWebMRI.html.md#deepinv.datasets.BrainWebMRI)                             | `x`                                                          | 20 MRI brain volumes                | 1x181x217x181 voxels                                            | 3D MRI volumes with T1, T2, T2\* or PD contrast.                                                                                                                                                                 |

<a id="data-transforms"></a>

## Data Transforms

We provide some torchvision-style transforms for use when loading data:

#### Data Transforms Overview

| **Transform**                                                                                                          | **Description**                                                          |
|------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| [`deepinv.datasets.utils.Rescale`](https://deepinv.org/api/stubs/deepinv.datasets.utils.Rescale.html.md#deepinv.datasets.utils.Rescale)         | Min-max or clip value rescaling.                                         |
| [`deepinv.datasets.utils.ToComplex`](https://deepinv.org/api/stubs/deepinv.datasets.utils.ToComplex.html.md#deepinv.datasets.utils.ToComplex)     | Add empty imaginary dimension to image.                                  |
| [`deepinv.datasets.utils.Crop`](https://deepinv.org/api/stubs/deepinv.datasets.utils.Crop.html.md#deepinv.datasets.utils.Crop)               | Crop image in corner or with arbitrary crop position and/or size.        |
| [`deepinv.datasets.MRISliceTransform`](https://deepinv.org/api/stubs/deepinv.datasets.MRISliceTransform.html.md#deepinv.datasets.MRISliceTransform) | Transform raw FastMRI data by simulating masks and estimating coil maps. |
