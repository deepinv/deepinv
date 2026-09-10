# CMRxReconSliceDataset

### *class* deepinv.datasets.CMRxReconSliceDataset(root=None, data_dir='SingleCoil/Cine/TrainingSet/FullSample', load_metadata_from_cache=False, save_metadata_to_cache=False, metadata_cache_file='dataset_cache.pkl', apply_mask=True, mask_dir='SingleCoil/Cine/TrainingSet/AccFactor04', mask_generator=None, transform=None, pad_size=(512, 256), noise_model=None, use_dict_output=False)

Bases: [`FastMRISliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.FastMRISliceDataset.html.md#deepinv.datasets.FastMRISliceDataset), [`MRIMixin`](https://deepinv.org/api/stubs/deepinv.utils.MRIMixin.html.md#deepinv.utils.MRIMixin)

CMRxRecon dynamic MRI dataset.

Wrapper for dynamic 2D+t MRI dataset from the [CMRxRecon 2023 challenge](https://cmrxrecon.github.io/).

The dataset returns sequences of long axis (`lax`) views and short axis (`sax`) slices along with 2D+t acceleration masks.

Return tuples `(x, y)` of target (ground truth) and kspace (measurements).

Optionally apply mask to measurements to get undersampled measurements.
Then the dataset returns tuples `(x, y, params)` where `params` is a dict `{'mask': mask}`.
This can be directly used with [`deepinv.Trainer`](https://deepinv.org/api/stubs/deepinv.Trainer.html.md#deepinv.Trainer) to train with undersampled measurements.
If masks are present in the data folders (in file format `cine_xax_mask.mat`) then these will be loaded.
If not, unique masks will be generated using a `mask_generator`, for example [`deepinv.physics.generator.RandomMaskGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.RandomMaskGenerator.html.md#deepinv.physics.generator.RandomMaskGenerator).

While the usual workflow in deepinv is for the dataset to return only ground truth `x` and the user
generates a measurement dataset using [`deepinv.datasets.generate_dataset()`](https://deepinv.org/api/stubs/deepinv.datasets.generate_dataset.html.md#deepinv.datasets.generate_dataset), here we compute the
measurements inside the dataset (and return a triplet `x, y, params` where `params` contains the mask)
because of the variable size of the data before padding, in line with the original CMRxRecon code.

#### NOTE
The data returned is directly compatible with [`deepinv.physics.DynamicMRI`](https://deepinv.org/api/stubs/deepinv.physics.DynamicMRI.html.md#deepinv.physics.DynamicMRI).
See [Tour of MRI functionality in DeepInverse](https://deepinv.org/auto_examples/physics/demo_mri_tour.html.md#sphx-glr-auto-examples-physics-demo-mri-tour-py) for example using this dataset.

We provide one single downloadable demo sample, see example below on how to use this.
Otherwise, download the full dataset from the [challenge website](https://cmrxrecon.github.io/).

**Raw data file structure:**

```default
root_dir --- data_dir --- P001 --- cine_lax.mat
          |            |        |
          |            |        -- cine_sax.mat
          |            -- PXXX
          -- mask_dir --- P001 --- cine_lax_mask.mat
                       |        |
                       |        -- cine_sax_mask.mat
                       -- PXXX
```

<hr />

Example:

```pycon
>>> from deepinv.datasets import CMRxReconSliceDataset, download_archive
>>> from deepinv.utils import get_image_url, get_cache_home
>>> from torch.utils.data import DataLoader
>>> download_archive(
...     get_image_url("CMRxRecon.zip"),
...     get_cache_home() / "CMRxRecon.zip",
...     extract=True,
... )
>>> dataset = CMRxReconSliceDataset(get_cache_home() / "CMRxRecon")
>>> x, y, params = next(iter(DataLoader(dataset)))
>>> x.shape # (B, C, T, H, W)
torch.Size([1, 2, 12, 512, 256])
>>> y.shape # (B, C, T, H, W)
torch.Size([1, 2, 12, 512, 256])
>>> 1 / params["mask"].mean()  # Approx 4x acceleration
tensor(4.2402)
```

* **Parameters:**
  * **root** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – path for dataset root folder.
  * **data_dir** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – directory containing target (ground truth) data, defaults to ‘SingleCoil/Cine/TrainingSet/FullSample’ which is default CMRxRecon folder structure
  * **load_metadata_from_cache** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – \_description_, defaults to False
  * **save_metadata_to_cache** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – \_description_, defaults to False
  * **metadata_cache_file** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – \_description_, defaults to “dataset_cache.pkl”
  * **apply_mask** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, mask is applied to subsample the kspace using a mask either
    loaded from `data_folder` or generated using `mask_generator`. If `False`, the mask of ones is used.
  * **mask_dir** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – dataset folder containing predefined acceleration masks. Defaults to the 4x acc. mask folder
    according to the CMRxRecon folder structure. To use masks, `apply_mask` must be `True`.
  * **mask_generator** ([*deepinv.physics.generator.BaseMaskGenerator*](https://deepinv.org/api/stubs/deepinv.physics.generator.BaseMaskGenerator.html.md#deepinv.physics.generator.BaseMaskGenerator)) – optional mask generator to randomly generate acceleration masks
    to apply to unpadded kspace. If specified, `mask_dir` must be `None` and `apply_mask` must be `True`.
  * **transform** (*Callable*) – optional transform to apply to the target image sequences before padding or physics is applied.
  * **pad_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – tuple of 2 ints (W, H) for all images to be padded to, if `None`, no padding.
  * **noise_model** ([*deepinv.physics.NoiseModel*](https://deepinv.org/api/stubs/deepinv.physics.NoiseModel.html.md#deepinv.physics.NoiseModel)) – optional noise model to apply to unpadded kspace.
  * **use_dict_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return output as dict with keys “x”, “y”, “params” instead of tuple (default `False`).

<a id="sphx-glr-backref-deepinv-datasets-cmrxreconslicedataset"></a>

## Examples using `CMRxReconSliceDataset`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div>
<!-- thumbnail-parent-div-close --></div>
