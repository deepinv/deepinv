# FastMRISliceDataset

### *class* deepinv.datasets.FastMRISliceDataset(root=None, target_root=None, load_metadata_from_cache=False, save_metadata_to_cache=False, metadata_cache_file='dataset_cache.pkl', slice_index='all', subsample_volumes=1.0, transform=None, filter_id=None, rng=None, use_dict_output=False)

Bases: [`ImageDataset`](https://deepinv.org/api/stubs/deepinv.datasets.ImageDataset.html.md#deepinv.datasets.ImageDataset), [`MRIMixin`](https://deepinv.org/api/stubs/deepinv.utils.MRIMixin.html.md#deepinv.utils.MRIMixin)

Dataset for [fastMRI](https://fastmri.med.nyu.edu/) that provides access to raw MR kspace data.

This dataset (from Knoll *et al.*<sup>[1](#footcite-knoll2020advancing)</sup>) randomly selects 2D slices from a dataset of 3D MRI volumes.
This class considers one data sample as one slice of a MRI scan, thus slices of the same MRI scan are considered independently in the dataset.

To download raw data, please go to the bottom of the page `https://fastmri.med.nyu.edu/` to download the brain/knee and train/validation/test volumes as `h5` files.

The dataset is loaded as tuples `(x, y, params)` where:

* `y` are the kspace measurements of shape `(2, (N,) H, W)` where N is the optional coil dimension depending on whether the data is singlecoil or multicoil.
  Note this kspace will be fully-sampled for training/validation datasets, and will be masked for test/challenge sets.
* `x` (“target”) are the (cropped) magnitude root-sum-square reconstructions of shape `(1, H, W)`.
  If target is not present in the data (i.e. challenge/test set), then `x` will be returned as `torch.nan`. Optionally set `target_root` to load targets from a different directory.
* `params` is a dict containing parameters `mask` and/or `coil_maps`.
  Note `mask` will be automatically loaded if it is present (i.e. challenge/test set).
  Otherwise, you can generate masks and/or estimate coil maps using [`deepinv.datasets.fastmri.MRISliceTransform`](https://deepinv.org/api/stubs/deepinv.datasets.MRISliceTransform.html.md#deepinv.datasets.MRISliceTransform).

#### TIP
`x` and `y` are related by [`deepinv.physics.MRI.A_adjoint()`](https://deepinv.org/api/stubs/deepinv.physics.MRI.html.md#deepinv.physics.MRI.A_adjoint) or [`deepinv.physics.MultiCoilMRI.A_adjoint()`](https://deepinv.org/api/stubs/deepinv.physics.MultiCoilMRI.html.md#deepinv.physics.MultiCoilMRI.A_adjoint)
depending on if `y` are multicoil or not, with `crop=True, rss=True`.

**Raw data file structure:** (each file contains the k-space data and some metadata related to the scan)

```default
self.root --- file1000005.h5
           |
           -- xxxxxxxxxxx.h5.
```

When using this class, consider using the `metadata_cache` options to speed up class initialisation after the first initialisation.

#### NOTE
We also provide a simple FastMRI dataset class in [`deepinv.datasets.fastmri.SimpleFastMRISliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.SimpleFastMRISliceDataset.html.md#deepinv.datasets.SimpleFastMRISliceDataset).
This allows you to save and load the dataset as 2D singlecoil slices much faster and all in-memory.
You can generate this using the method `save_simple_dataset`.

#### IMPORTANT
By using this dataset, you confirm that you have agreed to and signed the [FastMRI data use agreement](https://fastmri.med.nyu.edu/).

See the [fastMRI README](https://github.com/facebookresearch/fastMRI/blob/main/fastmri/data/README.md) for more details.

* **Parameters:**
  * **root** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – Path to the dataset.
  * **target_root** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – if specified, reads targets from files from this folder rather than root, assuming identical file structure. Defaults to None.
  * **load_metadata_from_cache** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Whether to load dataset metadata from cache.
  * **save_metadata_to_cache** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Whether to cache dataset metadata.
  * **metadata_cache_file** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – A file used to cache dataset information for faster load times.
  * **subsample_volumes** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – (optional) proportion of volumes to be randomly subsampled (float between 0 and 1).
  * **slice_index** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – if `"all"`, keep all slices per volume, if `int`, keep only that indexed slice per volume,
    if `int` or `tuple[int]`, index those slices, if `"middle"`, keep the middle slice, if `"middle+i"`, keep $2i+1$ about
    middle slice, if `"random"`, select random slice. Defaults to `"all"`.
  * **transform** (*Callable*) – optional transform function taking in (multicoil) kspace of shape (2, (N,) H, W) and targets of shape (1, H, W).
    Defaults to [`deepinv.datasets.fastmri.MRISliceTransform`](https://deepinv.org/api/stubs/deepinv.datasets.MRISliceTransform.html.md#deepinv.datasets.MRISliceTransform).

#### SEE ALSO
[`deepinv.datasets.MRISliceTransform`](https://deepinv.org/api/stubs/deepinv.datasets.MRISliceTransform.html.md#deepinv.datasets.MRISliceTransform)
: Transform for working with raw data: simulate masks and estimate coil maps.

* **Parameters:**
  * **filter_id** (*Callable*) – optional function that takes `SliceSampleID` named tuple and returns whether this id should be included.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator) *,* *None*) – optional torch random generator for shuffle slice indices
  * **use_dict_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return output as dict with keys “x”, “y”, “params” instead of tuple (default `False`).

<hr />

For examples using raw data, see [Tour of MRI functionality in DeepInverse](https://deepinv.org/auto_examples/physics/demo_mri_tour.html.md#sphx-glr-auto-examples-physics-demo-mri-tour-py).

* **Examples:**
  Instantiate dataset with sample data (from a demo multicoil brain volume):
  ```pycon
  >>> from deepinv.datasets import FastMRISliceDataset, download_archive
  >>> from deepinv.utils import get_image_url, get_cache_home
  >>> url = get_image_url("demo_fastmri_brain_multicoil.h5")
  >>> root = get_cache_home() / "fastmri" / "brain"
  >>> download_archive(url, root / "demo.h5")
  >>> dataset = FastMRISliceDataset(root=root, slice_index="all")
  >>> len(dataset)
  16
  >>> target, kspace = dataset[0]
  >>> target.shape # (1, W, W), varies per sample
  torch.Size([1, 213, 213])
  >>> kspace.shape # (2, N, H, W), varies per sample
  torch.Size([2, 4, 512, 213])
  ```

  Load one slice per volume:
  ```pycon
  >>> dataset = FastMRISliceDataset(root=root, slice_index=0)
  ```

  Use MRI transform to mask, estimate sensitivity maps, normalize and/or crop:
  ```pycon
  >>> from deepinv.datasets import MRISliceTransform
  >>> from deepinv.physics.generator import GaussianMaskGenerator
  >>> mask_generator = GaussianMaskGenerator((512, 213))
  >>> dataset = FastMRISliceDataset(root, transform=MRISliceTransform(mask_generator=mask_generator, estimate_coil_maps=True))
  >>> target, kspace, params = dataset[0]
  >>> params["mask"].shape
  torch.Size([1, 512, 213])
  >>> params["coil_maps"].shape
  torch.Size([4, 512, 213])
  ```

  Filter by volume ID:
  ```pycon
  >>> dataset = FastMRISliceDataset(root, filter_id=lambda s: "brain" in str(s.fname))
  >>> len(dataset)
  16
  ```

  Convert to a simple normalized padded in-memory slice dataset from the middle slices only:
  ```pycon
  >>> simple_set = FastMRISliceDataset(root=root, slice_index="middle").save_simple_dataset(root.parent / "simple_set.pt")
  >>> len(simple_set)
  1
  ```

  Instantiate dataset with metadata cache (speeds up subsequent instantiation):
  ```pycon
  >>> dataset = FastMRISliceDataset(root=root, load_metadata_from_cache=True, save_metadata_to_cache=True, metadata_cache_file=root.parent / "cache.pkl")
  Saving dataset cache to ...
  >>> import shutil; shutil.rmtree(root.parent)
  ```

<hr />

* **References:**

* <a id='footcite-knoll2020advancing'>**[1]**</a> Florian Knoll, Tullie Murrell, Anuroop Sriram, Nafissa Yakubova, Jure Zbontar, Michael Rabbat, Aaron Defazio, Matthew J Muckley, Daniel K Sodickson, C Lawrence Zitnick, and others. Advancing machine learning for mr image reconstruction with an open competition: overview of the 2019 fastmri challenge. *Magnetic resonance in medicine*, 84(6):3054–3070, 2020.

#### *class* SliceSampleID(fname, slice_ind, metadata)

Bases: [`NamedTuple`](https://docs.python.org/3.9/library/typing.html#typing.NamedTuple)

Data structure containing ID and metadata of specific slices within MRI data files.

#### fname *: [Path](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)*

Alias for field number 0

#### metadata *: [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)[[str](https://docs.python.org/3.9/library/stdtypes.html#str), [Any](https://docs.python.org/3.9/library/typing.html#typing.Any)]*

Alias for field number 2

#### slice_ind *: [int](https://docs.python.org/3.9/library/functions.html#int)*

Alias for field number 1

#### metadata_cache_manager(root, samples)

Read/write metadata cache file for populating list of sample ids.

* **Parameters:**
  * **root** (*Union* *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path) *]*) – root dir to save to metadata cache
  * **samples** (*Any*) – iterable (list, dict etc.) for populating with samples to read/write to metadata cache
* **Yield:**
  samples, either populated from metadata cache, or blank, to be yielded to be written to.

#### save_simple_dataset(dataset_path, pad_to_size=(320, 320), to_complex=False)

Convert dataset to a 2D singlecoil dataset and save as pickle file.

This allows the dataset to be loaded in memory with [`deepinv.datasets.fastmri.SimpleFastMRISliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.SimpleFastMRISliceDataset.html.md#deepinv.datasets.SimpleFastMRISliceDataset).

* **Example:**
  Load local brain dataset and convert to simple dataset
  ```default
  from deepinv.datasets import FastMRISliceDataset
  root = "/path/to/dataset/fastMRI/brain/multicoil_train"
  dataset = FastMRISliceDataset(root=root, slice_index="middle")
  subset = dataset.save_simple_dataset(root + "/fastmri_brain_singlecoil.pt")
  ```
* **Parameters:**
  * **dataset_path** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – desired path of dataset to be saved with file extension e.g. `fastmri_knee_singlecoil.pt`.
  * **pad_to_size** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if not None, normalize images to 0-1 then pad to provided shape. Must be set if images are of varying size,
    in order to successfully stack images to tensor.
* **Returns:**
  loaded SimpleFastMRISliceDataset
* **Return type:**
  [SimpleFastMRISliceDataset](https://deepinv.org/api/stubs/deepinv.datasets.SimpleFastMRISliceDataset.html.md#deepinv.datasets.SimpleFastMRISliceDataset)

#### *static* torch_shuffle(x, generator=None)

Shuffle list reproducibly using torch generator.

* **Parameters:**
  * **x** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list)) – list to be shuffled
  * **generator** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – torch Generator.
* **Return list:**
  shuffled list
* **Return type:**
  [list](https://docs.python.org/3.9/library/stdtypes.html#list)

<a id="sphx-glr-backref-deepinv-datasets-fastmrislicedataset"></a>

## Examples using `FastMRISliceDataset`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the self-supervised Artifact2Artifact loss for solving an undersampled sequential MRI reconstruction problem without ground truth.">  <div class="sphx-glr-thumbnail-title">Self-supervised MRI reconstruction with Artifact2Artifact</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a reconstruction network for an MRI inverse problem on a fully self-supervised way, i.e., using measurement data only.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Imaging for MRI.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate scan-specific self-supervised learning, that is, learning to reconstruct MRI scans from a single accelerated sample without ground truth.">  <div class="sphx-glr-thumbnail-title">Scan-specific zero-shot SSDU for MRI</div>
</div>
<!-- thumbnail-parent-div-close --></div>
