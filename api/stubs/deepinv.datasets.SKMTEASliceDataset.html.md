# SKMTEASliceDataset

### *class* deepinv.datasets.SKMTEASliceDataset(root=None, echo=0, acc=6, load_metadata_from_cache=False, save_metadata_to_cache=False, metadata_cache_file='skmtea_dataset_cache.pkl', filter_id=None, use_dict_output=False)

Bases: [`FastMRISliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.FastMRISliceDataset.html.md#deepinv.datasets.FastMRISliceDataset), [`MRIMixin`](https://deepinv.org/api/stubs/deepinv.utils.MRIMixin.html.md#deepinv.utils.MRIMixin)

SKM-TEA dataset for raw multicoil MRI kspace data.

Wraps the SKM-TEA dataset proposed in Desai *et al.*<sup>[1](#footcite-desai2021skm)</sup>.
The dataset returns 2D slices from a dataset of 3D MRI volumes.

To download raw data as `h5` files, see the [SKM-TEA website](https://github.com/StanfordMIMI/skm-tea).

The dataset is loaded as tuples `(x, y, params)` where:

* `y` are the undersampled kspace measurements of shape `(2, N, H, W)` where N is the coil dimension.
* `x` are the complex SENSE reconstructions from fully-sampled kspace of shape `(2, H, W)`.
* `params` is a dict containing parameters `mask` and `coil_maps` provided by the dataset, where `mask` are
  elliptical Poisson disc undersampling masks and `coil_maps` are sensitivity maps estimated using JSENSE.

#### TIP
The data can be directly related with [`deepinv.physics.MultiCoilMRI`](https://deepinv.org/api/stubs/deepinv.physics.MultiCoilMRI.html.md#deepinv.physics.MultiCoilMRI)

```default
x, y, params = next(iter(DataLoader(SKMTEADataset())))
from deepinv.physics import MultiCoilMRI
physics = MultiCoilMRI(**params)
y1 = physics(x)
```

Then `y` and `y1` are almost identical.

**Raw data file structure:** (each file contains the k-space data and some metadata related to the scan)

```default
self.root --- xxx0.h5
           |
           -- xxx1.h5.
```

When using this class, consider using the `metadata_cache` options to speed up class initialisation after the first initialisation.

* **Parameters:**
  * **root** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – Path to the dataset.
  * **echo** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – which qDESS echo to use, defaults to 0.
  * **acc** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – acceleration of mask to load, choose from 4, 6, 8, 10, 12 or 16.
  * **load_metadata_from_cache** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Whether to load dataset metadata from cache.
  * **save_metadata_to_cache** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Whether to cache dataset metadata.
  * **metadata_cache_file** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – A file used to cache dataset information for faster load times.
  * **filter_id** (*Callable*) – optional function that takes `SliceSampleID` named tuple and returns whether this id should be included.
  * **use_dict_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return output as dict with keys “x”, “y”, “params” instead of tuple (default `False`).

<hr />

* **Examples:**
  Load data:
  ```pycon
  >>> from deepinv.datasets import SKMTEASliceDataset
  >>> from torch.utils.data import DataLoader
  >>> dataset = SKMTEASliceDataset(".")
  >>> len(dataset)
  512
  >>> x, y, params = next(iter(DataLoader(dataset)))
  >>> x.shape # (B, 2, H, W)
  torch.Size([1, 2, 512, 160])
  >>> y.shape # (B, 2, N, H, W)
  torch.Size([1, 2, 8, 512, 160])
  ```

<hr />

* **References:**

* <a id='footcite-desai2021skm'>**[1]**</a> Arjun D Desai, Andrew M Schmidt, Elka B Rubin, Christopher Michael Sandino, Marianne Susan Black, Valentina Mazzoli, Kathryn J Stevens, Robert Boutin, Christopher Re, Garry E Gold, and others. Skm-tea: a dataset for accelerated mri reconstruction with dense image labels for quantitative clinical evaluation. In *Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)*. 2021.

#### zero_pad(x, shape, mode='constant', value=0)

Perform zero padding.

Code taken from [https://github.com/ad12/meddlr/blob/main/meddlr/ops/utils.py#L38](https://github.com/ad12/meddlr/blob/main/meddlr/ops/utils.py#L38)
