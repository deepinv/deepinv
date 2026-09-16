# SimpleFastMRISliceDataset

### *class* deepinv.datasets.SimpleFastMRISliceDataset(root_dir=None, anatomy='knee', file_name=None, train=True, sample_index=None, train_percent=1.0, transform=None, download=False, use_dict_output=False)

Bases: [`ImageDataset`](https://deepinv.org/api/stubs/deepinv.datasets.ImageDataset.html.md#deepinv.datasets.ImageDataset)

Simple FastMRI image dataset.

Loads in-memory a saved and processed subset of 2D slices from the full FastMRI slice dataset of Knoll *et al.*<sup>[1](#footcite-knoll2020advancing)</sup>, for quick loading.

#### IMPORTANT
By using this dataset, you confirm that you have agreed to and signed the [FastMRI data use agreement](https://fastmri.med.nyu.edu/).

These datasets are generated using [`deepinv.datasets.FastMRISliceDataset.save_simple_dataset()`](https://deepinv.org/api/stubs/deepinv.datasets.FastMRISliceDataset.html.md#deepinv.datasets.FastMRISliceDataset.save_simple_dataset).
You can use this to generate your own custom dataset and load using the `file_name` argument.

We provide a pregenerated mini saved subset for singlecoil FastMRI knees (total 2 images)
and RSS reconstructions of multicoil brains (total 2 images).
These originate from their respective fully-sampled volumes converted to images via root-sum-of-square (RSS).
Each slice is the middle slice from one independent volume.
The images are of shape (2x320x320) and are normalized per-sample (0-1) and padded.
Download the dataset using `download=True`, and load them using the `anatomy` argument.

#### NOTE
Since images are obtained from RSS, the imaginary part of each sample is 0.

<hr />

* **Examples:**
  Load mini demo knee dataset:
  ```pycon
  >>> from deepinv.datasets import SimpleFastMRISliceDataset
  >>> from deepinv.utils import get_cache_home
  >>> dataset = SimpleFastMRISliceDataset(get_cache_home(), anatomy="knee", download=True)
  >>> len(dataset)
  2
  ```
* **Parameters:**
  * **root_dir** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – dataset root directory
  * **anatomy** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – load either fastmri “knee” or “brain” slice datasets.
  * **file_name** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – optional, name of local dataset to load, overrides `anatomy`. If `None`, load dataset based on `anatomy` parameter.
  * **train** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to use training set or test set, defaults to True
  * **sample_index** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – if specified only load this sample, defaults to None
  * **train_percent** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – percentage train for train/test split, defaults to 1.
  * **transform** (*Callable*) – optional transform for images, defaults to None
  * **download** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, downloads the dataset from the internet and puts it in root directory.
    If dataset is already downloaded, it is not downloaded again. Default at False.
  * **use_dict_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return output as dict with keys “x”, “y”, “params” instead of tuple (default `False`).

<hr />

* **References:**

* <a id='footcite-knoll2020advancing'>**[1]**</a> Florian Knoll, Tullie Murrell, Anuroop Sriram, Nafissa Yakubova, Jure Zbontar, Michael Rabbat, Aaron Defazio, Matthew J Muckley, Daniel K Sodickson, C Lawrence Zitnick, and others. Advancing machine learning for mr image reconstruction with an open competition: overview of the 2019 fastmri challenge. *Magnetic resonance in medicine*, 84(6):3054–3070, 2020.

<a id="sphx-glr-backref-deepinv-datasets-simplefastmrislicedataset"></a>

## Examples using `SimpleFastMRISliceDataset`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to get started with DeepInverse in under 5 minutes.">  <div class="sphx-glr-thumbnail-title">5 minute quickstart tutorial</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the self-supervised Artifact2Artifact loss for solving an undersampled sequential MRI reconstruction problem without ground truth.">  <div class="sphx-glr-thumbnail-title">Self-supervised MRI reconstruction with Artifact2Artifact</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a reconstruction network for an MRI inverse problem on a fully self-supervised way, i.e., using measurement data only.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Imaging for MRI.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
