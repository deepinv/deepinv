# BrainWebPET

### *class* deepinv.datasets.BrainWebPET(root=None, subject_ids=None, download=True, transform=None, pet_class=None, contrast=(), random_degradations_kwargs=None, lesion_diameters=None, lesion_kwargs=None, seed=0, use_dict_output=False)

Bases: [`ImageDataset`](https://deepinv.org/api/stubs/deepinv.datasets.ImageDataset.html.md#deepinv.datasets.ImageDataset)

BrainWeb PET phantoms.

Loads synthetic 3D volumes from BrainWeb dataset <sup>[1](#footcite-collinsdesignconstructionrealistic1998)</sup>,
of shape `(1, 127, 344, 344)`.
The dataset has been adapted to emission tomography, and returns an emission and attenuation map, at the Siemens Biograph mMR isotropic resolution of 2.0863 mm per voxel.

Passing `lesion_diameters` adds high activity lesions with `brainweb.add_lesions` and includes a `lesion_mask` in the returned params, where the background is labelled `0` and lesions are labelled from `1` onwards.

This dataset relies on the original implementation of Casper da Costa-Luis:
<[https://github.com/casperdcl/brainweb](https://github.com/casperdcl/brainweb)>\`_. Install it with `pip install brainweb`.
See the original implementation for a detailed description of the keyword arguments.

#### NOTE
For a version of this dataset dedicated to magnetic resonance imaging, which contains
more contrast options, see [`deepinv.datasets.BrainWebMRI`](https://deepinv.org/api/stubs/deepinv.datasets.BrainWebMRI.html.md#deepinv.datasets.BrainWebMRI).

* **Parameters:**
  * **root** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path) *,* *None*) – Dataset directory. Defaults to the DeepInv cache.
  * **subject_ids** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*collections.abc.Sequence*](https://docs.python.org/3.9/library/collections.abc.html#collections.abc.Sequence) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]* *,* *None*) – Subjects to include in the dataset. Defaults to `None` which includes all subjects.
  * **download** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Download missing subjects. Defaults to `True`.
  * **pet_class** ([*type*](https://docs.python.org/3.9/library/functions.html#type) *[**brainweb.Act* *]* *,* *None*) – BrainWeb PET activity preset. Defaults to `brainweb.FDG`.
  * **lesion_diameters** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *]* *,* *None*) – Lesion diameters in mm. Defaults to `None`, which adds no lesions.
  * **contrast** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*collections.abc.Sequence*](https://docs.python.org/3.9/library/collections.abc.html#collections.abc.Sequence) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *]*) – Contrasts to include in the returned parameters. Valid values are `"T1"` and `"T2"`. Defaults to an empty tuple.
  * **lesion_kwargs** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict) *,* *None*) – Keyword arguments for `brainweb.add_lesions`.
  * **random_degradations_kwargs** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict) *,* *None*) – Keyword arguments for `brainweb.get_mmr_fromfile` controlling random structural degradations.
  * **transform** ([*collections.abc.Callable*](https://docs.python.org/3.9/library/collections.abc.html#collections.abc.Callable) *,* *None*) – Optional transform to apply to the returned volumes.
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* *None*) – Seed used when adding random lesions.
  * **use_dict_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return output as dict with keys “x”, “y”, “params” instead of tuple (default `False`).

<hr />

* **Example:**

```pycon
>>> import brainweb
>>> import numpy as np
>>> from deepinv.datasets import BrainWebPET
>>> class RandomFDG(brainweb.FDG):
...     greyMatter = lambda: np.random.normal(128, 8)
>>> dataset = BrainWebPET(
...     root="data/brainweb_pet",
...     random_degradations_kwargs={"petNoise": 0.5, "petSigma": 2},
...     contrast=["T1", "T2"],
...     pet_class=RandomFDG,
...     lesion_diameters=[15, 7],
...     lesion_kwargs={"intensity": [200, 150], "blur": [0, 0], "thresh": 30},
... )
>>> emission, params = dataset[0]
>>> emission.shape == params["attenuation"].shape
True
```

<hr />

* **References:**

* <a id='footcite-collinsdesignconstructionrealistic1998'>**[1]**</a> D. L. Collins, A. P. Zijdenbos, V. Kollokian, J. G. Sled, N. J. Kabani, C. J. Holmes, and A. C. Evans. Design and construction of a realistic digital brain phantom. *IEEE transactions on medical imaging*, 17(3):463–468, June 1998. [doi:10.1109/42.712135](https://doi.org/10.1109/42.712135).
