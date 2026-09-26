# BrainWebMRI

### *class* deepinv.datasets.BrainWebMRI(subject_ids=None, contrast='T1', download=True, root=None, transform=None, use_dict_output=False)

Bases: [`ImageDataset`](https://deepinv.org/api/stubs/deepinv.datasets.ImageDataset.html.md#deepinv.datasets.ImageDataset)

Dataset for [BrainWeb](https://brainweb.bic.mni.mcgill.ca/).

BrainWeb brain phantom for Magnetic Resonance Imaging (MRI) research <sup>[1](#footcite-collinsdesignconstructionrealistic1998)</sup>.
The dataset consists of 22 MRI brain phantom scans: 21 normal brains and 1 multiple sclerosis brain (patient 1).
Several contrasts are available for each patient: T1, T2, T2\* and PD.
Each T1 volume has shape (1, 181, 256, 256) and is scaled by 1 / 4095 to that it is normalized with values in [0, 1].

This dataset relies on the original implementation of Pierre-Antoine Comby:
<[https://github.com/paquiteau/brainweb-dl](https://github.com/paquiteau/brainweb-dl)>\`_. Install it with `pip install brainweb-dl`.

#### NOTE
For a version of this dataset with dedicated features for emission tomography, such
as emission / attenuation maps and hot lesions, see [`deepinv.datasets.BrainWebPET`](https://deepinv.org/api/stubs/deepinv.datasets.BrainWebPET.html.md#deepinv.datasets.BrainWebPET).

* **Parameters:**
  * **subject_ids** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*collections.abc.Sequence*](https://docs.python.org/3.9/library/collections.abc.html#collections.abc.Sequence) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]* *,* *None*) – Subjects to include in the dataset. Possible values: [4, 5, 6, 18, 20, 38, 41-54]. Defaults to all subjects.
  * **contrast** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – MRI contrast to return: `"T1"`, `"T2"`, `"T2*"` or `"PD"`. Defaults to `"T1"`.
  * **download** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Download missing subjects. Defaults to `True`.
  * **transform** ([*collections.abc.Callable*](https://docs.python.org/3.9/library/collections.abc.html#collections.abc.Callable) *,* *None*) – Optional volume transform.
  * **root** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path) *,* *None*) – Root directory of dataset. Directory path from where we load and save the dataset.
  * **use_dict_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return output as dict with keys “x”, “y”, “params” instead of tuple (default `False`).

<hr />

* **References:**

* <a id='footcite-collinsdesignconstructionrealistic1998'>**[1]**</a> D. L. Collins, A. P. Zijdenbos, V. Kollokian, J. G. Sled, N. J. Kabani, C. J. Holmes, and A. C. Evans. Design and construction of a realistic digital brain phantom. *IEEE transactions on medical imaging*, 17(3):463–468, June 1998. [doi:10.1109/42.712135](https://doi.org/10.1109/42.712135).
