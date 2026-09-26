# DeteCTDataset

### *class* deepinv.datasets.DeteCTDataset(root, problem='full', n_angles=3600, slice_ids='all', use_dict_output=False)

Bases: [`ImageDataset`](https://deepinv.org/api/stubs/deepinv.datasets.ImageDataset.html.md#deepinv.datasets.ImageDataset)

2DeteCT dataset of 2D Computed Tomography acquisitions.

The dataset was acquired by Kiss *et al.*<sup>[1](#footcite-kiss20232detect)</sup> and used for benchmarking CT reconstruction algorithms in Kiss *et al.*<sup>[2](#footcite-kiss2025benchmarking)</sup>.
The data is industrial CT projection data (i.e. sinograms) of various materials acquired using a proprietary scanner from [Centrum Wiskunde & Informatica](https://www.cwi.nl/en/).
The samples contain materials resembling the attenuation of human anatomy; see Kiss *et al.*<sup>[1](#footcite-kiss20232detect)</sup> for more details.

The projections (shape `(1,n_angles,956)`) are preprocessed (flat/dark-corrected, log-transformed, all in PyTorch) following [LION](https://github.com/CambridgeCIA/LION)
such that the setup matches exactly Kiss *et al.*<sup>[2](#footcite-kiss2025benchmarking)</sup>, such that the dataset can be used to compare DeepInverse image reconstruction methods
with the values reported in Kiss *et al.*<sup>[2](#footcite-kiss2025benchmarking)</sup>.

Each sample is scanned 3 times: `mode1`, `mode2` and `mode3`. See below for their usage.

“Ground truth” `x` are also provided as iterative recons using all angles, of shape `(1,1024,1024)`.

To download the data from [Zenodo](https://doi.org/10.5281/zenodo.8014758),
use [`download_dataset`](#deepinv.datasets.DeteCTDataset.download_dataset), which extracts each archive into
the `2DeteCT_slicesXXXX-YYYY` (+ `_RecSeg`) subfolders in root. Note: for the test set, you only need to download slices `4001-5000`, i.e. do
`dinv.datasets.DeteCTDataset.download_dataset(root='/path/to/2DeteCT', blocks='test')`.

* **Parameters:**
  * **root** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – root dir, should contain subfolders named `2DeteCT_slicesXXXX-YYYY` (+ `_RecSeg`)
  * **problem** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – benchmarking problem from 2DeteCT.
    - `full`: `mode2` acquired data (3600 projections)
    - `sparse_view`: `mode2` acquired data then evenly subsampled
    - `limited_angle`: `mode2` acquired data then limited angles taken
    - `low_dose`: `mode1` acquired data (3W instead of 90W)
    - `beam_hardening`: `mode3` acquired data (acquired without a filter, leading to beam-hardening)
  * **n_angles** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – kept projections for sparse_view/limited_angle, defaults to 3600 (i.e. all angles).
  * **slice_ids** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – `all` (default, every slice found from 1-5000), `train`/`val`/`test` (LION 3930/550/470 sample split), or `ood` (out-of-distribution slices 5521-6370).
  * **use_dict_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return output as dict with keys “x”, “y”, “params” instead of tuple (default `False`).
* **Examples:**
  Download a single sample slice (ID 4531) from HuggingFace and load it:
  ```pycon
  >>> import shutil, deepinv as dinv
  >>> from deepinv.datasets import DeteCTDataset, download_archive
  >>> download_archive(dinv.utils.get_image_url("2DeteCT_slices_4001-5000_slice04531.zip"), "2DeteCT/data.zip", extract=True)
  >>> x, y = DeteCTDataset("2DeteCT", slice_ids="test")[0]
  >>> print(x.shape, y.shape) # (1,H,W), (1, num_angles, detector length)
  torch.Size([1, 1024, 1024]) torch.Size([1, 3600, 956])
  >>> shutil.rmtree("2DeteCT")
  ```

<hr />

* **References:**

* <a id='footcite-kiss20232detect'>**[1]**</a> Maximilian Kiss, Sophia Coban, K. Joost Batenburg, Tristan van Leeuwen, and Felix Lucka. 2detect - a large 2d expandable, trainable, experimental computed tomography dataset for machine learning. *Scientific Data*, 2023. [doi:10.1038/s41597-023-02484-6](https://doi.org/10.1038/s41597-023-02484-6).
* <a id='footcite-kiss2025benchmarking'>**[2]**</a> Maximilian Kiss, Ander Biguri, Zakhar Shumaylov, Ferdia Sherry, K. Joost Batenburg, Carola-Bibiane Schönlieb, and Felix Lucka. Benchmarking learned algorithms for computed tomography image reconstruction tasks. *Applied Mathematics for Modern Challenges*, 2025. [doi:10.3934/ammc.2025001](https://doi.org/10.3934/ammc.2025001).

#### *static* download_dataset(root, blocks='all', force_download=False)

Download and extract the 2DeteCT archives from Zenodo into `root`.

Each block’s raw data and reference reconstructions (RecSeg) are extracted into the
`2DeteCT_slicesXXXX-YYYY` (+ `_RecSeg`) subfolders expected by the dataset.

#### WARNING
The archives are very large (up to ~34GB each); `blocks="all"` needs several hundred GB of disk.

* **Parameters:**
  * **root** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – dir to download into (same `root` passed to init).
  * **blocks** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list)) – which slice ranges to download: `"all"` (slices 1-5000),
    `"test"` (only slices 4001-5000, i.e. the benchmark test set), `"ood"` (out-of-distribution slices 5521-6370),
    or a list of ranges from `["1-1000", "1001-2000", "2001-3000", "3001-4000", "4001-5000", "OOD"]`.
  * **force_download** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – re-download even if the archive already exists.

#### *static* get_astra_geometry(problem='full', n_angles=None)

Get astra object geometry and project geometry for 2DeteCT setup.

Construct geometry objects to pass to [`deepinv.physics.TomographyWithAstra`](https://deepinv.org/api/stubs/deepinv.physics.TomographyWithAstra.html.md#deepinv.physics.TomographyWithAstra)
in order to test physics-conditioned algorithms on the 2DeteCT benchmark.

The object geometry values and fan-beam projection geometry values are taken from [LION](https://github.com/CambridgeCIA/LION).

The projection geometry is defined as conebeam with one detector row.

Usage

```default
import deepinv as dinv
obj_geom, proj_geom = dinv.datasets.DeteCTDataset.get_astra_geometry()
physics = dinv.physics.TomographyWithAstra(
    object_geometry=obj_geom,
    projection_geometry=proj_geom,
    is_2d=True, # important
    normalize=True,
    device=device,
    noise_model=dinv.physics.PoissonGaussianNoise(),
)
```

* **Parameters:**
  * **problem** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – 2DeteCT benchmark problem, either “sparse_view” or “limited_angle”, for how to undersample angles.
  * **n_angles** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – for sparse_view or limited_angle, how many angles.
* **Return tuple:**
  obj_geom dict, proj_geom dict
* **Return type:**
  [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)

<a id="sphx-glr-backref-deepinv-datasets-detectdataset"></a>

## Examples using `DeteCTDataset`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="The data is taken from the 2DeteCT benchmark kiss2025benchmarking and dataset kiss20232detect, which is an industrial CT dataset of various materials acquired using a proprietary scanner from CWI (i.e. sinogram-to-image). The setup is matched exactly to kiss2025benchmarking, such that you can compare DeepInverse image reconstruction methods with the values reported in kiss2025benchmarking.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_astra_2detect_thumb.png)

[Reconstruct real CT sinograms with the 2DeteCT benchmark](https://deepinv.org/auto_examples/external-libraries/demo_astra_2detect.html.md)

  <div class="sphx-glr-thumbnail-title">Reconstruct real CT sinograms with the 2DeteCT benchmark</div>
</div>
<!-- thumbnail-parent-div-close --></div>
