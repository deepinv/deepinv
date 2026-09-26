# CalgarySliceDataset

### *class* deepinv.datasets.CalgarySliceDataset(root, transform=None, \*\*kwargs)

Bases: [`FastMRISliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.FastMRISliceDataset.html.md#deepinv.datasets.FastMRISliceDataset)

Dataset for [Calgary-Campinas](https://sites.google.com/view/calgary-campinas-dataset) 12-coil raw brain kspace.

Loads Calgary `h5` volumes of shape `(num_slices, H, W, 2N)`, where slice dim is in image domain and `H,W` is kspace.
The dataset loads and preprocesses all kspace slices per volume, of shape `(2, N, H, W)`. These are fully-sampled for train/val volumes and masked for the test set.

Also computes the GT `x`, the magnitude root-sum-square reconstructions of shape `(1, H, W)`, or `torch.nan` for the masked test set.

The dataset is loaded as a dict with keys `'x', 'y', 'params'` when `use_dict_output=True` (default) or tuples `(x, y, params)` when `False`, where
`params` optionally contains the sampling `mask` and, if desired, estimated `coil_maps`.

#### NOTE
The test set comes already masked, which [`deepinv.datasets.CalgarySliceTransform`](https://deepinv.org/api/stubs/deepinv.datasets.CalgarySliceTransform.html.md#deepinv.datasets.CalgarySliceTransform) estimates. For the validation set, the data
is fully-sampled. You can simulate masked data using precomputed Poisson-disk masks as follows

```default
mask_file = f"R{acceleration}_{y.shape[-2]}x{y.shape[-1]}.npy"
torch.hub.download_url_to_file(f"https://huggingface.co/datasets/NKI-AI/direct-mri-masks/resolve/main/calgary_campinas_masks/{mask_file.name}", str(mask_file))
masks = np.load(mask_file) # (100, H, W) bool
mask = torch.from_numpy(masks[0]).float().unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
y *= mask
```

Calgary kspace uses the opposite centering convention to deepinv, so it is converted here (a half-FOV checkerboard shift)
so that `y` works directly with [`deepinv.physics.MultiCoilMRI`](https://deepinv.org/api/stubs/deepinv.physics.MultiCoilMRI.html.md#deepinv.physics.MultiCoilMRI).

* **Parameters:**
  * **root** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – path to the dataset.
  * **transform** (*Callable*) – transform taking `(target, kspace)`, defaults to [`deepinv.datasets.CalgarySliceTransform`](https://deepinv.org/api/stubs/deepinv.datasets.CalgarySliceTransform.html.md#deepinv.datasets.CalgarySliceTransform).
  * **kwargs** – passed to [`deepinv.datasets.FastMRISliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.FastMRISliceDataset.html.md#deepinv.datasets.FastMRISliceDataset) (e.g. `slice_index`, `filter_id`, metadata cache, `use_dict_output`).

<hr />

* **Examples:**
  Download a Calgary test volume and load its middle slice:
  ```pycon
  >>> import deepinv as dinv
  >>> from deepinv.datasets import CalgarySliceDataset, download_archive
  >>> root = dinv.utils.get_cache_home() / "calgary"
  >>> download_archive(dinv.utils.get_image_url("demo_calgary_test_e13991s3_P01536.7.h5"), root / "vol.h5")
  >>> batch = CalgarySliceDataset(root, slice_index="middle", use_dict_output=True)[0]
  >>> batch['y'].shape  # (2, N, H, W) multicoil k-space
  torch.Size([2, 12, 218, 170])
  ```

<a id="sphx-glr-backref-deepinv-datasets-calgaryslicedataset"></a>

## Examples using `CalgarySliceDataset`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo reconstructs undersampled k-space data for 2D cardiac and brain MRI on:">![](auto_examples/models/images/thumb/sphx_glr_demo_mri_pretrained_thumb.png)

[Reconstruct undersampled k-space for cardiac and brain MRI](https://deepinv.org/auto_examples/models/demo_mri_pretrained.html.md)

  <div class="sphx-glr-thumbnail-title">Reconstruct undersampled k-space for cardiac and brain MRI</div>
</div>
<!-- thumbnail-parent-div-close --></div>
