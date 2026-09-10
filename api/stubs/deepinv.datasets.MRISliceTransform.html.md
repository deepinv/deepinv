# MRISliceTransform

### *class* deepinv.datasets.MRISliceTransform(mask_generator=None, seed_mask_generator=True, estimate_coil_maps=False, acs=None, espirit_crop=0.95, prewhiten=False, normalize=False)

Bases: [`MRIMixin`](https://deepinv.org/api/stubs/deepinv.utils.MRIMixin.html.md#deepinv.utils.MRIMixin)

FastMRI raw data preprocessing.

Preprocess raw kspace data:

* Optionally prewhiten kspace
* Optionally normalize kspace
* Optionally generate mask/load existing mask (i.e. for challenge/test sets)
* Optionally estimate coil maps (applicable only when using with [`multi-coil MRI physics`](https://deepinv.org/api/stubs/deepinv.physics.MultiCoilMRI.html.md#deepinv.physics.MultiCoilMRI)).

To be used with [`deepinv.datasets.FastMRISliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.FastMRISliceDataset.html.md#deepinv.datasets.FastMRISliceDataset). See below for input and output shapes.

* **Parameters:**
  * **mask_generator** ([*deepinv.physics.generator.BaseMaskGenerator*](https://deepinv.org/api/stubs/deepinv.physics.generator.BaseMaskGenerator.html.md#deepinv.physics.generator.BaseMaskGenerator)) – optional mask generator for simulating masked measurements retrospectively.
  * **seed_mask_generator** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, generated mask for given kspace is **always** the same.
    This should be `True` for test set. For supervised training, set to `False` for higher diversity in kspace undersampling.
  * **estimate_coil_maps** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int)) – if `True`, estimate coil maps using [`deepinv.physics.MultiCoilMRI.estimate_coil_maps()`](https://deepinv.org/api/stubs/deepinv.physics.MultiCoilMRI.html.md#deepinv.physics.MultiCoilMRI.estimate_coil_maps).
  * **acs** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – optional number of low frequency lines for autocalibration. If `None`, look for acs lines in `mask_generator` attributes (if exists)
    or in metadata (only available for FastMRI test/challenge data). If unavailable, and ACS required, then raises error.
  * **espirit_crop** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – crop parameter used in ESPIRiT coil map sensitivity estimation algorithm, default to 0.95. Lower crop = estimated maps may extend outside anatomy of interest, high crop = maps may be smaller than anatomy.
  * **prewhiten** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*slice*](https://docs.python.org/3.9/library/functions.html#slice) *,* [*slice*](https://docs.python.org/3.9/library/functions.html#slice) *]* *,* [*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, prewhiten kspace noise across coils,
    defaults to using a 30x30 slice in the top left corner. Optionally set tuple of slices for custom location. Defaults to False.
  * **normalize** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, normalize kspace by 99th percentile of RSS reconstruction of kspace ACS block.
    if `int` or `float`, normalize kspace by `normalize / kspace.max()`.

#### generate_maps(kspace, metadata=None)

Estimate coil maps using [`deepinv.physics.MultiCoilMRI.estimate_coil_maps()`](https://deepinv.org/api/stubs/deepinv.physics.MultiCoilMRI.html.md#deepinv.physics.MultiCoilMRI.estimate_coil_maps).

* **Parameters:**
  * **kspace** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input kspace of shape (2, N, H, W)
  * **metadata** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – optional metadata.
* **Returns:**
  estimated coil maps of shape (N, H, W) and complex dtype
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### generate_mask(kspace, seed)

Simulate mask from mask generator.

* **Parameters:**
  * **kspace** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input fully-sampled kspace of shape (2, (N,) H, W) where (N,) is optional multicoil
  * **seed** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int)) – mask generator seed. Useful for specifying same mask per data sample.
* **Returns:**
  mask of shape (C, H, W)
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### get_acs(metadata=None)

Get number of low frequency lines for autocalibration.

First checks `acs` attribute.
Then checks `mask_generator.n_center`.
Then looks in `metadata["acs"]` if it the `acs` key is present in the data.
Finally, raises error if ACS not set anywhere.

* **Parameters:**
  **metadata** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – metadata dictionary.
* **Return int:**
  acs size

#### normalize_kspace(kspace, metadata=None)

Normalize kspace by percentile of RSS of ACS.

* **Parameters:**
  * **kspace** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input kspace of shape (2, (N,) H, W)
  * **metadata** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – optional metadata.
* **Returns:**
  whitened kspace.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### prewhiten_kspace(kspace)

Prewhiten kspace using Cholesky decomposition.

* **Parameters:**
  **kspace** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input multicoil kspace of shape (2, N, H, W)
* **Returns:**
  whitened kspace.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-datasets-mrislicetransform"></a>

## Examples using `MRISliceTransform`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate scan-specific self-supervised learning, that is, learning to reconstruct MRI scans from a single accelerated sample without ground truth.">  <div class="sphx-glr-thumbnail-title">Scan-specific zero-shot SSDU for MRI</div>
</div>
<!-- thumbnail-parent-div-close --></div>
