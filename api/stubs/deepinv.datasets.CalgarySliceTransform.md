# CalgarySliceTransform

### *class* deepinv.datasets.CalgarySliceTransform(mask_generator=None, seed_mask_generator=True, estimate_coil_maps=False, acs=None, espirit_crop=0.95, prewhiten=False, normalize=False)

Bases: [`MRISliceTransform`](https://deepinv.org/api/stubs/deepinv.datasets.MRISliceTransform.md#deepinv.datasets.MRISliceTransform)

Extract params and estimate coil maps for Calgary raw data.

To be used with [`deepinv.datasets.CalgarySliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.CalgarySliceDataset.md#deepinv.datasets.CalgarySliceDataset).

#### NOTE
The test set comes already masked, so this transform estimates the mask from the zeros of `y`. For the validation set, the data
is fully-sampled, so the estimated mask will be all-ones.

<a id="sphx-glr-backref-deepinv-datasets-calgaryslicetransform"></a>

## Examples using `CalgarySliceTransform`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo reconstructs undersampled k-space data for 2D cardiac and brain MRI on:">![](auto_examples/models/images/thumb/sphx_glr_demo_mri_pretrained_thumb.png)

[Reconstruct undersampled k-space for cardiac and brain MRI](https://deepinv.org/auto_examples/models/demo_mri_pretrained.md)

  <div class="sphx-glr-thumbnail-title">Reconstruct undersampled k-space for cardiac and brain MRI</div>
</div>
<!-- thumbnail-parent-div-close --></div>
