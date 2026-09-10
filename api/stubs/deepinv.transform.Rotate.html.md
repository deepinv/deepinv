# Rotate

### *class* deepinv.transform.Rotate(\*args, limits=360.0, multiples=1.0, positive=False, interpolation_mode=None, \*\*kwargs)

Bases: [`Transform`](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform)

2D Rotations.

Generates `n_trans` randomly rotated versions of 2D images with zero padding (without replacement).

Picks integer angles between -limits and limits, by default -360 to 360. Set `positive=True` to clip to positive degrees.
For exact pixel rotations (0, 90, 180, 270 etc.), set `multiples=90`.

By default, output will be cropped/padded to input shape. Set `constant_shape=False` to let output shape differ from input shape.

See [`deepinv.transform.Transform`](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform) for further details and examples.

* **Parameters:**
  * **limits** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – images are rotated in the range of angles (-limits, limits).
  * **multiples** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – angles are selected uniformly from $\pm$ multiples of `multiples`. Default to 1 (i.e integers)
    When multiples is a multiple of 90, no interpolation is performed.
  * **positive** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if True, only consider positive angles.
  * **interpolation_mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *torchvision.transforms.InterpolationMode*) – interpolation mode or equivalent string used for rotation,
    defaults to `nearest`. See `torchvision.transforms.InterpolationMode` for options.
  * **n_trans** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of transformed versions generated per input image.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – random number generator, if `None`, use [`torch.Generator`](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator), defaults to `None`

<a id="sphx-glr-backref-deepinv-transform-rotate"></a>

## Examples using `Rotate`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates various geometric image transformations implemented in deepinv that can be used in Equivariant Imaging (EI) for self-supervised learning:">  <div class="sphx-glr-thumbnail-title">Image transformations for Equivariant Imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a reconstruction network for an MRI inverse problem on a fully self-supervised way, i.e., using measurement data only.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Imaging for MRI.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Equivariant splitting consists in minimizing a self-supervised loss to train a reconstruction model using measurement data only :footcitesechaud26Equivariant.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the use of our deepinv.transform module for use in solving imaging problems. These can be used for:">  <div class="sphx-glr-thumbnail-title">Image transforms for equivariance & augmentations</div>
</div>
<!-- thumbnail-parent-div-close --></div>
