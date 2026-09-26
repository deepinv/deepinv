# Shift

### *class* deepinv.transform.Shift(\*args, shift_max=1.0, \*\*kwargs)

Bases: [`Transform`](https://deepinv.org/api/stubs/deepinv.transform.Transform.md#deepinv.transform.Transform)

Fast integer 2D translations.

Generates `n_trans` randomly shifted versions of 2D images with circular padding.

See [`deepinv.transform.Transform`](https://deepinv.org/api/stubs/deepinv.transform.Transform.md#deepinv.transform.Transform) for further details and examples.

* **Parameters:**
  * **shift_max** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – maximum shift as fraction of total height/width.
  * **n_trans** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of transformed versions generated per input image.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – random number generator, if None, use torch.Generator(), defaults to None

<a id="sphx-glr-backref-deepinv-transform-shift"></a>

## Examples using `Shift`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model terris2025reconstruct to solve inverse problems.">![](auto_examples/models/images/thumb/sphx_glr_demo_foundation_model_thumb.png)

[Inference and fine-tune a foundation model](https://deepinv.org/auto_examples/models/demo_foundation_model.md)

  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates various geometric image transformations implemented in deepinv that can be used in Equivariant Imaging (EI) for self-supervised learning:">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_ei_transforms_thumb.png)

[Image transformations for Equivariant Imaging](https://deepinv.org/auto_examples/self-supervised-learning/demo_ei_transforms.md)

  <div class="sphx-glr-thumbnail-title">Image transformations for Equivariant Imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the use of our deepinv.transform module for use in solving imaging problems. These can be used for:">![](auto_examples/transforms-equivariance/images/thumb/sphx_glr_demo_transforms_thumb.png)

[Image transforms for equivariance & augmentations](https://deepinv.org/auto_examples/transforms-equivariance/demo_transforms.md)

  <div class="sphx-glr-thumbnail-title">Image transforms for equivariance & augmentations</div>
</div>
<!-- thumbnail-parent-div-close --></div>
