# Decolorize

### *class* deepinv.physics.Decolorize(channels=3, srf='rec601', device='cpu', \*\*kwargs)

Bases: [`DecomposablePhysics`](https://deepinv.org/api/stubs/deepinv.physics.DecomposablePhysics.html.md#deepinv.physics.DecomposablePhysics)

Converts n-channel images to grayscale.

The image channels are multiplied by factors determined by the spectral response function (SRF), then summed to produce a grayscale image.

We provide various ways of defining the SRF including the [rec601](https://en.wikipedia.org/wiki/Rec._601) convention for RGB images.

In the adjoint operation, we multiply the grayscale image by the coefficients in the SRF.

Images must be tensors with C channels, i.e. `(B,C,H,W)`. The measurements are grayscale images.

* **Parameters:**
  * **channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of channels in the input image.
  * **srf** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list)) – 

    spectral response function. Either pass in user-defined SRF (must be of length channels),
    or `rec601` (default) following the [rec601](https://en.wikipedia.org/wiki/Rec._601) convention,
    or `flat` for a flat SRF (i.e. averages channels), or `random` for random SRF (e.g. to initialize joint learning).
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – device on which to perform the computations. Default: `cpu`.

<hr />

* **Examples:**
  Decolorize a 3x3 image:
  ```pycon
  >>> import torch
  >>> from deepinv.physics import Decolorize
  >>> x = torch.ones((1, 3, 3, 3), requires_grad=False) # 3x3 RGB image
  >>> physics = Decolorize()
  >>> physics(x)
  tensor([[[[1.0000, 1.0000, 1.0000],
            [1.0000, 1.0000, 1.0000],
            [1.0000, 1.0000, 1.0000]]]])
  ```

<a id="sphx-glr-backref-deepinv-physics-decolorize"></a>

## Examples using `Decolorize`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in physics.">  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we demonstrate remote sensing inverse problems for multispectral satellite imaging.">  <div class="sphx-glr-thumbnail-title">Remote sensing with satellite images</div>
</div>
<!-- thumbnail-parent-div-close --></div>
