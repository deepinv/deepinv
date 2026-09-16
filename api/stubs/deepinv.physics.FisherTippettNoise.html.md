# FisherTippettNoise

### *class* deepinv.physics.FisherTippettNoise(l=1.0)

Bases: [`NoiseModel`](https://deepinv.org/api/stubs/deepinv.physics.NoiseModel.html.md#deepinv.physics.NoiseModel)

Fisher-Tippett noise $p(y\vert x) = \frac{\ell^{\ell}}{\Gamma(\ell)}\mathrm{e}^{\ell(y-x)}\mathrm{e}^{-\ell\mathrm{e}^{(y-x)}}$

Distribution for modelling the noise of log-intensities images in SAR imaging.

#### WARNING
This noise model does not support the random number generator.

* **Parameters:**
  **l** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – noise level.

#### forward(x, l=None, \*\*kwargs)

Adds the noise to measurements x

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements (log-intensities)
  * **l** (*None* *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – noise level. If not None, it will overwrite the current noise level.
* **Returns:**
  noisy measurements (log-intensities)

<a id="sphx-glr-backref-deepinv-physics-fishertippettnoise"></a>

## Examples using `FisherTippettNoise`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example despeckles real clinical ultrasound B-mode images with Speckle2Self :footciteli2025speckle2self, a model pretrained without clean reference images.">  <div class="sphx-glr-thumbnail-title">Ultrasound despeckling from B-mode images</div>
</div>
<!-- thumbnail-parent-div-close --></div>
