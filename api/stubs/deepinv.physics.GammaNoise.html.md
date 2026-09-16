# GammaNoise

### *class* deepinv.physics.GammaNoise(l=1.0)

Bases: [`NoiseModel`](https://deepinv.org/api/stubs/deepinv.physics.NoiseModel.html.md#deepinv.physics.NoiseModel)

Gamma noise $y = \mathcal{G}(\ell, x/\ell)$

Follows the (shape, scale) parameterization of the Gamma distribution,
where the mean is given by $x$ and the variance is given by $x^2/\ell$,
see [https://en.wikipedia.org/wiki/Gamma_distribution](https://en.wikipedia.org/wiki/Gamma_distribution) for more details.

Distribution for modelling speckle noise (e.g. SAR images),
where $\ell>0$ controls the noise level (smaller values correspond to higher noise).

#### WARNING
This noise model does not support the random number generator.

* **Parameters:**
  **l** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – noise level.

#### forward(x, l=None, seed=None, \*\*kwargs)

Adds the noise to measurements x

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements
  * **l** (*None* *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – noise level. If not None, it will overwrite the current noise level.
* **Returns:**
  noisy measurements

<a id="sphx-glr-backref-deepinv-physics-gammanoise"></a>

## Examples using `GammaNoise`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, using noisy images only via the Generalized Recorrupted2Recorrupted (GR2R) loss :footcitemonroy2025generalized, which exploits knowledge about the noise distribution. You can change the noise distribution by selecting from predefined noise models such as Gaussian, Poisson, and Gamma noise.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the Generalized R2R loss.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
