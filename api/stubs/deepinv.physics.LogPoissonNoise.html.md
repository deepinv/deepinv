# LogPoissonNoise

### *class* deepinv.physics.LogPoissonNoise(N0=1024.0, mu=1 / 50.0, rng=None)

Bases: [`NoiseModel`](https://deepinv.org/api/stubs/deepinv.physics.NoiseModel.html.md#deepinv.physics.NoiseModel)

Log-Poisson noise $y = -\frac{1}{\mu} \log(\frac{\mathcal{P}(\exp(-\mu x) N_0)}{N_0})$.

This noise model is mostly used for modelling the noise for (low dose) computed tomography measurements.
Here, $N_0$ describes the average number of measured photons. It acts as a noise-level parameter, where a
larger value of $N_0$ corresponds to a lower strength of the noise.
The value $\mu$ acts as a normalization constant of the forward operator. Consequently, it should be chosen antiproportionally to the image size.

For more details on the interpretation of the parameters for CT measurements, we refer to the paper Leuschner *et al.*<sup>[1](#footcite-leuschner2021lodopab)</sup>.

* **Parameters:**
  * **N0** (*Union* *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – number of photons
  * **mu** (*Union* *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – normalization constant
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator) *,* *None*) – (optional) a pseudorandom random number generator for the parameter generation.

<hr />

* **Examples:**
  Adding LogPoisson noise to a physics operator by setting the `noise_model`
  attribute of the physics operator:
  ```pycon
  >>> from deepinv.physics import Denoising, LogPoissonNoise
  >>> import torch
  >>> physics = Denoising()
  >>> physics.noise_model = LogPoissonNoise()
  >>> x = torch.rand(1, 1, 2, 2)
  >>> y = physics(x)
  ```

<hr />

* **References:**

* <a id='footcite-leuschner2021lodopab'>**[1]**</a> Johannes Leuschner, Maximilian Schmidt, Daniel Otero Baguer, and Peter Maass. Lodopab-ct, a benchmark dataset for low-dose computed tomography reconstruction. *Scientific Data*, 8(1):109, 2021.

#### forward(x, mu=None, N0=None, seed=None, \*\*kwargs)

Adds the noise to measurements x

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements
  * **mu** (*None* *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – number of photons.
    If not None, it will overwrite the current number of photons.
  * **N0** (*None* *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – normalization constant.
    If not None, it will overwrite the current normalization constant.
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the seed for the random number generator, if `rng` is provided.
* **Returns:**
  noisy measurements

<a id="sphx-glr-backref-deepinv-physics-logpoissonnoise"></a>

## Examples using `LogPoissonNoise`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the Astra tomography toolbox with deepinv, a popular toolbox for tomography with GPU acceleration.">  <div class="sphx-glr-thumbnail-title">Low-dose CT with ASTRA backend and Total-Variation (TV) prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we use patch priors for limited angle computed tomography. More precisely, we consider the inverse problem y = \\mathrm{noisy}(Ax), where A is the discretized Radon transform with 100 equispace angles between 20 and 160 degrees. For the reconstruction, we minimize the variational problem">  <div class="sphx-glr-thumbnail-title">Patch priors for limited-angle computed tomography</div>
</div>
<!-- thumbnail-parent-div-close --></div>
