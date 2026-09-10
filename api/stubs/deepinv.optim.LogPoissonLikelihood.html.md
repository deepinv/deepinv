# LogPoissonLikelihood

### *class* deepinv.optim.LogPoissonLikelihood(N0=1024.0, mu=1 / 50.0)

Bases: [`DataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)

Log-Poisson negative log-likelihood.

$$
\datafid{z}{y} =  N_0 (1^{\top} \exp(-\mu z)+ \mu \exp(-\mu y)^{\top}x)
$$

Corresponds to [`deepinv.physics.LogPoissonNoise`](https://deepinv.org/api/stubs/deepinv.physics.LogPoissonNoise.html.md#deepinv.physics.LogPoissonNoise) with the same arguments $N_0$ and $\mu$.
There is no closed-form of the proximal operator known.

* **Parameters:**
  * **N0** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – average number of photons
  * **mu** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – normalization constant

<a id="sphx-glr-backref-deepinv-optim-logpoissonlikelihood"></a>

## Examples using `LogPoissonLikelihood`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the Astra tomography toolbox with deepinv, a popular toolbox for tomography with GPU acceleration.">  <div class="sphx-glr-thumbnail-title">Low-dose CT with ASTRA backend and Total-Variation (TV) prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we use patch priors for limited angle computed tomography. More precisely, we consider the inverse problem y = \\mathrm{noisy}(Ax), where A is the discretized Radon transform with 100 equispace angles between 20 and 160 degrees. For the reconstruction, we minimize the variational problem">  <div class="sphx-glr-thumbnail-title">Patch priors for limited-angle computed tomography</div>
</div>
<!-- thumbnail-parent-div-close --></div>
