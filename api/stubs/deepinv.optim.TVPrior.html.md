# TVPrior

### *class* deepinv.optim.TVPrior(def_crit=1e-8, n_it_max=1000, \*args, \*\*kwargs)

Bases: [`Prior`](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)

Total variation (TV) prior $\reg{x} = \| D x \|_{1,2}$.

* **Parameters:**
  * **def_crit** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – default convergence criterion for the inner solver of the TV denoiser; default value: 1e-8.
  * **n_it_max** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximal number of iterations for the inner solver of the TV denoiser; default value: 1000.

#### fn(x, \*args, \*\*kwargs)

Computes the regularizer

$$
\reg{x} = \|Dx\|_{1,2}

$$

where D is the finite differences linear operator,
and the 2-norm is taken on the dimension of the differences.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the prior is computed.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) prior $g(x)$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### grad(x, \*args, \*\*kwargs)

Computes a subgradient of the total variation prior:

$$
\partial g_\sigma (x) = -\mathrm{div}\left(\frac{Dx}{|D x|}\right)

$$

where $D$ is the finite differences linear operator and $\mathrm{div}$ is its adjoint, the divergence operator.

At locations where the finite difference vanishes, the zero element
of the subdifferential is selected.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the subgradient is computed.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) subgradient at $x$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### nabla(x)

Applies the finite differences operator associated with tensors of the same shape as x.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the input tensor.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) finite differences of x.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### nabla_adjoint(x)

Applies the adjoint of the finite difference operator.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the input tensor.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) adjoint of the finite differences of x.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### prox(x, \*args, gamma=1.0, \*\*kwargs)

Compute the proximity operator of TV with the denoiser [`TVDenoiser`](https://deepinv.org/api/stubs/deepinv.models.TVDenoiser.html.md#deepinv.models.TVDenoiser).

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the proximity operator is computed.
  * **gamma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – stepsize of the proximity operator.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) proximity operator at $x$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-optim-tvprior"></a>

## Examples using `TVPrior`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using an iterative algorithm.">  <div class="sphx-glr-thumbnail-title">Use iterative reconstruction algorithms</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the Astra tomography toolbox with deepinv, a popular toolbox for tomography with GPU acceleration.">  <div class="sphx-glr-thumbnail-title">Low-dose CT with ASTRA backend and Total-Variation (TV) prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use Spyrit linear models and measurements with DeepInverse. Here we use the HadamSplit2d linear model from Spyrit.">  <div class="sphx-glr-thumbnail-title">Single-pixel imaging with Spyrit</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard TV prior for image deblurring. The problem writes as y = Ax + \\epsilon where A is a convolutional operator and \\epsilon is the realization of some Gaussian noise. The goal is to recover the original image x from the blurred and noisy image y. The TV prior is used to regularize the problem.">  <div class="sphx-glr-thumbnail-title">Image deblurring with Total-Variation (TV) prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a full-resolution multispectral image from raw snapshot mosaiced measurements using various reconstruction algorithms.">  <div class="sphx-glr-thumbnail-title">Multispectral demosaicing from raw sensor data</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to solve Poisson inverse problems using the Maximum-Likelihood Expectation-Maximization (MLEM) algorithm :footcitesheppMaximumLikelihoodReconstruction1982, also known as the Richardson-Lucy algorithm in the deconvolution setting :footciterichardsonBayesianBasedIterativeMethod1972,lucyIterativeTechniqueRectification1974.">  <div class="sphx-glr-thumbnail-title">Poisson Inverse Problems with Maximum-Likelihood Expectation-Maximization (MLEM)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows the use of the deepinv.physics.SpatialUnwrapping forward model and the deepinv.optim.ItohFidelity for unwrapping problems, which occur in modulo imaging, interferometry SAR and other imaging applications. It shows how to generate a wrapped phase image, apply blur and noise, and reconstruct the original phase using both DCT inversion and ADMM optimization.">  <div class="sphx-glr-thumbnail-title">Spatial unwrapping and modulo imaging</div>
</div>
<!-- thumbnail-parent-div-close --></div>
