# TVL1Prior

### *class* deepinv.optim.TVL1Prior(def_crit=1e-8, n_it_max=1000, \*args, \*\*kwargs)

Bases: [`TVPrior`](https://deepinv.org/api/stubs/deepinv.optim.TVPrior.html.md#deepinv.optim.TVPrior)

Total Variation (TV) prior with an L1 norm.

This prior computes the isotropic total variation regularization term.

The prior is defined as:

$$
\mathrm{TV}(x) = \sum_i \|\nabla x_i\|_1
$$

where $\nabla x$ denotes the discrete gradient of x.

* **Parameters:**
  * **def_crit** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – default convergence criterion for the inner solver of the TV denoiser; default value: 1e-8.
  * **n_it_max** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximal number of iterations for the inner solver of the TV denoiser; default value: 1000.

#### fn(x, \*args, \*\*kwargs)

Computes the regularizer

$$
\reg{x} = \|Dx\|_{1}

$$

where D is the finite differences linear operator.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the prior is computed.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) prior $g(x)$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
