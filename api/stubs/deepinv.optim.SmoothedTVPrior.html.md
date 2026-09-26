# SmoothedTVPrior

### *class* deepinv.optim.SmoothedTVPrior(eps=2e-1, \*args, \*\*kwargs)

Bases: [`TVPrior`](https://deepinv.org/api/stubs/deepinv.optim.TVPrior.html.md#deepinv.optim.TVPrior)

Smoothed total variation prior.

$$
g(x) = \sum_i \sqrt{\|(Dx)_i\|_2^2 + \varepsilon^2}

$$

A differentiable approximation of [`TVPrior`](https://deepinv.org/api/stubs/deepinv.optim.TVPrior.html.md#deepinv.optim.TVPrior), where the non-smooth
$\ell_2$ norm is replaced by a smoothed version parameterized by
$\varepsilon$. Since $g$ is differentiable everywhere, its
proximal operator has no closed form and is approximated with the inner
gradient-descent solver inherited from [`Potential`](https://deepinv.org/api/stubs/deepinv.optim.Potential.html.md#deepinv.optim.Potential).

* **Parameters:**
  **eps** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – smoothing parameter $\varepsilon > 0$. Default: `2e-1`.

#### fn(x, \*args, \*\*kwargs)

Computes the regularizer

$$
\reg{x} = \sum_i \sqrt{\|(Dx)_i\|_2^2 + \varepsilon^2}

$$

where D is the finite differences linear operator, and the 2-norm is taken
on the dimension of the differences.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the prior is computed.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) prior $g(x)$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### grad(x, \*args, \*\*kwargs)

Computes the closed-form gradient of the smoothed TV prior at $x$

$$
\nabla \reg{x} = D^\top \left( \frac{Dx}{\sqrt{\|Dx\|_2^2 + \varepsilon^2}} \right)

$$

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the gradient is computed.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) gradient $\nabla_x g$, computed in $x$.

#### prox(x, \*args, gamma=0.1, stepsize_inter=None, max_iter_inter=200, tol_inter=1e-3, \*\*kwargs)

Approximates the proximal operator using gradient descent.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the proximity operator is computed.
  * **gamma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – stepsize of the proximity operator.
  * **stepsize_inter** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – stepsize used for the internal gradient descent. By default, uses the one from the Liscphitz bound.
  * **max_iter_inter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximal number of iterations for the internal gradient descent.
  * **tol_inter** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – internal gradient descent has converged when the L2 distance between two consecutive iterates is smaller than `tol_inter`.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) proximity operator at $x$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
