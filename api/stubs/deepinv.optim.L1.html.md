# L1

### *class* deepinv.optim.L1

Bases: [`DataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)

$\ell_1$ data fidelity term.

In this case, the data fidelity term is defined as

$$
f(x) = \|Ax-y\|_1.
$$

#### prox(x, y, physics, \*args, gamma=1.0, stepsize=None, crit_conv=1e-5, max_iter=100, \*\*kwargs)

Proximal operator of the $\ell_1$ norm composed with A, i.e.

$$
\operatorname{prox}_{\gamma \ell_1}(x) = \underset{u}{\text{argmin}} \,\, \gamma \|Au-y\|_1+\frac{1}{2}\|u-x\|_2^2.
$$

Since no closed form is available for general measurement operators, we use a dual forward-backward algorithm.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the proximity operator is computed.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Data $y$ of the same dimension as $\forw{x}$.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – physics model.
  * **stepsize** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – step-size of the dual-forward-backward algorithm.
  * **gamma** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – stepsize of the proximity operator.
  * **crit_conv** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – convergence criterion of the dual-forward-backward algorithm.
  * **max_iter** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – maximum number of iterations of the dual-forward-backward algorithm.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) projection on the $\ell_2$ ball of radius `radius` and centered in `y`.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
