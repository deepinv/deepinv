# L1Distance

### *class* deepinv.optim.L1Distance

Bases: [`Distance`](https://deepinv.org/api/stubs/deepinv.optim.Distance.html.md#deepinv.optim.Distance)

$\ell_1$ distance

$$
f(x) = \|x-y\|_1.
$$

#### grad(x, y, \*args, \*\*kwargs)

Gradient of the gradient of the $\ell_1$ norm, i.e.

$$
\partial \datafid(x) = \operatorname{sign}(x-y)
$$

#### NOTE
The gradient is not defined at $x=y$.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the gradient is computed.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Data $y$ of the same dimension as $x$.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) gradient of the $\ell_1$ norm at `x`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### prox(u, y, \*args, gamma=1.0, \*\*kwargs)

Proximal operator of the $\ell_1$ norm, i.e.

$$
\operatorname{prox}_{\gamma \ell_1}(x) = \underset{z}{\text{argmin}} \,\, \gamma \|z-y\|_1+\frac{1}{2}\|z-x\|_2^2
$$

also known as the soft-thresholding operator.

* **Parameters:**
  * **u** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $u$ at which the proximity operator is computed.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Data $y$ of the same dimension as $x$.
  * **gamma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – stepsize (or soft-thresholding parameter).
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) soft-thresholding of `u` with parameter `gamma`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
