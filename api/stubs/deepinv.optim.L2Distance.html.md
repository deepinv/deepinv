# L2Distance

### *class* deepinv.optim.L2Distance(sigma=1.0)

Bases: [`Distance`](https://deepinv.org/api/stubs/deepinv.optim.Distance.html.md#deepinv.optim.Distance)

Implementation of $\distancename$ as the normalized $\ell_2$ norm

$$
f(x) = \frac{1}{2\sigma^2}\|x-y\|^2

$$

* **Parameters:**
  **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – normalization parameter. Default: 1.

#### fn(x, y, \*args, \*\*kwargs)

Computes the distance $\distance{x}{y}$ i.e.

$$
\distance{x}{y} = \frac{1}{2}\|x-y\|^2
$$

* **Parameters:**
  * **u** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the data fidelity is computed.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Data $y$.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) data fidelity $\datafid{u}{y}$ of size `B` with `B` the size of the batch.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### grad(x, y, \*args, \*\*kwargs)

Computes the gradient of $\distancename$, that is  $\nabla_{x}\distance{x}{y}$, i.e.

$$
\nabla_{x}\distance{x}{y} = \frac{1}{\sigma^2} x-y
$$

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the gradient is computed.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Observation $y$.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) gradient of the distance function $\nabla_{x}\distance{x}{y}$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### prox(x, y, \*args, gamma=1.0, \*\*kwargs)

Proximal operator of $\gamma \distance{x}{y} = \frac{\gamma}{2 \sigma^2} \|x-y\|^2$.

Computes $\operatorname{prox}_{\gamma \distancename}$, i.e.

$$
\operatorname{prox}_{\gamma \distancename} = \underset{u}{\text{argmin}} \frac{\gamma}{2\sigma^2}\|u-y\|_2^2+\frac{1}{2}\|u-x\|_2^2
$$

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the proximity operator is computed.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Data $y$.
  * **gamma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – thresholding parameter.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) proximity operator $\operatorname{prox}_{\gamma \distancename}(x)$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
