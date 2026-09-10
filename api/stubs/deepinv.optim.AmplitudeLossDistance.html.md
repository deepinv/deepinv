# AmplitudeLossDistance

### *class* deepinv.optim.AmplitudeLossDistance

Bases: [`Distance`](https://deepinv.org/api/stubs/deepinv.optim.Distance.html.md#deepinv.optim.Distance)

Amplitude loss for [`deepinv.physics.PhaseRetrieval`](https://deepinv.org/api/stubs/deepinv.physics.PhaseRetrieval.html.md#deepinv.physics.PhaseRetrieval) reconstruction, defined as

$$
f(x) = \sum_{i=1}^{m}{(\sqrt{|y_i - x|^2}-\sqrt{y_i})^2},
$$

where $y_i$ is the i-th entry of the measurements, and $m$ is the number of measurements.

#### fn(u, y, \*args, \*\*kwargs)

Computes the amplitude loss.

* **Parameters:**
  * **u** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – estimated measurements.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – true measurements.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) the amplitude loss of shape B where B is the batch size.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### grad(u, y, \*args, epsilon=1e-12, \*\*kwargs)

Computes the gradient of the amplitude loss $\distance{u}{y}$, i.e.,

$$
\nabla_{u}\distance{u}{y} = \frac{\sqrt{u}-\sqrt{y}}{\sqrt{u}}
$$

* **Parameters:**
  * **u** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $u$ at which the gradient is computed.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Data $y$.
  * **epsilon** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – small value to avoid division by zero.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) gradient of the amplitude loss function.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
