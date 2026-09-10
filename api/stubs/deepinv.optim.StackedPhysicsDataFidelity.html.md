# StackedPhysicsDataFidelity

### *class* deepinv.optim.StackedPhysicsDataFidelity(data_fidelity_list)

Bases: [`DataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)

Stacked data fidelity term $\datafid{x}{y} = \sum_i d_i(A_i(x),y_i)$.

Adapted to [`deepinv.physics.StackedPhysics`](https://deepinv.org/api/stubs/deepinv.physics.StackedPhysics.html.md#deepinv.physics.StackedPhysics) physics composed of multiple physics operators.

* **Parameters:**
  **data_fidelity_list** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*deepinv.optim.DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity) *]*) – list of data fidelity terms, one per physics operator.

<hr />

* **Examples:**
  Define a stacked data fidelity term with two data fidelity terms $f_1(A_1(x),y_1) + f_2(A_2(x,y_2)$:
  ```pycon
  >>> import torch
  >>> import deepinv as dinv
  >>> # define two observations, one with Gaussian noise and one with Poisson noise
  >>> physics1 = dinv.physics.Denoising(dinv.physics.GaussianNoise(.1))
  >>> physics2 = dinv.physics.Denoising(dinv.physics.PoissonNoise(.1))
  >>> physics = dinv.physics.StackedLinearPhysics([physics1, physics2])
  >>> fid1 = dinv.optim.L2()
  >>> fid2 = dinv.optim.PoissonLikelihood()
  >>> data_fidelity = dinv.optim.StackedPhysicsDataFidelity([fid1, fid2])
  >>> x = torch.ones(1, 1, 3, 3) # image
  >>> y = physics(x) # noisy measurements
  >>> d = data_fidelity(x, y, physics)
  ```

#### fn(x, y, physics, \*args, \*\*kwargs)

Computes the data fidelity term $\datafid{x}{y} = \sum_i d_i(A_i(x),y_i)$.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the data fidelity is computed.
  * **y** ([*deepinv.utils.TensorList*](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList)) – Stacked measurements $y$.
  * **physics** ([*deepinv.physics.StackedPhysics*](https://deepinv.org/api/stubs/deepinv.physics.StackedPhysics.html.md#deepinv.physics.StackedPhysics)) – physics model.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) data fidelity $\datafid{x}{y}$.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### grad(x, y, physics, \*args, \*\*kwargs)

Calculates the gradient of the data fidelity term $\datafidname$ at $x$.

The gradient is computed using the chain rule:

$$
\nabla_x \distance{\forw{x}}{y} = \sum_i \left. \frac{\partial A_i}{\partial x} \right|_x^\top \nabla_u \distance{u}{y_i},
$$

where $\left. \frac{\partial A_i}{\partial x} \right|_x$ is the Jacobian of $A_i$ at $x$,
and $\nabla_u \distance{u}{y_i}$ is computed using `grad_d` with $u = \forw{x}$.
The multiplication is computed using the `A_vjp` method of each physics.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the gradient is computed.
  * **y** ([*deepinv.utils.TensorList*](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList)) – Stacked measurements $y$.
  * **physics** ([*deepinv.physics.StackedPhysics*](https://deepinv.org/api/stubs/deepinv.physics.StackedPhysics.html.md#deepinv.physics.StackedPhysics)) – Stacked physics model.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) gradient $\nabla_x \datafid{x}{y}$, computed in $x$.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
