# LinearPhysicsMultiScaler

### *class* deepinv.physics.LinearPhysicsMultiScaler(physics, img_size, filter='sinc', factors=(2, 4, 8), device='cpu', \*\*kwargs)

Bases: [`PhysicsMultiScaler`](https://deepinv.org/api/stubs/deepinv.physics.PhysicsMultiScaler.html.md#deepinv.physics.PhysicsMultiScaler), [`LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)

Multi-scale wrapper for linear physics operators.

See [`PhysicsMultiScaler`](https://deepinv.org/api/stubs/deepinv.physics.PhysicsMultiScaler.html.md#deepinv.physics.PhysicsMultiScaler) for details.

* **Examples:**
  A multi-scale BlurFFT operator can be created as follows:
  ```pycon
  >>> import torch
  >>> import deepinv as dinv
  >>> physics = dinv.physics.BlurFFT(img_size=(1, 32, 32), filter=dinv.physics.functional.gaussian_blur(sigma=(0.2, 0.2)))
  >>> x = torch.rand((1, 1, 8, 8))  # define an image 4 times smaller than the physics input size (scale = 2)
  >>> new_physics = dinv.physics.LinearPhysicsMultiScaler(physics, (1, 32, 32), factors=[2, 4, 8])  # define a multi-scale physics with base img size (1, 32, 32)
  >>> y = new_physics(x, scale=2)  # applying physics at scale 2
  >>> print(y.shape)
  torch.Size([1, 1, 32, 32])
  ```
* **Parameters:**
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – base physics operator.
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – shape of the input image (C, H, W).
  * **filter** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – type of filter to use for upsampling, e.g., ‘sinc’, ‘nearest’, ‘bilinear’.
  * **factors** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,*  *...* *]*) – list of factors to use for upsampling.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – device to use for the upsampling operator, e.g., ‘cpu’, ‘mps’, ‘cuda’.

#### A_dagger(y, scale=None, \*\*kwargs)

Computes the pseudo-inverse of the linear operator $A$.

If the scale is set to 0, it uses the base physics pseudo-inverse, which might have a more efficient implementation.

* **Parameters:**
  **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements tensor
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) estimated signal tensor

#### prox_l2(z, y, gamma, solver='CG', max_iter=None, tol=None, verbose=False, scale=None, \*\*kwargs)

Computes proximal operator of $f(x) = \frac{1}{2}\|Ax-y\|^2$, i.e.,

$$
\underset{x}{\arg\min} \; \frac{\gamma}{2}\|Ax-y\|^2 + \frac{1}{2}\|x-z\|^2
$$

If the scale is set to 0, it uses the base physics proximal operator, which might have a more efficient implementation.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements tensor
  * **z** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – signal tensor
  * **gamma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – hyperparameter of the proximal operator
  * **solver** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – solver to use for the proximal operator, see [`deepinv.optim.linear.least_squares()`](https://deepinv.org/api/stubs/deepinv.optim.linear.least_squares.html.md#deepinv.optim.linear.least_squares) for details
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of iterations for iterative solvers
  * **tol** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – tolerance for iterative solvers
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to print information during the solver execution
  * **scale** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – scale at which to apply the physics operator
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) estimated signal tensor
