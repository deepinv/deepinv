# least_squares_implicit_backward

### deepinv.optim.linear.least_squares_implicit_backward(physics, y, z=None, init=None, gamma=None, \*\*kwargs)

Least squares solver with O(1) memory backward propagation using implicit differentiation.
The function is similar to [`deepinv.optim.linear.least_squares()`](https://deepinv.org/api/stubs/deepinv.optim.linear.least_squares.html.md#deepinv.optim.linear.least_squares) for the forward pass, but uses implicit differentiation for the backward pass, which reduces memory consumption to O(1) in the number of iterations.

This function supports backpropagation with respect to the inputs $y$, $z$ and $\gamma$ and also with respect to the parameters of the physics operator $A_\theta$ if they require gradients. See [Reducing the memory and computational complexity of unfolded network training](https://deepinv.org/auto_examples/unfolded/demo_unfolded_constant_memory.html.md#sphx-glr-auto-examples-unfolded-demo-unfolded-constant-memory-py) and the notes below for more details.

Let $h(z, y, \theta, \gamma)$ denote the output of the least squares solver, i.e. the solution of the following problem:

$$
h(z, y, \theta, \gamma) = \underset{x}{\arg\min} \; \frac{\gamma}{2}\|A_\theta x-y\|^2 + \frac{1}{2}\|x-z\|^2
$$

When the forward least-squares solver converges to the exact minimizer, we have the following closed-form expressions for $h(z, y, \theta, \gamma)$:

$$
h(z, y, \theta, \gamma) = \left( A_\theta^{\top} A_\theta + \frac{1}{\gamma} I \right)^{-1} \left( A_\theta^{\top} y + \frac{1}{\gamma} z \right)
$$

Let $M$ denote the inverse $\left( A_\theta^T A_\theta + \frac{1}{\gamma} I \right)^{-1}$. In the forward, we need to compute the vector-Jacobian products (VJPs), which can be computed as follows:

$$
\left( \frac{\partial h}{\partial z} \right)^{\top} v               &= \frac{1}{\gamma} M v \\
\left( \frac{\partial h}{\partial y} \right)^{\top} v               &= A_\theta M v \\
\left( \frac{\partial h}{\partial \gamma} \right)^{\top} v          &=   (h - z)^\top M  v / \gamma^2 \\
\left( \frac{\partial h}{\partial \theta} \right)^{\top} v          &= \frac{\partial p}{\partial \theta}
$$

where $p =  (y - A_\theta h)^{\top} A_\theta M v$ and $\frac{\partial p}{\partial \theta}$ can be computed using the standard backpropagation mechanism (autograd).

#### NOTE
This function only supports first-order gradients. Higher-order gradients are not supported. If you need higher-order gradients, please use [`deepinv.optim.linear.least_squares()`](https://deepinv.org/api/stubs/deepinv.optim.linear.least_squares.html.md#deepinv.optim.linear.least_squares) instead but be aware that it requires storing all intermediate iterates, which can be memory-intensive.

#### NOTE
This function also supports implicit gradients with respect to the parameters of the physics operator $A_\theta$ if they require gradients. This is useful for learning the physics parameters in an end-to-end fashion. The gradients are accumulated in-place in the `.grad` attribute of the parameters of the physics operator. To make this work, the function takes as input the physics operator itself (not just its matmul functions) and checks if any of its parameters require gradients. If so, it triggers the backward pass accordingly.

#### WARNING
Implicit gradients can be incorrect if the least squares solver does not converge sufficiently. Make sure to set the `max_iter` and `tol` parameters of the least squares solver appropriately to ensure convergence. You can monitor the convergence by setting `verbose=True` in the least squares solver via `kwargs`. If the solver does not converge, the implicit gradients can be very inaccurate and lead to divergence of the training.

#### WARNING
This function does not support [`deepinv.utils.TensorList`](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) inputs yet. If you use [`deepinv.utils.TensorList`](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) as inputs, the function will fall back to standard least squares with full backpropagation.

#### TIP
If you do not need gradients with respect to the physics parameters, you can set `requires_grad=False` for all parameters of the physics operator to avoid the additional backward pass. This can save some computation time.

#### TIP
Training unfolded network with implicit differentiation can reduce memory consumption significantly, especially when using many iterations. On GPU, we can expect a memory reduction factor of about 2x-3x compared to standard backpropagation and a speed-up of about 1.2x-1.5x. The exact numbers depend on the problem and the number of iterations.

* **Parameters:**
  * **physics** ([*deepinv.physics.LinearPhysics*](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)) – physics operator [`deepinv.physics.LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics).
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor of shape (B, …)
  * **z** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *,* *None*) – input tensor of shape (B, …). Default is `None`, which corresponds to a zero tensor.
  * **init** (*None* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Optional initial guess, only used for the forward pass. Default is `None`, which corresponds to a zero initialization.
  * **gamma** (*None* *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – regularization parameter $\gamma > 0$. Default is `None`. Can be batched (shape (B, …)) or a scalar.
  * **kwargs** – additional arguments to be passed to the least squares solver.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) $x$ of shape (B, …), the solution of the least squares problem.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
