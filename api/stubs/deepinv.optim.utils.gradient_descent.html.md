# gradient_descent

### deepinv.optim.utils.gradient_descent(grad_f, x, step_size=1.0, max_iter=100, tol=1e-5)

Standard gradient descent algorithm\`.

* **Parameters:**
  * **grad_f** (*Callable*) – gradient of function to bz minimized as a callable function.
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor.
  * **step_size** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – (constant) step size of the gradient descent algorithm.
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of iterations.
  * **tol** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – absolute tolerance for stopping the algorithm.
* **Returns:**
  torch.Tensor $x$ minimizing $f(x)$.
