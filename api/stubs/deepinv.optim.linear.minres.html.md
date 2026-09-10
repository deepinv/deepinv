# minres

### deepinv.optim.linear.minres(A, b, init=None, max_iter=1e2, tol=1e-5, eps=1e-6, parallel_dim=0, verbose=False, precon=lambda x: ...)

Minimal Residual Method for solving symmetric equations.

Solves $Ax=b$ with $A$ symmetric using the MINRES algorithm in Paige and Saunders [[106](https://deepinv.org/user_guide/other/biblio.html.md#id22)]

The method assumes that $A$ is hermite.
For more details see: [https://en.wikipedia.org/wiki/Minimal_residual_method](https://en.wikipedia.org/wiki/Minimal_residual_method)

Based on [https://github.com/cornellius-gp/linear_operator](https://github.com/cornellius-gp/linear_operator)
Modifications and simplifications for compatibility with deepinverse

* **Parameters:**
  * **A** (*Callable*) – Linear operator as a callable function.
  * **b** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor of shape (B, …)
  * **init** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Optional initial guess.
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of MINRES iterations.
  * **tol** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – absolute tolerance for stopping the MINRES algorithm.
  * **parallel_dim** (*None* *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – dimensions to be considered as batch dimensions. If None, all dimensions are considered as batch dimensions.
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Output progress information in the console.
  * **precon** (*Callable*) – preconditioner is a callable function (not tested). Must be positive definite
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) $x$ of shape (B, …)
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
