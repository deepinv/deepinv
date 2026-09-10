# conjugate_gradient

### deepinv.optim.linear.conjugate_gradient(A, b, max_iter=1e2, tol=1e-5, eps=1e-8, parallel_dim=0, init=None, verbose=False)

Standard conjugate gradient algorithm.

It solves the linear system $Ax=b$, where $A$ is a (square) linear operator and $b$ is a tensor.

For more details see: [http://en.wikipedia.org/wiki/Conjugate_gradient_method](http://en.wikipedia.org/wiki/Conjugate_gradient_method)

* **Parameters:**
  * **A** (*Callable*) – Linear operator as a callable function, has to be square!
  * **b** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor of shape (B, …)
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of CG iterations
  * **tol** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – absolute tolerance for stopping the CG algorithm.
  * **eps** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – a small value for numerical stability
  * **parallel_dim** (*None* *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – dimensions to be considered as batch dimensions. If None, all dimensions are considered as batch dimensions.
  * **init** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Optional initial guess.
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Output progress information in the console.
* **Returns:**
  [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) $x$ of shape (B, …) verifying $Ax=b$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
