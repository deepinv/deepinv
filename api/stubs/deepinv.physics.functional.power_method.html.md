# power_method

### deepinv.physics.functional.power_method(operator, x0, max_iter=100, tol=1e-6, verbose=False, \*\*kwargs)

Runs the power iteration method to estimate the largest singular value of a linear operator.

* **Parameters:**
  * **operator** (*Callable*) – function that applies the linear operator.
  * **x0** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – initial vector for the power iteration.
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of iterations. Default: `100`.
  * **tol** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – tolerance for convergence. Default: `1e-6`.
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, prints convergence information. Default: `False`.
  * **kwargs** – additional arguments to be passed to the operator.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) Estimated largest singular value of the operator.
