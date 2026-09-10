# bicgstab

### deepinv.optim.linear.bicgstab(A, b, init=None, max_iter=1e2, tol=1e-5, parallel_dim=0, verbose=False, left_precon=lambda x: ..., right_precon=lambda x: ...)

Biconjugate gradient stabilized algorithm.

Solves $Ax=b$ with $A$ squared using the BiCGSTAB algorithm in Van der Vorst [[149](https://deepinv.org/user_guide/other/biblio.html.md#id21)].

For more details see: [http://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method](http://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method)

* **Parameters:**
  * **A** (*Callable*) – Linear operator as a callable function.
  * **b** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor of shape (B, …)
  * **init** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Optional initial guess.
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of BiCGSTAB iterations.
  * **tol** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – absolute tolerance for stopping the BiCGSTAB algorithm.
  * **parallel_dim** (*None* *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – dimensions to be considered as batch dimensions. If None, all dimensions are considered as batch dimensions.
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Output progress information in the console.
  * **left_precon** (*Callable*) – left preconditioner as a callable function.
  * **right_precon** (*Callable*) – right preconditioner as a callable function.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) $x$ of shape (B, …)
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
