# lsqr

### deepinv.optim.linear.lsqr(A, AT, b, eta=0.0, x0=None, tol=1e-6, conlim=1e8, max_iter=100, parallel_dim=0, verbose=False, \*\*kwargs)

LSQR algorithm for solving linear systems.

Code adapted from SciPy’s implementation of LSQR: [https://github.com/scipy/scipy/blob/v1.15.1/scipy/sparse/linalg/_isolve/lsqr.py](https://github.com/scipy/scipy/blob/v1.15.1/scipy/sparse/linalg/_isolve/lsqr.py)

The function solves the linear system $\min_x \|Ax-b\|^2 + \eta \|x-x_0\|^2$ in the least squares sense
using the LSQR algorithm of Paige and Saunders [[107](https://deepinv.org/user_guide/other/biblio.html.md#id23)].

* **Parameters:**
  * **A** (*Callable*) – Linear operator as a callable function.
  * **AT** (*Callable*) – Adjoint operator as a callable function.
  * **b** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor of shape (B, …)
  * **eta** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – damping parameter $eta \geq 0$. Can be batched (shape (B, …)) or a scalar.
  * **x0** (*None* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Optional $x_0$, which is also used as the initial guess.
  * **tol** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – relative tolerance for stopping the LSQR algorithm.
  * **conlim** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – maximum value of the condition number of the system.
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of LSQR iterations.
  * **parallel_dim** (*None* *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – dimensions to be considered as batch dimensions. If None, all dimensions are considered as batch dimensions.
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Output progress information in the console.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) $x$ of shape (B, …), ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) condition number of the system.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
