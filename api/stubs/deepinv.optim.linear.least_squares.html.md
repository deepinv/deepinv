# least_squares

### deepinv.optim.linear.least_squares(A, AT, y, z=0.0, init=None, gamma=None, parallel_dim=0, AAT=None, ATA=None, solver='CG', max_iter=100, tol=1e-6, \*\*kwargs)

Solves $\min_x \|Ax-y\|^2 + \frac{1}{\gamma}\|x-z\|^2$ using the specified solver.

The solvers are stopped either when $\|Ax-y\| \leq \text{tol} \times \|y\|$ or
when the maximum number of iterations is reached.

The solution depends on the regularization parameter $\gamma$:

- If `gamma=None` ($\gamma = \infty$), it solves the unregularized least squares problem $\min_x \|Ax-y\|^2$.
  : - If $A$ is overcomplete (rows>=columns), it computes the minimum norm solution $x = A^{\top}(AA^{\top})^{-1}y$.
    - If $A$ is undercomplete (columns>rows), it computes the least squares solution $x = (A^{\top}A)^{-1}A^{\top}y$.
- If $0 < \gamma < \infty$, it computes the least squares solution $x = (A^{\top}A + \frac{1}{\gamma}I)^{-1}(A^{\top}y + \frac{1}{\gamma}z)$.

#### WARNING
If $\gamma \leq 0$, the problem can become non-convex and the solvers are not designed for that.
A warning is raised, but solvers continue anyway (except for LSQR, which cannot be used for negative $\gamma$).

Available solvers are:

- `'CG'`: [Conjugate Gradient](https://en.wikipedia.org/wiki/Conjugate_gradient_method).
- `'BiCGStab'`: [Biconjugate Gradient Stabilized method](https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method)
- `'lsqr'`: [Least Squares QR](https://www-leland.stanford.edu/group/SOL/software/lsqr/lsqr-toms82a.pdf)
- `'minres'`: [Minimal Residual Method](https://en.wikipedia.org/wiki/Minimal_residual_method)

#### NOTE
Both `'CG'` and `'BiCGStab'` are used for squared linear systems, while `'lsqr'` is used for rectangular systems.

If the chosen solver requires a squared system, we map to the problem to the normal equations:
If the size of $y$ is larger than $x$ (overcomplete problem), it computes $(A^{\top} A)^{-1} A^{\top} y$,
otherwise (incomplete problem) it computes $A^{\top} (A A^{\top})^{-1} y$.

* **Parameters:**
  * **A** (*Callable*) – Linear operator $A$ as a callable function.
  * **AT** (*Callable*) – Adjoint operator $A^{\top}$ as a callable function.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor of shape (B, …)
  * **z** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor of shape (B, …) or scalar.
  * **init** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – (Optional) initial guess for the solver. If None, it is set to a tensor of zeros.
  * **gamma** (*None* *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – (Optional) inverse regularization parameter. Can be batched (shape (B, …)) or a scalar.
    If multi-dimensional tensor, then its shape must match that of $A^{\top} y$.
    If None, it is set to $\infty$ (no regularization).
  * **solver** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – solver to be used, options are `'CG'`, `'BiCGStab'`, `'lsqr'` and `'minres'`.
  * **AAT** (*Callable*) – (Optional) Efficient implementation of $A(A^{\top}(x))$. If not provided, it is computed as $A(A^{\top}(x))$.
  * **ATA** (*Callable*) – (Optional) Efficient implementation of $A^{\top}(A(x))$. If not provided, it is computed as $A^{\top}(A(x))$.
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of iterations.
  * **tol** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – relative tolerance for stopping the algorithm.
  * **parallel_dim** (*None* *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – dimensions to be considered as batch dimensions. If None, all dimensions are considered as batch dimensions.
  * **kwargs** – Keyword arguments to be passed to the solver.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) $x$ of shape (B, …).
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
