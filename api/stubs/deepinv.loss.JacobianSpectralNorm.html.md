# JacobianSpectralNorm

### *class* deepinv.loss.JacobianSpectralNorm(max_iter=10, tol=1e-3, eval_mode=False, verbose=False, reduction='max', reduced_batchsize=None)

Bases: [`Loss`](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss)

Computes the spectral norm of the Jacobian.

Given a function $f:\mathbb{R}^n\to\mathbb{R}^n$, this module computes the spectral
norm of the Jacobian of $f$ in $x$, i.e.

$$
\|\frac{df}{du}(x)\|_2.
$$

This spectral norm is computed with a power method leveraging jacobian vector products, as proposed by Pesquet *et al.*<sup>[1](#footcite-pesquet2021learning)</sup>.

#### NOTE
This implementation assumes that the input $x$ is batched with shape `(B, ...)`, where B is the batch size.

* **Parameters:**
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of iteration of the power method.
  * **tol** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – tolerance for the convergence of the power method.
  * **eval_mode** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – set to `False` if one does not want to backpropagate through the spectral norm (default), set to `True` otherwise.
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to print computation details or not.
  * **reduction** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – reduction in batch dimension. One of [“mean”, “sum”, “max”], operation to be performed after all spectral norms have been computed. If `None`, a vector of length `batch_size` will be returned. Defaults to “max”.
  * **reduced_batchsize** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – if not `None`, the batch size will be reduced to this value for the computation of the spectral norm. Can be useful to reduce memory usage and computation time when the batch size is large.

<hr />

* **Examples:**

```pycon
>>> import torch
>>> from deepinv.loss.regularisers import JacobianSpectralNorm
>>> _ = torch.manual_seed(0)
>>>
>>> reg_l2 = JacobianSpectralNorm(max_iter=100, tol=1e-5, eval_mode=False, verbose=True)
>>> A = torch.diag(torch.Tensor(range(1, 51))).unsqueeze(0)  # creates a diagonal matrix with largest eigenvalue = 50
>>> x = torch.randn((1, A.shape[1])).unsqueeze(0).requires_grad_()
>>> out = x @ A
>>> regval = reg_l2(out, x)
>>> print(regval) # returns approx 50
tensor(49.9999)
```

<hr />

* **References:**

* <a id='footcite-pesquet2021learning'>**[1]**</a> Jean-Christophe Pesquet, Audrey Repetti, Matthieu Terris, and Yves Wiaux. Learning maximally monotone operators for image recovery. *SIAM Journal on Imaging Sciences*, 14(3):1206–1237, 2021.

#### forward(y, x, \*\*kwargs)

Computes the spectral norm of the Jacobian of $f$ in $x$.

#### WARNING
The input $x$ must have requires_grad=True before evaluating $f$.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – output of the function $f$ at $x$, of dimension `(B, ...)`
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input of the function $f$, of dimension `(B, ...)`

If x has multiple dimensions, it’s assumed the first one corresponds to the batch dimension.
