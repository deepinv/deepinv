# FNEJacobianSpectralNorm

### *class* deepinv.loss.FNEJacobianSpectralNorm(max_iter=10, tol=1e-3, eval_mode=False, verbose=False, reduction='max', reduced_batchsize=None)

Bases: [`Loss`](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss)

Computes the Firm-Nonexpansiveness Jacobian spectral norm.

Given a function $f:\mathbb{R}^n\to\mathbb{R}^n$, this module computes the spectral
norm of the Jacobian of $2f-\operatorname{Id}$ (where $\operatorname{Id}$ denotes the
identity) in $x$, i.e.

$$
\|\frac{d(2f-\operatorname{Id})}{du}(x)\|_2,
$$

as proposed by Pesquet *et al.*<sup>[1](#footcite-pesquet2021learning)</sup>.
This spectral norm is computed with the [`deepinv.loss.JacobianSpectralNorm`](https://deepinv.org/api/stubs/deepinv.loss.JacobianSpectralNorm.html.md#deepinv.loss.JacobianSpectralNorm) class.

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
>>> from deepinv.loss.regularisers import FNEJacobianSpectralNorm
>>> _ = torch.manual_seed(0)
>>>
>>> reg_fne = FNEJacobianSpectralNorm(max_iter=100, tol=1e-5, eval_mode=False, verbose=True)
>>> A = torch.diag(torch.Tensor(range(1, 51))).unsqueeze(0)  # creates a diagonal matrix with largest eigenvalue = 50
>>>
>>> def model_base(x):
...     return x @ A
>>>
>>> def FNE_model(x):
...     A_bis = torch.linalg.inv((A + torch.eye(A.shape[1])))  # Creates the resolvent of A, which is firmly nonexpansive
...     return x @ A_bis
>>>
>>> x = torch.randn((1, A.shape[1])).unsqueeze(0)
>>>
>>> out = model_base(x)
>>> regval = reg_fne(out, x, model_base)
>>> print(regval) # returns approx 99 (model is expansive, with Lipschitz constant 50)
tensor(98.9999)
>>> out = FNE_model(x)
>>> regval = reg_fne(out, x, FNE_model)
>>> print(regval) # returns a value smaller than 1 (model is firmly nonexpansive)
tensor(0.9595)
```

<hr />

* **References:**

* <a id='footcite-pesquet2021learning'>**[1]**</a> Jean-Christophe Pesquet, Audrey Repetti, Matthieu Terris, and Yves Wiaux. Learning maximally monotone operators for image recovery. *SIAM Journal on Imaging Sciences*, 14(3):1206–1237, 2021.

#### forward(y_in, x_in, model, \*args_model, interpolation=False, \*\*kwargs_model)

Computes the Firm-Nonexpansiveness (FNE) Jacobian spectral norm of a model.

* **Parameters:**
  * **y_in** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input of the model (by default), of dimension `(B, ...)`.
  * **x_in** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – an additional point of the model (by default), of dimension `(B, ...)`.
  * **model** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – neural network, or function, of which we want to compute the FNE Jacobian spectral norm.
  * **\*args_model** – additional arguments of the model.
  * **interpolation** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to input to model an interpolation between y_in and x_in instead of y_in (default is `False`).
  * **\*\*kwargs_model** – additional keyword arguments of the model.

<a id="sphx-glr-backref-deepinv-loss-fnejacobianspectralnorm"></a>

## Examples using `FNEJacobianSpectralNorm`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This code shows you how to use sampling algorithms to quantify uncertainty of a reconstruction from incomplete and noisy measurements.">  <div class="sphx-glr-thumbnail-title">Uncertainty quantification with PnP-ULA.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
