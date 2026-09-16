# BaseUnfold

### *class* deepinv.unfolded.BaseUnfold(iterator, params_algo=MappingProxyType({'lambda': 1.0, 'stepsize': 1.0}), data_fidelity=None, prior=None, max_iter=5, trainable_params=('lambda', 'stepsize'), device=torch.device('cpu'), \*args, \*\*kwargs)

Bases: [`BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim)

Base class for unfolded algorithms. Child of [`deepinv.optim.BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim).

#### Deprecated
Deprecated since version 0.3.6: The `BaseUnfold` class is deprecated and will be removed in future versions.
Instead of using this function, define an unfolded algorithm using the [`deepinv.optim.BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim) class with argument `unfold=True`,
e.g. `model = PGD(data_fidelity, prior, ..., unfold = True, ...)`.

Enables to turn any iterative optimization algorithm into an unfolded algorithm, i.e. an algorithm
that can be trained end-to-end, with learnable parameters. Recall that the algorithms have the
following form (see [`deepinv.optim.OptimIterator`](https://deepinv.org/api/stubs/deepinv.optim.OptimIterator.html.md#deepinv.optim.OptimIterator)):

$$
z_{k+1} &= \operatorname{step}_f(x_k, z_k, y, A, \gamma, ...)\\
x_{k+1} &= \operatorname{step}_g(x_k, z_k, y, A, \lambda, \sigma, ...)

$$

where $\operatorname{step}_f$ and $\operatorname{step}_g$ are learnable modules.
These modules encompass trainable parameters of the algorithm (e.g. stepsize $\gamma$, regularization parameter $\lambda$, prior parameter (`g_param`) $\sigma$ …)
as well as trainable priors (e.g. a deep denoiser).

* **Parameters:**
  * **iteration** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*deepinv.optim.OptimIterator*](https://deepinv.org/api/stubs/deepinv.optim.OptimIterator.html.md#deepinv.optim.OptimIterator)) – either the name of the algorithm to be used,
    or directly an optim iterator.
    If an algorithm name (string), should be either `"GD"` (gradient descent), `"PGD"` (proximal gradient descent),
    `"ADMM"` (ADMM),
    `"HQS"` (half-quadratic splitting), `"CP"` (Chambolle-Pock) or `"DRS"` (Douglas Rachford). See
    <optim> for more details.
  * **params_algo** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – dictionary containing all the relevant parameters for running the algorithm,
    e.g. the stepsize, regularisation parameter, denoising standard deviation.
    Each value of the dictionary can be either Iterable (distinct value for each iteration) or
    a single float (same value for each iteration).
    Default: `{"stepsize": 1.0, "lambda": 1.0}`. See [Optimization Parameters](https://deepinv.org/user_guide/reconstruction/optimization.html.md#optim-params) for more details.
  * **deepinv.optim.DataFidelity** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *,*) – data-fidelity term.
    Either a single instance (same data-fidelity for each iteration) or a list of instances of
    [`deepinv.optim.DataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity) (distinct data-fidelity for each iteration). Default: `None`.
  * **prior** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *,* [*deepinv.optim.Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)) – regularization prior.
    Either a single instance (same prior for each iteration) or a list of instances of
    deepinv.optim.Prior (distinct prior for each iteration). Default: `None`.
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of iterations of the unfolded algorithm. Default: 5.
  * **trainable_params** (*Sequence*) – List of parameters to be trained. Each parameter should be a key of the `params_algo`
    dictionary for the [`deepinv.optim.OptimIterator`](https://deepinv.org/api/stubs/deepinv.optim.OptimIterator.html.md#deepinv.optim.OptimIterator) class.
    This does not encompass the trainable weights of the prior module.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – Device on which to perform the computations. Default: `torch.device("cpu")`.
  * **g_first** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to perform the step on $g$ before that on $f$ before or not. default: False
  * **kwargs** – Keyword arguments to be passed to the [`deepinv.optim.BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim) class.

#### forward(y, physics, x_gt=None, compute_metrics=False, \*\*kwargs)

Runs the fixed-point iteration algorithm. This is the same forward as in the parent BaseOptim class, but without the `torch.no_grad()` context manager.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurement vector.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – physics of the problem for the acquisition of `y`.
  * **x_gt** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – (optional) ground truth image, for plotting the PSNR across optim iterations.
  * **compute_metrics** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to compute the metrics or not. Default: `False`.
* **Returns:**
  If `compute_metrics` is `False`,  returns ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) the output of the algorithm.
  Else, returns (torch.Tensor, dict) the output of the algorithm and the metrics.
