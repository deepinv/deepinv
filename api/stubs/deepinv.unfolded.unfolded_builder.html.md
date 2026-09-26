# unfolded_builder

### deepinv.unfolded.unfolded_builder(iteration, params_algo=MappingProxyType({'lambda': 1.0, 'stepsize': 1.0}), data_fidelity=None, prior=None, max_iter=5, trainable_params=('lambda', 'stepsize'), device=torch.device('cpu'), cost_fn=None, g_first=False, bregman_potential=None, \*\*kwargs)

Helper function for building an unfolded architecture.

#### Deprecated
Deprecated since version 0.3.6: The `unfolded_builder` function is deprecated and will be removed in future versions.
Instead of using this function, define an unfolded algorithm using the [`deepinv.optim.BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim) class with argument `unfold=True`,
e.g. `model = PGD(data_fidelity, prior, ..., unfold = True, ...)`.

* **Parameters:**
  * **iteration** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*deepinv.optim.OptimIterator*](https://deepinv.org/api/stubs/deepinv.optim.OptimIterator.html.md#deepinv.optim.OptimIterator)) – either the name of the algorithm to be used,
    or directly an optim iterator.
    If an algorithm name (string), should be either `"GD"` (gradient descent), `"PGD"` (proximal gradient descent),
    `"ADMM"` (ADMM),
    `"HQS"` (half-quadratic splitting), `"CP"` (Chambolle-Pock) or `"DRS"` (Douglas Rachford). See
    [Optimization](https://deepinv.org/user_guide/reconstruction/optimization.html.md#optim) for more details.
  * **params_algo** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – dictionary containing all the relevant parameters for running the algorithm,
    e.g. the stepsize, regularisation parameter, denoising standard deviation.
    Each value of the dictionary can be either Iterable (distinct value for each iteration) or
    a single float (same value for each iteration).
    Default: `{"stepsize": 1.0, "lambda": 1.0}`. See [Optimization Parameters](https://deepinv.org/user_guide/reconstruction/optimization.html.md#optim-params) for more details.
  * **deepinv.optim.DataFidelity** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *,*) – data-fidelity term.
    Either a single instance (same data-fidelity for each iteration) or a list of instances of
    [`deepinv.optim.DataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity) (distinct data-fidelity for each iteration). Default: `None`.
  * **prior** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *,* [*deepinv.optim.Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)) – regularization prior.
    Either a single instance (same prior for each iteration - weight tied) or a list of instances of
    deepinv.optim.Prior (distinct prior for each iteration - weight untied). Default: `None`.
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of iterations of the unfolded algorithm. Default: 5.
  * **trainable_params** (*Sequence*) – List of parameters to be trained. Each parameter should be a key of the `params_algo`
    dictionary for the [`deepinv.optim.OptimIterator`](https://deepinv.org/api/stubs/deepinv.optim.OptimIterator.html.md#deepinv.optim.OptimIterator) class.
    This does not encompass the trainable weights of the prior module.
  * **cost_fn** (*Callable*) – Custom user input cost function. default: None.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – Device on which to perform the computations. Default: `torch.device("cpu")`.
  * **g_first** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to perform the step on $g$ before that on $f$ before or not. default: False
  * **bregman_potential** ([*deepinv.optim.Bregman*](https://deepinv.org/api/stubs/deepinv.optim.Bregman.html.md#deepinv.optim.Bregman)) – Bregman potential used for Bregman optimization algorithms such as Mirror Descent. Default: `None`, comes back to standard Euclidean optimization.
  * **kwargs** – additional arguments to be passed to the [`deepinv.optim.BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim) class.
* **Returns:**
  an unfolded architecture (instance of [`deepinv.unfolded.BaseUnfold`](https://deepinv.org/api/stubs/deepinv.unfolded.BaseUnfold.html.md#deepinv.unfolded.BaseUnfold)).

<hr />

* **Example:**

```pycon
>>> import torch
>>> import deepinv as dinv
>>>
>>> # Create a trainable unfolded architecture
>>> model = dinv.unfolded.unfolded_builder(
...     iteration="PGD",
...     data_fidelity=dinv.optim.data_fidelity.L2(),
...     prior=dinv.optim.PnP(dinv.models.DnCNN(in_channels=1, out_channels=1)),
...     params_algo={"stepsize": 1.0, "g_param": 1.0},
...     trainable_params=["stepsize", "g_param"]
... )
>>> # Forward pass
>>> x = torch.randn(1, 1, 16, 16)
>>> physics = dinv.physics.Denoising()
>>> y = physics(x)
>>> x_hat = model(y, physics)
```
