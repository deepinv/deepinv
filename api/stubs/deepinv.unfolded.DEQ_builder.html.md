# DEQ_builder

### deepinv.unfolded.DEQ_builder(iteration, params_algo=None, data_fidelity=None, prior=None, cost_fn=None, g_first=False, bregman_potential=None, \*\*kwargs)

Helper function for building an instance of the [`deepinv.unfolded.BaseDEQ`](https://deepinv.org/api/stubs/deepinv.unfolded.BaseDEQ.html.md#deepinv.unfolded.BaseDEQ) class.

#### Deprecated
Deprecated since version 0.3.6: The `DEQ_builder` function is deprecated and will be removed in future versions.
Instead of using this function, define a DEQ algorithm using the [`deepinv.optim.BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim) class with argument `DEQ=True`,
e.g. `model = PGD(data_fidelity, prior, ..., DEQ = True, ...)`.

#### NOTE
For now DEQ is only possible with PGD, HQS and GD optimization algorithms.

* **Parameters:**
  * **iteration** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*deepinv.optim.OptimIterator*](https://deepinv.org/api/stubs/deepinv.optim.OptimIterator.html.md#deepinv.optim.OptimIterator)) – either the name of the algorithm to be used,
    or directly an optim iterator.
    If an algorithm name (string), should be either `"PGD"` (proximal gradient descent), `"ADMM"` (ADMM),
    `"HQS"` (half-quadratic splitting), `"CP"` (Chambolle-Pock) or `"DRS"` (Douglas Rachford).
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
  * **cost_fn** (*Callable*) – Custom user input cost function. default: None.
  * **g_first** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to perform the step on $g$ before that on $f$ before or not. default: False
  * **bregman_potential** ([*deepinv.optim.Bregman*](https://deepinv.org/api/stubs/deepinv.optim.Bregman.html.md#deepinv.optim.Bregman)) – Bregman potential used for Bregman optimization algorithms such as Mirror Descent. Default: None, comes back to standard Euclidean optimization.
  * **kwargs** – additional arguments to be passed to the [`deepinv.unfolded.BaseUnfold`](https://deepinv.org/api/stubs/deepinv.unfolded.BaseUnfold.html.md#deepinv.unfolded.BaseUnfold) class.
