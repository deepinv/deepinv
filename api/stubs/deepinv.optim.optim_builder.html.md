# optim_builder

### deepinv.optim.optim_builder(iteration, max_iter=100, params_algo=MappingProxyType({'lambda': 1.0, 'stepsize': 1.0, 'g_param': 0.05}), data_fidelity=None, prior=None, cost_fn=None, g_first=False, bregman_potential=None, \*\*kwargs)

> Helper function for building an instance of the [`deepinv.optim.BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim) class.

#### NOTE
> Since 0.3.6, instead of using this function, it is possible to define optimization algorithms using directly the algorithm name e.g.
> `model = PGD(data_fidelity, prior, ...)`.
* **param str, deepinv.optim.OptimIterator iteration:**
  either the name of the algorithm to be used, or directly an optim iterator.
  If an algorithm name (string), should be either `"GD"` (gradient descent), `"PGD"` (proximal gradient descent), `"ADMM"` (ADMM),
  `"HQS"` (half-quadratic splitting), `"PDCP"` (Primal-Dual Chambolle-Pock),  `"DRS"` (Douglas Rachford), `"MD"` (Mirror Descent) or `"PMD"` (Proximal Mirror Descent).
* **param int max_iter:**
  maximum number of iterations of the optimization algorithm. Default: 100.
* **param dict params_algo:**
  dictionary containing all the relevant parameters for running the algorithm,
  e.g. the stepsize, regularization parameter, denoising standard deviation.
  Each value of the dictionary can be either Iterable (distinct value for each iteration) or
  a single float (same value for each iteration). See [Optimization Parameters](https://deepinv.org/user_guide/reconstruction/optimization.html.md#optim-params) for more details.
  Default: `{"stepsize": 1.0, "lambda": 1.0}`.
* **param list, deepinv.optim.DataFidelity:**
  data-fidelity term.
  Either a single instance (same data-fidelity for each iteration) or a list of instances of
  [`deepinv.optim.DataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity) (distinct data-fidelity for each iteration). Default: `None`.
* **param list, deepinv.optim.Prior prior:**
  regularization prior.
  Either a single instance (same prior for each iteration) or a list of instances of
  deepinv.optim.Prior (distinct prior for each iteration). Default: `None`.
* **param Callable cost_fn:**
  Custom user input cost function.
  `cost_fn(x, data_fidelity, prior, cur_params, y, physics)` takes as input
  the current primal variable ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)), the current data-fidelity ([`deepinv.optim.DataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)),
  the current prior ([`deepinv.optim.Prior`](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)), the current parameters (dict), and the measurement ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)).
  Default: `None`.
* **param bool g_first:**
  whether to perform the step on $g$ before that on $f$ before or not. Default: `False`
* **param deepinv.optim.Bregman bregman_potential:**
  Bregman potential used for Bregman optimization algorithms such as Mirror Descent. Default: `None`, uses standard Euclidean optimization.
* **param kwargs:**
  additional arguments to be passed to the [`deepinv.optim.BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim) class.
* **return:**
  an instance of the [`deepinv.optim.BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim) class.

## Examples using `optim_builder`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use Spyrit linear models and measurements with DeepInverse. Here we use the HadamSplit2d linear model from Spyrit.">  <div class="sphx-glr-thumbnail-title">Single-pixel imaging with Spyrit</div>
</div>
<!-- thumbnail-parent-div-close --></div>
