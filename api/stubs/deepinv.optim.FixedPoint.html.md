# FixedPoint

### *class* deepinv.optim.FixedPoint(iterator=None, update_params_fn=None, update_data_fidelity_fn=None, update_prior_fn=None, init_iterate_fn=None, init_metrics_fn=None, update_metrics_fn=None, backtracking_check_fn=None, check_conv_fn=None, max_iter=50, early_stop=True, anderson_acceleration_config=None, backtracking_config=None, verbose=False, show_progress_bar=False)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Fixed-point iterations module.

This module implements the fixed-point iteration algorithm given a specific fixed-point iterator (e.g.
proximal gradient iteration, the ADMM iteration, see [Predefined Algorithms](https://deepinv.org/user_guide/reconstruction/optimization.html.md#optim-iterators)), that is
for $k=1,2,...$

$$
\qquad (x_{k+1}, u_{k+1}) = \operatorname{FixedPoint}(x_k, u_k, f, g, A, y, ...) \hspace{2cm} (1)

$$

where $f$ is the data-fidelity term, $g$ is the prior, $A$ is the physics model, $y$ is the data.

* **Examples:**
  This example shows how to use the [`FixedPoint`](#deepinv.optim.FixedPoint) class to solve the problem
  $\min_x 0.5*||Ax-y||_2^2 + \lambda*||x||_1$ with the PGD algorithm, where A is the identity operator,
  $\lambda = 1$ and $y = [2, 2]$.
  ```pycon
  >>> import deepinv as dinv
  >>> # Create the measurement operator A
  >>> A = torch.tensor([[1, 0], [0, 1]], dtype=torch.float64)
  >>> A_forward = lambda v: A @ v
  >>> A_adjoint = lambda v: A.transpose(0, 1) @ v
  >>> # Define the physics model associated to this operator
  >>> physics = dinv.physics.LinearPhysics(A=A_forward, A_adjoint=A_adjoint)
  >>> # Define the measurement y
  >>> y = torch.tensor([2, 2], dtype=torch.float64)
  >>> # Define the data fidelity term
  >>> data_fidelity = dinv.optim.data_fidelity.L2()
  >>> # Define the prior term
  >>> prior = dinv.optim.prior.L1Prior()
  >>> # Define the parameters of the algorithm
  >>> params_algo = {"g_param": 1.0, "stepsize": 1.0, "lambda": 1.0, "beta": 1.0}
  >>> # Choose the iterator associated to the PGD algorithm
  >>> iterator = dinv.optim.optim_iterators.PGDIteration()
  >>> # Iterate the iterator
  >>> x_init = torch.tensor([2, 2], dtype=torch.float64)  # Define initialisation of the algorithm
  >>> X = {"est": (x_init ,), "cost": []}                 # Iterates are stored in a dictionary of the form {'est': (x,z), 'cost': F}
  >>> max_iter = 50
  >>> for it in range(max_iter):
  ...     X = iterator(X, data_fidelity, prior, params_algo, y, physics)
  >>> # Return the solution
  >>> X["est"][0]
  tensor([1., 1.], dtype=torch.float64)
  ```
* **Parameters:**
  * **iterator** ([*deepinv.optim.OptimIterator*](https://deepinv.org/api/stubs/deepinv.optim.OptimIterator.html.md#deepinv.optim.OptimIterator)) – function that takes as input the current iterate, as
    well as parameters of the optimization problem (prior, measurements, etc.)
  * **update_params_fn** (*Callable*) – function that returns the parameters to be used at each iteration. Default: `None`.
  * **update_prior_fn** (*Callable*) – function that returns the prior to be used at each iteration. Default: `None`.
  * **init_iterate_fn** (*Callable*) – function that returns the initial iterate. Default: `None`.
  * **init_metrics_fn** (*Callable*) – function that returns the initial metrics. Default: `None`.
  * **backtracking_check_fn** (*Callable*) – function that performs a sufficent decrease check on the last iteration and returns a bool indicating if we can proceed to next iteration. Default: `None`.
  * **check_conv_fn** (*Callable*) – function that checks the convergence after each iteration, returns a bool indicating if convergence has been reached. Default: `None`.
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of iterations. Default: `50`.
  * **early_stop** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if True, the algorithm stops when the convergence criterion is reached. Default: `True`.
  * **anderson_acceleration_config** ([*deepinv.optim.AndersonAccelerationConfig*](https://deepinv.org/api/stubs/deepinv.optim.AndersonAccelerationConfig.html.md#deepinv.optim.AndersonAccelerationConfig)) – parameters for Anderson acceleration of the fixed-point iterations.
  * **backtracking_config** ([*deepinv.optim.BacktrackingConfig*](https://deepinv.org/api/stubs/deepinv.optim.BacktrackingConfig.html.md#deepinv.optim.BacktrackingConfig)) – parameters for backtracking line-search stepsize strategy.
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if True, various convergence information are printed during the iterations. Default: `False`.
  * **show_progress_bar** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if True, a progress bar is displayed during the iterations. Default: `False`.

#### anderson_acceleration_step(it, X_prev, TX_prev, cur_data_fidelity, cur_prior, cur_params, \*args)

Anderson acceleration step.

Code inspired from [this tutorial](http://implicit-layers-tutorial.org/deep_equilibrium_models/).

* **Parameters:**
  * **it** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – current iteration.
  * **X_prev** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – previous iterate.
  * **TX_prev** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – output of the fixed-point operator evaluated at X_prev
  * **cur_data_fidelity** ([*deepinv.optim.DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)) – Instance of the DataFidelity class defining the current data_fidelity.
  * **cur_prior** ([*deepinv.optim.Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)) – Instance of the Prior class defining the current prior.
  * **cur_params** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – Dictionary containing the current parameters of the algorithm.
  * **args** – arguments for the iterator.
* **Return dict:**
  new iterates after Anderson acceleration step (same form as input iterates X_prev and TX_prev).
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)[[str](https://docs.python.org/3.9/library/stdtypes.html#str), [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)] | [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]

#### forward(\*args, init=None, compute_metrics=False, x_gt=None, \*\*kwargs)

Loops over the fixed-point iterator as (1) and returns the fixed point.

The iterates are stored in a dictionary of the form `X = {'est': (x_k, u_k), 'cost': F_k}` where:

> * `est` is a tuple containing the current primal and auxiliary iterates,
> * `cost` is the value of the cost function at the current iterate.

Since the prior and parameters (stepsize, regularisation parameter, etc.) can change at each iteration,
the prior and parameters are updated before each call to the iterator.

* **Parameters:**
  * **init** (*Callable* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *,* [*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – 

    initialization of the algorithm.
    Either a Callable function of the form `init(y, physics)` or a fixed torch.Tensor initialization.
    The output of the function or the fixed initialization can be either:
    - a tuple $(x_0, z_0)$ (where `x_0` and `z_0` are the initial primal and dual variables),
    - a [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) $x_0$ (if no dual variables $z_0$ are used), or
    - a dictionary of the form `X = {'est': (x_0, z_0)}`.
  * **compute_metrics** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, the metrics are computed along the iterations. Default: `False`.
  * **x_gt** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – ground truth solution. Default: `None`.
  * **args** – optional arguments for the iterator. Commonly (y,physics) where `y` (torch.Tensor y) is the measurement and
    `physics` (deepinv.physics) is the physics model.
  * **kwargs** – optional keyword arguments for the iterator.
* **Return tuple:**
  `(x,metrics)` with `x` the fixed-point solution (dict) and `metrics` the computed along the iterations if `compute_metrics` is `True` or `None` otherwise.
* **Return type:**
  [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[dict](https://docs.python.org/3.9/library/stdtypes.html#dict)[[str](https://docs.python.org/3.9/library/stdtypes.html#str), [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)] | [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)], [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)[[str](https://docs.python.org/3.9/library/stdtypes.html#str), [list](https://docs.python.org/3.9/library/stdtypes.html#list)] | None]

#### init_anderson_acceleration(X)

Initialize the Anderson acceleration algorithm.
Code inspired from [this tutorial](http://implicit-layers-tutorial.org/deep_equilibrium_models/).

Initializes the history buffers and the H and q matrices used in Anderson acceleration.
H and q are used to solve the linear system Hp = q at each Anderson acceleration step.
H is initialized as a (m+1)x(m+1) matrix where m is the history size, with first row and column set to 1.
Q is initialized as a (m+1)x1 vector with first element set to 1.

* **Parameters:**
  **X** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – initial iterate.

#### single_iteration(X, it, \*args, \*\*kwargs)

Performs a single iteration of the fixed-point algorithm, including Anderson acceleration and backtracking if specified.

The iterates are stored in a dictionary of the form `X = {'est': (x_k, u_k), 'cost': F_k}` where:

> * `est` is a tuple containing the current primal and auxiliary iterates,
> * `cost` is the value of the cost function at the current iterate.
* **Parameters:**
  * **X** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – current iterate of the algorithm.
  * **it** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – current iteration number.
* **Returns:**
  new iterate after one step of the algorithm.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)[[str](https://docs.python.org/3.9/library/stdtypes.html#str), [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)] | [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]
