# BaseOptim

### *class* deepinv.optim.BaseOptim(iterator, params_algo=MappingProxyType({'lambda': 1.0, 'stepsize': 1.0}), data_fidelity=None, prior=None, max_iter=100, crit_conv='residual', thres_conv=1e-5, early_stop=False, has_cost=False, backtracking=None, custom_metrics=None, custom_init=None, get_output=lambda X: ..., unfold=False, trainable_params=None, DEQ=None, anderson_acceleration=False, verbose=False, show_progress_bar=False, \*\*kwargs)

Bases: [`Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor)

Class for optimization algorithms, consists in iterating a fixed-point operator.

Module solving the problem

$$
\begin{equation}
\label{eq:min_prob}
\tag{1}
\underset{x}{\arg\min} \quad  \datafid{x}{y} + \lambda \reg{x},
\end{equation}

$$

where the first term $\datafidname:\xset\times\yset \mapsto \mathbb{R}_{+}$ enforces data-fidelity, the second
term $\regname:\xset\mapsto \mathbb{R}_{+}$ acts as a regularization and
$\lambda > 0$ is a regularization parameter. More precisely, the data-fidelity term penalizes the discrepancy
between the data $y$ and the forward operator $A$ applied to the variable $x$, as

$$
\datafid{x}{y} = \distance{Ax}{y}

$$

where $\distance{\cdot}{\cdot}$ is a distance function, and where $A:\xset\mapsto \yset$ is the forward
operator (see [`deepinv.physics.Physics`](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics))

Optimization algorithms for minimising the problem above can be written as fixed point algorithms,
i.e. for $k=1,2,...$

$$
\qquad (x_{k+1}, z_{k+1}) = \operatorname{FixedPoint}(x_k, z_k, f, g, A, y, ...)

$$

where $x_k$ is a variable converging to the solution of the minimization problem, and
$z_k$ is an additional “dual” variable that may be required in the computation of the fixed point operator.

If the algorithm is minimizing an explicit and fixed cost function $F(x) =  \datafid{x}{y} + \lambda \reg{x}$,
the value of the cost function is computed along the iterations and can be used for convergence criterion.
Moreover, backtracking can be used to adapt the stepsize at each iteration. Backtracking consists in choosing
the largest stepsize $\tau$ such that, at each iteration, sufficient decrease of the cost function $F$ is achieved.
More precisely, Given $\gamma \in (0,1/2)$ and $\eta \in (0,1)$ and an initial stepsize $\tau > 0$,
the following update rule is applied at each iteration $k$:

$$
\text{ while } F(x_k) - F(x_{k+1}) < \frac{\gamma}{\tau} || x_{k-1} - x_k ||^2, \,\, \text{ do } \tau \leftarrow \eta \tau

$$

#### NOTE
To use backtracking, the optimized function (i.e., both the the data-fidelity and prior) must be explicit and provide a computable cost for the current iterate.
If the prior is not explicit (e.g. a denoiser) or if the argument `has_cost` is set to `False`, backtracking is automatically disabled.

The variable `params_algo` is a dictionary containing all the relevant parameters for running the algorithm.
If the value associated with the key is a float, the algorithm will use the same parameter across all iterations.
If the value is list of length max_iter, the algorithm will use the corresponding parameter at each iteration.

By default, the intial iterates are initialized with the adjoint applied to the measurement $A^{\top}y$, when the adjoint is defined, and with the observation $y$ if the adjoint is not defined.
Custom initialization can be defined with the `custom_init` class argument or via `init` argument in the `forward` method.

The variable `data_fidelity` is a list of instances of [`deepinv.optim.DataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity) (or a single instance).
If a single instance, the same data-fidelity is used at each iteration. If a list, the data-fidelity can change at each iteration.
The same holds for the variable `prior` which is a list of instances of [`deepinv.optim.Prior`](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior) (or a single instance).

Setting `unfold` to `True` enables to turn this iterative optimization algorithm into an unfolded algorithm, i.e. an algorithm
that can be trained end-to-end, with learnable parameters. These learnable parameters encompass the trainable parameters of the algorithm which
can be chosen with the `trainable_params` argument
(e.g. `stepsize` $\gamma$, regularization parameter `lambda_reg` $\lambda$, prior parameter (`g_param` or `sigma_denoiser`) $\sigma$ …)
but also the trainable priors (e.g. a deep denoiser) or forward models.

If `DEQ` is set to `True`, the algorithm is unfolded as a Deep Equilibrium model, i.e. the algorithm is virtually unrolled infinitely, leveraging the implicit function theorem.
The backward pass is then performed using fixed point iterations to find solutions of the fixed-point equation

$$
\begin{equation}
v = \left(\frac{\partial \operatorname{FixedPoint}(x^\star)}{\partial x^\star} \right )^{\top} v + u.
\end{equation}
$$

where $u$ is the incoming gradient from the backward pass,
and $x^\star$ is the equilibrium point of the forward pass. See [this tutorial](http://implicit-layers-tutorial.org/deep_equilibrium_models/) for more details.

Note also that by default, if the prior has trainable parameters (e.g. a neural network denoiser), these parameters are tranable by default.

#### NOTE
For now DEQ is only possible with PGD, HQS and GD optimization algorithms.
If the model is used for inference only, use the `with torch.no_grad():` context when calling the model in order to avoid unnecessary gradient computations.

```pycon
>>> import deepinv as dinv
>>> # This minimal example shows how to use the BaseOptim class to solve the problem
>>> #                min_x 0.5  ||Ax-y||_2^2 + \lambda ||x||_1
>>> # with the PGD algorithm, where A is the identity operator, lambda = 1 and y = [2, 2].
>>>
>>> # Create the measurement operator A
>>> A = torch.tensor([[1, 0], [0, 1]], dtype=torch.float64)
>>> A_forward = lambda v: A @ v
>>> A_adjoint = lambda v: A.transpose(0, 1) @ v
>>>
>>> # Define the physics model associated to this operator
>>> physics = dinv.physics.LinearPhysics(A=A_forward, A_adjoint=A_adjoint)
>>>
>>> # Define the measurement y
>>> y = torch.tensor([2, 2], dtype=torch.float64)
>>>
>>> # Define the data fidelity term
>>> data_fidelity = dinv.optim.data_fidelity.L2()
>>>
>>> # Define the prior
>>> prior = dinv.optim.Prior(g = lambda x, *args: torch.linalg.vector_norm(x, ord=1, dim=tuple(range(1, x.ndim))))
>>>
>>> # Define the parameters of the algorithm
>>> params_algo = {"stepsize": 0.5, "lambda": 1.0}
>>>
>>> # Define the fixed-point iterator
>>> iterator = dinv.optim.optim_iterators.PGDIteration()
>>>
>>> # Define the optimization algorithm
>>> optimalgo = dinv.optim.BaseOptim(iterator,
...                     data_fidelity=data_fidelity,
...                     params_algo=params_algo,
...                     prior=prior)
>>>
>>> # Run the optimization algorithm
>>> with torch.no_grad(): xhat = optimalgo(y, physics)
>>> print(xhat)
tensor([1., 1.], dtype=torch.float64)
```

* **Parameters:**
  * **iterator** ([*deepinv.optim.OptimIterator*](https://deepinv.org/api/stubs/deepinv.optim.OptimIterator.html.md#deepinv.optim.OptimIterator)) – Fixed-point iterator of the optimization algorithm of interest.
  * **params_algo** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – dictionary containing all the relevant parameters for running the algorithm,
    e.g. the stepsize, regularization parameter, denoising standard deviation.
    Each value of the dictionary can be either Iterable (distinct value for each iteration) or
    a single float (same value for each iteration).
    Default: `{"stepsize": 1.0, "lambda": 1.0}`. See [Optimization Parameters](https://deepinv.org/user_guide/reconstruction/optimization.html.md#optim-params) for more details.
  * **data_fidelity** ([*deepinv.optim.DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*deepinv.optim.DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity) *]*) – data-fidelity term.
    Either a single instance (same data-fidelity for each iteration) or a list of instances of
    [`deepinv.optim.DataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity) (distinct data fidelity for each iteration). Default: `None` corresponding to $\datafid{x}{y} = 0$.
  * **prior** ([*deepinv.optim.Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*deepinv.optim.Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior) *]*) – regularization prior.
    Either a single instance (same prior for each iteration) or a list of instances of
    [`deepinv.optim.Prior`](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior) (distinct prior for each iteration). Default: `None` corresponding to $\reg{x} = 0$.
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of iterations of the optimization algorithm. Default: 100.
  * **crit_conv** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – convergence criterion to be used for claiming convergence, either `"residual"` (residual
    of the iterate norm) or `"cost"` (on the cost function). Default: `"residual"`
  * **thres_conv** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – value of the threshold for claiming convergence. Default: `1e-05`.
  * **early_stop** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to stop the algorithm once the convergence criterion is reached. Default: `True`.
  * **has_cost** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether the algorithm has an explicit cost function or not. Default: `False`.
    If the prior is not explicit (e.g. a denoiser) `prior.explicit_prior = False`, then `has_cost` is automatically set to `False`.
  * **custom_metrics** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*deepinv.loss.metric.Metric*](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric) *]*) – dictionary containing custom metrics to be computed at each iteration.
  * **backtracking** ([*BacktrackingConfig*](https://deepinv.org/api/stubs/deepinv.optim.BacktrackingConfig.html.md#deepinv.optim.BacktrackingConfig) *,* [*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – configuration for using a backtracking line-search strategy for automatic stepsize adaptation.
    If `None` (default) or `False`, stepsize backtracking is disabled. Otherwise, `backtracking` must be an instance of [`deepinv.optim.BacktrackingConfig`](https://deepinv.org/api/stubs/deepinv.optim.BacktrackingConfig.html.md#deepinv.optim.BacktrackingConfig), which defines the parameters for backtracking line-search.
    If `True`, the default `BacktrackingConfig` is used.
  * **custom_init** (*Callable*) – 

    Custom initialization of the algorithm.
    The callable function `custom_init(y, physics)` takes as input the measurement $y$ and the physics `physics` and returns the initialization in the form of either:
    - a tuple $(x_0, z_0)$ (where `x_0` and `z_0` are the initial primal and dual variables),
    - a torch.Tensor $x_0$ (if no dual variables $z_0$ are used), or
    - a dictionary of the form `X = {'est': (x_0, z_0)}`.

    Note that custom initialization can also be directly defined via the `init` argument in the `forward` method.

    If `None` (default value), the algorithm is initialized with the adjoint $A^{\top}y$ when the adjoint is defined,
    and with the observation `y` if the adjoint is not defined. Default: `None`.
  * **get_output** (*Callable*) – Custom output of the algorithm.
    The callable function `get_output(X)` takes as input the dictionary `X` containing the primal and auxiliary variables and returns the desired output. Default : `X['est'][0]`.
  * **unfold** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to unfold the algorithm and make the model parameters trainable. Default: `False`.
  * **trainable_params** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *]*) – list of the algorithmic parameters to be made trainable (must be chosen among the keys of the dictionary `params_algo`).
    Default: `None`, which means that all parameters in params_algo are trainable. For no trainable parameters, set to an empty list `[]`.
  * **DEQ** ([*DEQConfig*](https://deepinv.org/api/stubs/deepinv.optim.DEQConfig.html.md#deepinv.optim.DEQConfig) *,* [*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Configuration for a Deep Equilibrium (DEQ) unfolding strategy.
    DEQ algorithms are virtually unrolled infinitely, leveraging the implicit function theorem.
    If `None` (default) or `False`, DEQ is disabled and the algorithm runs a standard finite number of iterations.
    Otherwise, `DEQ` must be an instance of [`deepinv.optim.DEQConfig`](https://deepinv.org/api/stubs/deepinv.optim.DEQConfig.html.md#deepinv.optim.DEQConfig), which defines the parameters
    for forward and backward equilibrium-based implicit differentiation.
    If `True`, the default `DEQConfig` is used.
  * **anderson_acceleration** ([*AndersonAccelerationConfig*](https://deepinv.org/api/stubs/deepinv.optim.AndersonAccelerationConfig.html.md#deepinv.optim.AndersonAccelerationConfig) *,* [*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Configuration of Anderson acceleration for the fixed-point iterations.
    If `None` (default) or `False`, Anderson acceleration is disabled.
    Otherwise, `anderson_acceleration` must be an instance of [`deepinv.optim.AndersonAccelerationConfig`](https://deepinv.org/api/stubs/deepinv.optim.AndersonAccelerationConfig.html.md#deepinv.optim.AndersonAccelerationConfig), which defines the parameters for Anderson acceleration.
    If `True`, the default `AndersonAccelerationConfig` is used.
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to print relevant information of the algorithm during its run,
    such as convergence criterion at each iterate. Default: `False`.
  * **show_progress_bar** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – show progress bar during optimization.
* **Returns:**
  a torch model that solves the optimization problem.

#### DEQ_additional_step(X, y, physics, \*\*kwargs)

For Deep Equilibrium models, performs an additional step at the equilibrium point
to compute the gradient of the fixed point operator with respect to the input.

* **Parameters:**
  * **X** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – dictionary defining the current update at the equilibrium point.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurement vector.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – physics of the problem for the acquisition of `y`.

#### backtracking_check_fn(X_prev, X)

Performs stepsize backtracking if the sufficient decrease condition is not verified.

* **Parameters:**
  * **X_prev** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – dictionary containing the primal and dual previous iterates.
  * **X** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – dictionary containing the current primal and dual iterates.

#### check_conv_fn(it, X_prev, X)

Checks the convergence of the algorithm.

* **Parameters:**
  * **it** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – iteration number.
  * **X_prev** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – dictionary containing the primal and dual previous iterates.
  * **X** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – dictionary containing the current primal and dual iterates.
* **Return bool:**
  `True` if the algorithm has converged, `False` otherwise.
* **Return type:**
  [bool](https://docs.python.org/3.9/library/functions.html#bool)

#### forward(y, physics, init=None, x_gt=None, compute_metrics=False, \*\*kwargs)

Runs the fixed-point iteration algorithm for solving [(1)](https://deepinv.org/user_guide/reconstruction/optimization.html.md#optim).

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurement vector.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – physics of the problem for the acquisition of `y`.
  * **init** (*Callable* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *,* [*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – 

    initialization of the algorithm. Default: `None`.
    if `None` (and the class `custom_init``argument is ``None`), the algorithm is initialized with the adjoint $A^{\top}y$ when the adjoint is defined, and with the observation `y` if the adjoint is not defined.
    `init` can be either a fixed initialization or a Callable function of the form `init(y, physics)` that takes as input
    the measurement $y$ and the physics `physics`. The output of the function or the fixed initialization can be either:
    - a tuple $(x_0, z_0)$ (where `x_0` and `z_0` are the initial primal and dual variables),
    - a [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) $x_0$ (if no dual variables $z_0$ are used), or
    - a dictionary of the form `X = {'est': (x_0, z_0)}`.

    Note that custom initialization can also be defined via the `custom_init` class argument.
  * **x_gt** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – (optional) ground truth image, for plotting the PSNR across optim iterations.
  * **compute_metrics** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to compute the metrics or not. Default: `False`.
  * **kwargs** – optional keyword arguments for the optimization iterator (see [`deepinv.optim.OptimIterator`](https://deepinv.org/api/stubs/deepinv.optim.OptimIterator.html.md#deepinv.optim.OptimIterator))
* **Returns:**
  If `compute_metrics` is `False`,  returns ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) the output of the algorithm. Else, returns (torch.Tensor, dict) the output of the algorithm and the metrics.

#### init_iterate_fn(y, physics, init=None, cost_fn=None)

Initializes the iterate of the algorithm.
The first iterate is stored in a dictionary of the form `X = {'est': (x_0, u_0), 'cost': F_0}` where:

> * `est` is a tuple containing the first primal and auxiliary iterates.
> * `cost` is the value of the cost function at the first iterate.

By default, the first (primal and dual) iterate of the algorithm is chosen as $A^{\top}y$ when the adjoint is defined, and with the observation `y` if the adjoint is not defined.
A custom initialization is possible via the `custom_init` class argument or via the `init` argument.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurement vector.
  * **deepinv.physics** – physics of the problem.
  * **init** (*Callable* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *,* [*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – 

    initialization of the algorithm.
    Either a Callable function of the form `init(y, physics)` or a fixed torch.Tensor initialization.
    The output of the function or the fixed initialization can be either:
    - a tuple $(x_0, z_0)$ (where `x_0` and `z_0` are the initial primal and dual variables),
    - a [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) $x_0$ (if no dual variables $z_0$ are used), or
    - a dictionary of the form `X = {'est': (x_0, z_0)}`.
  * **cost_fn** (*Callable*) – function that computes the cost function.
    `cost_fn(x, data_fidelity, prior, cur_params, y, physics)` takes as input
    the current primal variable ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)), the current data-fidelity ([`deepinv.optim.DataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)),
    the current prior ([`deepinv.optim.Prior`](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)), the current parameters (dict), and the measurement ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)).
    Default: `None`.
* **Returns:**
  a dictionary containing the first iterate of the algorithm.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)

#### init_metrics_fn(X_init, x_gt=None)

Initializes the metrics.

Metrics are computed for each batch and for each iteration.
They are represented by a list of list, and `metrics[metric_name][i,j]` contains the metric `metric_name`
computed for batch i, at iteration j.

* **Parameters:**
  * **X_init** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – dictionary containing the primal and auxiliary initial iterates.
  * **x_gt** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – ground truth image, required for PSNR computation. Default: `None`.
* **Return dict:**
  A dictionary containing the metrics.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)[[str](https://docs.python.org/3.9/library/stdtypes.html#str), [list](https://docs.python.org/3.9/library/stdtypes.html#list)]

#### update_data_fidelity_fn(it)

For each data_fidelity function in `data_fidelity`, selects the data_fidelity value for iteration `it`
(if this data_fidelity depends on the iteration number).

* **Parameters:**
  **it** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – iteration number.
* **Returns:**
  the data_fidelity at iteration `it`.
* **Return type:**
  [*DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)

#### update_metrics_fn(metrics, X_prev, X, x_gt=None)

Function that compute all the metrics, across all batches, for the current iteration.

* **Parameters:**
  * **metrics** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – dictionary containing the metrics. Each metric is computed for each batch.
  * **X_prev** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – dictionary containing the primal and dual previous iterates.
  * **X** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – dictionary containing the current primal and dual iterates.
  * **x_gt** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – ground truth image, required for PSNR computation. Default: None.
* **Return dict:**
  a dictionary containing the updated metrics.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)[[str](https://docs.python.org/3.9/library/stdtypes.html#str), [list](https://docs.python.org/3.9/library/stdtypes.html#list)]

#### update_params_fn(it)

For each parameter `params_algo`, selects the parameter value for iteration `it`
(if this parameter depends on the iteration number).

* **Parameters:**
  **it** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – iteration number.
* **Returns:**
  a dictionary containing the parameters at iteration `it`.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)[[str](https://docs.python.org/3.9/library/stdtypes.html#str), [float](https://docs.python.org/3.9/library/functions.html#float) | [*Iterable*](https://docs.python.org/3.9/library/collections.abc.html#collections.abc.Iterable)]

#### update_prior_fn(it)

For each prior function in `prior`, selects the prior value for iteration `it`
(if this prior depends on the iteration number).

* **Parameters:**
  **it** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – iteration number.
* **Returns:**
  the prior at iteration `it`.
* **Return type:**
  [*Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)

<a id="sphx-glr-backref-deepinv-optim-baseoptim"></a>

## Examples using `BaseOptim`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using an iterative algorithm.">  <div class="sphx-glr-thumbnail-title">Use iterative reconstruction algorithms</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using a pretrained model in one line.">  <div class="sphx-glr-thumbnail-title">Use a pretrained model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to get started with DeepInverse in under 5 minutes.">  <div class="sphx-glr-thumbnail-title">5 minute quickstart tutorial</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates blind image deblurring using the pretrained kernel estimation network from the paper :footcitecarbajal2023blind. The network estimates spatially-varying blur kernels from a blurred image, which are then used in a space-varying blur physics model to reconstruct the sharp image using a non-blind deblurring algorithm.">  <div class="sphx-glr-thumbnail-title">Blind deblurring with kernel estimation network</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In many large-scale imaging problems, the size of the image/volume to reconstruct is very large, making it impossible to train reconstruction networks (in this example, unfolded networks) with a single GPU. The deepinv.distributed framework enables training a model on multiple GPUs, by carefully parallelizing the data fidelity and denoising steps inside the network.">  <div class="sphx-glr-thumbnail-title">Distributed Training of Unfolded Networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the Astra tomography toolbox with deepinv, a popular toolbox for tomography with GPU acceleration.">  <div class="sphx-glr-thumbnail-title">Low-dose CT with ASTRA backend and Total-Variation (TV) prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use Spyrit linear models and measurements with DeepInverse. Here we use the HadamSplit2d linear model from Spyrit.">  <div class="sphx-glr-thumbnail-title">Single-pixel imaging with Spyrit</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example, we investigate a simple 2D Radio Interferometry (RI) imaging task with deepinverse. The following example and data are taken from :footciteaghabiglou2024r2d2. If you are interested in RI imaging problem and would like to see more examples or try the state-of-the-art algorithms, please check BASPLib.">  <div class="sphx-glr-thumbnail-title">Radio interferometric imaging with deepinverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard TV prior for image deblurring. The problem writes as y = Ax + \\epsilon where A is a convolutional operator and \\epsilon is the realization of some Gaussian noise. The goal is to recover the original image x from the blurred and noisy image y. The TV prior is used to regularize the problem.">  <div class="sphx-glr-thumbnail-title">Image deblurring with Total-Variation (TV) prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example, we show how to solve a deblurring inverse problem using an explicit prior.">  <div class="sphx-glr-thumbnail-title">Image deblurring with custom deep explicit prior.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a full-resolution multispectral image from raw snapshot mosaiced measurements using various reconstruction algorithms.">  <div class="sphx-glr-thumbnail-title">Multispectral demosaicing from raw sensor data</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to solve Poisson inverse problems using the Maximum-Likelihood Expectation-Maximization (MLEM) algorithm :footcitesheppMaximumLikelihoodReconstruction1982, also known as the Richardson-Lucy algorithm in the deconvolution setting :footciterichardsonBayesianBasedIterativeMethod1972,lucyIterativeTechniqueRectification1974.">  <div class="sphx-glr-thumbnail-title">Poisson Inverse Problems with Maximum-Likelihood Expectation-Maximization (MLEM)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard wavelet prior for image inpainting. The problem writes as y = Ax + \\epsilon where A is a mask and \\epsilon is the realization of some Gaussian noise. The goal is to recover the original image x from the blurred and noisy image y. The wavelet prior is used to regularize the problem.">  <div class="sphx-glr-thumbnail-title">Image inpainting with wavelet prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct an image from them.">  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 2D</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct a volume from them.">  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 3D</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to create a random phase retrieval operator and generate phaseless measurements from a given image. The example showcases 4 different reconstruction methods to recover the image from the phaseless measurements:">  <div class="sphx-glr-thumbnail-title">Random phase retrieval and reconstruction methods.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we show how to use the deepinv.physics.Scattering forward model.">  <div class="sphx-glr-thumbnail-title">Inverse scattering problem</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows the use of the deepinv.physics.SpatialUnwrapping forward model and the deepinv.optim.ItohFidelity for unwrapping problems, which occur in modulo imaging, interferometry SAR and other imaging applications. It shows how to generate a wrapped phase image, apply blur and noise, and reconstruct the original phase using both DCT inversion and ADMM optimization.">  <div class="sphx-glr-thumbnail-title">Spatial unwrapping and modulo imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo illustrates the impact of different Hadamard pattern ordering algorithms in the Single Pixel Camera (SPC), a computational imaging system that uses a single photodetector to capture images by projecting the scene onto a series of patterns. The SPC is implemented in the DeepInverse library.">  <div class="sphx-glr-thumbnail-title">Pattern Ordering in a Compressive Single Pixel Camera</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the DPIR method to solve a PnP image deblurring problem. The DPIR method is described in :footcitezhang2021plug. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3929-3938).">  <div class="sphx-glr-thumbnail-title">DPIR method for PnP image deblurring.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to define your own optimization algorithm. For example, here, we implement the Primal-Dual Condat-Vu (CV) algorithm, and apply it for Single Pixel Camera reconstruction.">  <div class="sphx-glr-thumbnail-title">PnP with custom optimization algorithm (Primal-Dual Condat-Vu)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use a mirror descent algorithm for solving an inverse problem with Poisson noise. See :footcitehurault2023convergent for more details.">  <div class="sphx-glr-thumbnail-title">Plug-and-Play algorithm with Mirror Descent for Poisson noise inverse problems.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Plug-and-Play (PnP) is known to be challenging to apply to certain inverse problems like inpainting. One way to overcome this is to use a multi-scale approach instead of the standard single-scale approach.">  <div class="sphx-glr-thumbnail-title">Multi-scale Plug-and-Play for Inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Implementation of :footciteromano2017little using as plug-in denoiser the Gradient-Step Denoiser (GSPnP) :footcitehurault2021gradient which provides an explicit prior.">  <div class="sphx-glr-thumbnail-title">Regularization by Denoising (RED) for Super-Resolution.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard PnP algorithm with DnCNN denoiser for computed tomography.">  <div class="sphx-glr-thumbnail-title">Vanilla PnP for computed tomography (CT).</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the self-supervised Artifact2Artifact loss for solving an undersampled sequential MRI reconstruction problem without ground truth.">  <div class="sphx-glr-thumbnail-title">Self-supervised MRI reconstruction with Artifact2Artifact</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a reconstruction network for an MRI inverse problem on a fully self-supervised way, i.e., using measurement data only.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Imaging for MRI.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This a toy example to show you how to use DEQ to solve a deblurring problem. Note that this is a small dataset for training. For optimal results, use a larger dataset.">  <div class="sphx-glr-thumbnail-title">Deep Equilibrium (DEQ) algorithms for image deblurring</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to implement the LISTA algorithm :footcitegregor2010learning, for a compressed sensing problem. In a nutshell, LISTA is an unfolded proximal gradient algorithm involving a soft-thresholding proximal operator with learnable thresholding parameters.">  <div class="sphx-glr-thumbnail-title">Learned Iterative Soft-Thresholding Algorithm (LISTA) for compressed sensing</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to implement a learned unrolled proximal gradient descent algorithm with a custom prior function. The custom prior in use is The algorithm is trained on a dataset of compressed sensing measurements of MNIST images.">  <div class="sphx-glr-thumbnail-title">Learned iterative custom prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="where both the data fidelity and the prior are learned modules, distinct for each iterations.">  <div class="sphx-glr-thumbnail-title">Learned Primal-Dual algorithm for CT scan.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Some unfolded architectures rely on a least-squares solver &lt;deepinv.optim.linear.least_squares&gt; to compute the proximal step w.r.t. the data-fidelity term (e.g., deepinv.optim.optim_iterators.ADMMIteration or deepinv.optim.optim_iterators.HQSIteration):  ">  <div class="sphx-glr-thumbnail-title">Reducing the memory and computational complexity of unfolded network training</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Image inpainting consists in solving y = Ax where A is a mask operator. This problem can be reformulated as the following minimization problem:">  <div class="sphx-glr-thumbnail-title">Unfolded Chambolle-Pock for constrained image inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use vanilla unfolded Plug-and-Play. The DnCNN denoiser and the algorithm parameters (stepsize, regularization parameters) are trained jointly. For simplicity, we show how to train the algorithm on a  small dataset. For optimal results, use a larger dataset.">  <div class="sphx-glr-thumbnail-title">Vanilla Unfolded algorithm for super-resolution</div>
</div>
<!-- thumbnail-parent-div-close --></div>
