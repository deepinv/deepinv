# FISTA

### *class* deepinv.optim.FISTA(data_fidelity=None, prior=None, lambda_reg=1.0, stepsize=1.0, a=3, g_param=None, sigma_denoiser=None, max_iter=100, crit_conv='residual', thres_conv=1e-5, early_stop=False, custom_metrics=None, custom_init=None, g_first=False, unfold=False, trainable_params=None, cost_fn=None, params_algo=None, \*\*kwargs)

Bases: [`BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim)

FISTA module for acceleration of the Proximal Gradient Descent algorithm.
If the attribute `g_first` is set to False (by default), the FISTA iterations are given by

$$
u_{k} &= z_k -  \gamma \nabla f(z_k) \\
x_{k+1} &= \operatorname{prox}_{\gamma \lambda \regname}(u_k) \\
z_{k+1} &= x_{k+1} + \alpha_k (x_{k+1} - x_k),

$$

where $\gamma$ is a stepsize that should satisfy $\gamma \leq 1/\operatorname{Lip}(\|\nabla f\|)$ and
$\alpha_k = (k+a-1)/(k+a)$,  with $a$ a parameter that should be strictly greater than 2.

If the attribute `g_first` is set to True, the functions $f$ and $\regname$ are inverted in the previous iteration.
The FISTA iterations are defined in the iterator class [`deepinv.optim.optim_iterators.FISTAIteration`](https://deepinv.org/api/stubs/deepinv.optim.optim_iterators.FISTAIteration.html.md#deepinv.optim.optim_iterators.FISTAIteration).
For using early stopping or stepsize backtracking, see the documentation of the [`deepinv.optim.BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim) class.

If the attribute `unfold` is set to `True`, the algorithm is unfolded and the parameters of the algorithm are trainable.
By default, all the algorithm parameters are trainable : the stepsize $\gamma$, the regularization parameter $\lambda$, the prior parameter, and the parameter $a$ of the FISTA algorithm.
Use the `trainable_params` argument to adjust the list of trainable parameters.
Note also that by default, if the prior has trainable parameters (e.g. a neural network denoiser), these parameters are learnable by default.
If the model is used for inference only, use the `with torch.no_grad():` context when calling the model in order to avoid unnecessary gradient computations.

* **Parameters:**
  * **data_fidelity** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *,* [*deepinv.optim.DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)) – data-fidelity term $\datafid{x}{y}$.
    Either a single instance (same data-fidelity for each iteration) or a list of instances of
    [`deepinv.optim.DataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity) (distinct data fidelity for each iteration). Default: `None` corresponding to $\datafid{x}{y} = 0$.
  * **prior** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *,* [*deepinv.optim.Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)) – regularization prior $\reg{x}$.
    Either a single instance (same prior for each iteration) or a list of instances of
    [`deepinv.optim.Prior`](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior) (distinct prior for each iteration). Default: `None` corresponding to $\reg{x} = 0$.
  * **lambda_reg** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – regularization parameter $\lambda$. Default: `1.0`.
  * **stepsize** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – stepsize parameter $\gamma$. Default: `1.0`.
  * **a** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – parameter of the FISTA algorithm, should be strictly greater than 2. Default: `3`.
  * **g_param** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – parameter of the prior function. For example the noise level for a denoising prior. Default: `None`.
  * **sigma_denoiser** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – same as `g_param`. If both `g_param` and `sigma_denoiser` are provided, `g_param` is used. Default: `None`.
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of iterations of the optimization algorithm. Default: `100`.
  * **crit_conv** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – convergence criterion to be used for claiming convergence, either `"residual"` (residual
    of the iterate norm) or `"cost"` (on the cost function). Default: `"residual"`
  * **thres_conv** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – convergence threshold for the chosen convergence criterion. Default: `1e-5`.
  * **early_stop** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to stop the algorithm as soon as the convergence criterion is met. Default: `False`.
  * **custom_metrics** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – dictionary of custom metric functions to be computed along the iterations. The keys of the dictionary are the names of the metrics, and the values are functions that take as input the current and previous iterates, and return a scalar value. Default: `None`.
  * **custom_init** (*Callable*) – 

    Custom initialization of the algorithm.
    The callable function `custom_init(y, physics)` takes as input the measurement $y$ and the physics `physics` and returns the initialization in the form of either:
    - a tuple $(x_0, z_0)$ (where `x_0` and `z_0` are the initial primal and dual variables),
    - a torch.Tensor $x_0$ (if no dual variables $z_0$ are used), or
    - a dictionary of the form `X = {'est': (x_0, z_0)}`.

    Note that custom initialization can also be directly defined via the `init` argument in the `forward` method.

    If `None` (default value), the algorithm is initialized with the adjoint $A^{\top}y$ when the adjoint is defined,
    and with the observation `y` if the adjoint is not defined. Default: `None`.
  * **g_first** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to perform the proximal step on $\reg{x}$ before that on $\datafid{x}{y}$, or the opposite. Default: `False`.
  * **unfold** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to unfold the algorithm or not. Default: `False`.
  * **trainable_params** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list)) – list of FISTA parameters to be trained if `unfold` is True. To choose between `["lambda", "stepsize", "g_param", "a"]`. Default: None, which means that all parameters are trainable if `unfold` is True. For no trainable parameters, set to an empty list.
  * **cost_fn** (*Callable*) – Custom user input cost function.
    `cost_fn(x, data_fidelity, prior, cur_params, y, physics)` takes as input
    the current primal variable ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)), the current data-fidelity ([`deepinv.optim.DataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)),
    the current prior ([`deepinv.optim.Prior`](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)), the current parameters (dict), and the measurement ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)).
    Default: `None`.
  * **params_algo** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – optionally, directly provide the FISTA parameters in a dictionary. This will overwrite the parameters in the arguments `stepsize`, `lambda_reg`, `g_param` and `a`.

<a id="sphx-glr-backref-deepinv-optim-fista"></a>

## Examples using `FISTA`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In this example, we investigate a simple 2D Radio Interferometry (RI) imaging task with deepinverse. The following example and data are taken from :footciteaghabiglou2024r2d2. If you are interested in RI imaging problem and would like to see more examples or try the state-of-the-art algorithms, please check BASPLib.">  <div class="sphx-glr-thumbnail-title">Radio interferometric imaging with deepinverse</div>
</div>
<!-- thumbnail-parent-div-close --></div>
