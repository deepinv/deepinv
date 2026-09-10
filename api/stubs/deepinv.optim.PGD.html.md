# PGD

### *class* deepinv.optim.PGD(data_fidelity=None, prior=None, lambda_reg=1.0, stepsize=1.0, g_param=None, sigma_denoiser=None, max_iter=100, crit_conv='residual', thres_conv=1e-5, early_stop=False, backtracking=None, custom_metrics=None, custom_init=None, g_first=False, unfold=False, trainable_params=None, DEQ=None, anderson_acceleration=None, cost_fn=None, params_algo=None, \*\*kwargs)

Bases: [`BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim)

Proximal Gradient Descent (PGD) module for solving the problem

$$
\begin{equation}
\label{eq:min_prob}
\tag{1}
\underset{x}{\arg\min} \quad  \datafid{x}{y} + \lambda \reg{x},
\end{equation}

$$

where $\datafid{x}{y}$ is the data-fidelity term, $\reg{x}$ is the regularization term.
If the attribute `g_first` is set to False (by default), the PGD iterations are given by

$$
x_{k+1} = \operatorname{prox}_{\gamma \lambda \regname}(x_k - \gamma \nabla f(x_k)).

$$

If the attribute `g_first` is set to True, the functions $f$ and $\regname$ are inverted in the previous iteration.
The PGD iterations are defined in the iterator class [`deepinv.optim.optim_iterators.PGDIteration`](https://deepinv.org/api/stubs/deepinv.optim.optim_iterators.PGDIteration.html.md#deepinv.optim.optim_iterators.PGDIteration).
For using early stopping or stepsize backtracking, see the documentation of the [`deepinv.optim.BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim) class.

If the attribute `unfold` is set to `True`, the algorithm is unfolded and the parameters of the algorithm are trainable.
By default, all the algorithm parameters are trainable : the stepsize $\gamma$, the regularization parameter $\lambda$, the prior parameter.
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
  * **g_param** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – parameter of the prior function. For example the noise level for a denoising prior. Default: `None`.
  * **sigma_denoiser** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – same as `g_param`. If both `g_param` and `sigma_denoiser` are provided, `g_param` is used. Default: `None`.
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of iterations of the optimization algorithm. Default: `100`.
  * **crit_conv** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – convergence criterion to be used for claiming convergence, either `"residual"` (residual
    of the iterate norm) or `"cost"` (on the cost function). Default: `"residual"`
  * **thres_conv** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – convergence threshold for the chosen convergence criterion. Default: `1e-5`.
  * **early_stop** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to stop the algorithm as soon as the convergence criterion is met. Default: `False`.
  * **backtracking** ([*deepinv.optim.BacktrackingConfig*](https://deepinv.org/api/stubs/deepinv.optim.BacktrackingConfig.html.md#deepinv.optim.BacktrackingConfig) *,* [*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – configuration for using a backtracking line-search strategy for automatic stepsize adaptation.
    If None (default), stepsize backtracking is disabled. Otherwise, `backtracking` must be an instance of [`deepinv.optim.BacktrackingConfig`](https://deepinv.org/api/stubs/deepinv.optim.BacktrackingConfig.html.md#deepinv.optim.BacktrackingConfig), which defines the parameters for backtracking line-search.
    By default, backtracking is disabled (i.e., `backtracking=None`), and as soon as `backtracking` is not `None`, the default `BacktrackingConfig` is used.
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
  * **trainable_params** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list)) – list of PGD parameters to be trained if `unfold` is True. To choose between `["lambda", "stepsize", "g_param"]`. Default: None, which means that all parameters are trainable if `unfold` is True. For no trainable parameters, set to an empty list.
  * **DEQ** ([*deepinv.optim.DEQConfig*](https://deepinv.org/api/stubs/deepinv.optim.DEQConfig.html.md#deepinv.optim.DEQConfig) *,* [*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Configuration for a Deep Equilibrium (DEQ) unfolding strategy.
    DEQ algorithms are virtually unrolled infinitely, leveraging the implicit function theorem.
    If `None` (default) or `False`, DEQ is disabled and the algorithm runs a standard finite number of iterations.
    Otherwise, `DEQ` must be an instance of [`deepinv.optim.DEQConfig`](https://deepinv.org/api/stubs/deepinv.optim.DEQConfig.html.md#deepinv.optim.DEQConfig), which defines the parameters
    for forward and backward equilibrium-based implicit differentiation.
    If `True`, the default `DEQConfig` is used.
  * **anderson_acceleration** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Configure Anderson acceleration for the fixed-point iterations.
    If `None` (default) or `False`, Anderson acceleration is disabled.
    Otherwise, `anderson_acceleration` must be an instance of [`deepinv.optim.AndersonAccelerationConfig`](https://deepinv.org/api/stubs/deepinv.optim.AndersonAccelerationConfig.html.md#deepinv.optim.AndersonAccelerationConfig), which defines the parameters for Anderson acceleration.
    If `True`, the default `AndersonAccelerationConfig` is used.
  * **cost_fn** (*Callable*) – Custom user input cost function.
    `cost_fn(x, data_fidelity, prior, cur_params, y, physics)` takes as input
    the current primal variable ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)), the current data-fidelity ([`deepinv.optim.DataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)),
    the current prior ([`deepinv.optim.Prior`](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)), the current parameters (dict), and the measurement ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)).
    Default: `None`.
  * **params_algo** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – optionally, directly provide the PGD parameters in a dictionary. This will overwrite the parameters in the arguments `stepsize`, `lambda_reg` and `g_param`.

<a id="sphx-glr-backref-deepinv-optim-pgd"></a>

## Examples using `PGD`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using an iterative algorithm.">  <div class="sphx-glr-thumbnail-title">Use iterative reconstruction algorithms</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the Astra tomography toolbox with deepinv, a popular toolbox for tomography with GPU acceleration.">  <div class="sphx-glr-thumbnail-title">Low-dose CT with ASTRA backend and Total-Variation (TV) prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard TV prior for image deblurring. The problem writes as y = Ax + \\epsilon where A is a convolutional operator and \\epsilon is the realization of some Gaussian noise. The goal is to recover the original image x from the blurred and noisy image y. The TV prior is used to regularize the problem.">  <div class="sphx-glr-thumbnail-title">Image deblurring with Total-Variation (TV) prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a full-resolution multispectral image from raw snapshot mosaiced measurements using various reconstruction algorithms.">  <div class="sphx-glr-thumbnail-title">Multispectral demosaicing from raw sensor data</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to create a random phase retrieval operator and generate phaseless measurements from a given image. The example showcases 4 different reconstruction methods to recover the image from the phaseless measurements:">  <div class="sphx-glr-thumbnail-title">Random phase retrieval and reconstruction methods.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Plug-and-Play (PnP) is known to be challenging to apply to certain inverse problems like inpainting. One way to overcome this is to use a multi-scale approach instead of the standard single-scale approach.">  <div class="sphx-glr-thumbnail-title">Multi-scale Plug-and-Play for Inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Implementation of :footciteromano2017little using as plug-in denoiser the Gradient-Step Denoiser (GSPnP) :footcitehurault2021gradient which provides an explicit prior.">  <div class="sphx-glr-thumbnail-title">Regularization by Denoising (RED) for Super-Resolution.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard PnP algorithm with DnCNN denoiser for computed tomography.">  <div class="sphx-glr-thumbnail-title">Vanilla PnP for computed tomography (CT).</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This a toy example to show you how to use DEQ to solve a deblurring problem. Note that this is a small dataset for training. For optimal results, use a larger dataset.">  <div class="sphx-glr-thumbnail-title">Deep Equilibrium (DEQ) algorithms for image deblurring</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to implement the LISTA algorithm :footcitegregor2010learning, for a compressed sensing problem. In a nutshell, LISTA is an unfolded proximal gradient algorithm involving a soft-thresholding proximal operator with learnable thresholding parameters.">  <div class="sphx-glr-thumbnail-title">Learned Iterative Soft-Thresholding Algorithm (LISTA) for compressed sensing</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to implement a learned unrolled proximal gradient descent algorithm with a custom prior function. The custom prior in use is The algorithm is trained on a dataset of compressed sensing measurements of MNIST images.">  <div class="sphx-glr-thumbnail-title">Learned iterative custom prior</div>
</div>
<!-- thumbnail-parent-div-close --></div>
