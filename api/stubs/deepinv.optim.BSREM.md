# BSREM

### *class* deepinv.optim.BSREM(data_fidelity=None, prior=None, lambda_reg=1.0, g_param=None, sigma_denoiser=None, num_subsets=2, stepsize=1.0, eps=1e-6, sensitivity_threshold=1e-2, max_iter=100, crit_conv='residual', thres_conv=1e-5, early_stop=False, custom_metrics=None, custom_init=None, unfold=False, trainable_params=None, cost_fn=None, params_algo=None, \*\*kwargs)

Bases: [`BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.md#deepinv.optim.BaseOptim)

Block Sequential Regularized Expectation Maximization (BSREM) for Poisson inverse problems.

BSREM is a relaxed ordered-subsets algorithm for minimizing the penalized Poisson negative log-likelihood

$$
\min_{x \in \mathbb{R}^n_{+}}\; \mathrm{KL}(y, Ax) + \lambda \reg{x}.
$$

while preserving convergence guarantees <sup>[1](#footcite-depierrofastemlikemethods2001)</sup><sup>[2](#footcite-ahngloballyconvergentimage2003)</sup>.
With $L$ subsets, one complete iteration of the algorithm applies the following update for $l=1,\ldots,L$:

$$
x_{k,l+1} = \mathcal{P}_{+}\left[x_{k,l} - \alpha_k
\frac{x_{k,l}}{\bar{s}} \odot \left(\nabla f_l(x_{k,l})
+ \frac{\lambda}{L}\nabla \reg{x_{k,l}}\right)\right],
$$

where $\bar{s}=A^T\mathbf{1}/L$ is the average subset sensitivity,
$\alpha_k$ is the relaxation step size, which is annealed over the
iterations, and $\mathcal{P}_{+}$ clamps the iterate to the positive orthant.

See [`deepinv.optim.optim_iterators.BSREMIteration`](https://deepinv.org/api/stubs/deepinv.optim.optim_iterators.BSREMIteration.md#deepinv.optim.optim_iterators.BSREMIteration) for the details of one iteration.

#### TIP
The description of the algorithm above assumes unit Poisson gain.
If a non-unit gain is used, the implementation automatically scales the
preconditioner and prior to adapt to the gain.

A custom annealing schedule for the relaxation step size can be supplied as
an iterable, for example
`stepsize=[1 / (1 + 0.1 * k) for k in range(max_iter)]`.

#### NOTE
The user can provide either the full measurement tensor `y` and full
tomography `physics`, or pre-split measurements passed as a
[`deepinv.utils.TensorList`](https://deepinv.org/api/stubs/deepinv.utils.TensorList.md#deepinv.utils.TensorList) and pre-split physics passed as a
[`deepinv.physics.StackedLinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.StackedLinearPhysics.md#deepinv.physics.StackedLinearPhysics). See
[`deepinv.physics.split_physics()`](https://deepinv.org/api/stubs/deepinv.physics.split_physics.md#deepinv.physics.split_physics) and
[`deepinv.physics.split_measurements()`](https://deepinv.org/api/stubs/deepinv.physics.split_measurements.md#deepinv.physics.split_measurements).

#### NOTE
By default, the algorithm is initialized with a tensor of ones with the
same shape as $A^T y$. This can be overridden using
`custom_init`.

* **Parameters:**
  * **num_subsets** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of ordered subsets used for the splitting of the physics and measurements. Ignored when pre-split inputs are provided. Default: `2`.
  * **data_fidelity** ([*deepinv.optim.DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.md#deepinv.optim.DataFidelity) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.md#deepinv.optim.DataFidelity) *]*) – data fidelity used by the subset updates and to evaluate the objective. If `None`, defaults to [`deepinv.optim.PoissonLikelihood`](https://deepinv.org/api/stubs/deepinv.optim.PoissonLikelihood.md#deepinv.optim.PoissonLikelihood).
  * **prior** ([*deepinv.optim.Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.md#deepinv.optim.Prior) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.md#deepinv.optim.Prior) *]*) – differentiable prior term. If `None`, no regularization is applied. Default: `None`.
  * **lambda_reg** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – regularization parameter $\lambda$. Default: `1.0`.
  * **g_param** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – parameter passed to the prior. Default: `None`.
  * **sigma_denoiser** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – alias for `g_param`. If both are provided, `g_param` takes precedence. Default: `None`.
  * **stepsize** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*collections.abc.Iterable*](https://docs.python.org/3.9/library/collections.abc.html#collections.abc.Iterable) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *]*) – annealing schedule of the relaxation step size. If an iterable is used, it must contain at least `max_iter` entries. Default: `1.0`.
  * **eps** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – positive value used for safe divisions and the positivity projection. Default: `1e-6`.
  * **sensitivity_threshold** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – relative sensitivity threshold defining the reconstruction support. Default: `1e-2`.
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of BSREM epochs. Default: `100`.
  * **crit_conv** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – convergence criterion, either `"residual"` or `"cost"`. Default: `"residual"`.
  * **thres_conv** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – convergence threshold for `crit_conv`. Default: `1e-5`.
  * **early_stop** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – stop when the convergence criterion is met. Default: `False`.
  * **custom_metrics** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – custom metrics computed after every epoch. Default: `None`.
  * **custom_init** (*Callable*) – custom initialization function. BSREM passes the split measurements and stacked subset physics to this function. If `None`, the reconstruction is initialized with ones. Default: `None`.
  * **unfold** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to unfold the algorithm. Default: `False`.
  * **trainable_params** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list)) – algorithm parameters to train when unfolded, chosen from `["lambda", "stepsize", "g_param"]`. If `None`, all parameters are trainable. Default: `None`.
  * **cost_fn** (*Callable*) – custom cost function used for metrics and convergence. BSREM calls it with a [`deepinv.optim.StackedPhysicsDataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.StackedPhysicsDataFidelity.md#deepinv.optim.StackedPhysicsDataFidelity), split measurements, and stacked subset physics. Default: `None`.
  * **params_algo** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – optional algorithm parameters. When provided, this overrides `stepsize`, `lambda_reg`, and `g_param`.

<hr />

* **References:**

* <a id='footcite-depierrofastemlikemethods2001'>**[1]**</a> A.R. De Pierro and M.E.B. Yamagishi. Fast EM-like methods for maximum “a posteriori” estimates in emission tomography. *IEEE Transactions on Medical Imaging*, 20(4):280–288, April 2001. [doi:10.1109/42.921477](https://doi.org/10.1109/42.921477).
* <a id='footcite-ahngloballyconvergentimage2003'>**[2]**</a> Sangtae Ahn and Jeffrey A. Fessler. Globally convergent image reconstruction for emission tomography using relaxed ordered subsets algorithms. *IEEE Transactions on Medical Imaging*, 22(5):613–626, May 2003. [doi:10.1109/TMI.2003.812251](https://doi.org/10.1109/TMI.2003.812251).

#### forward(y, physics, \*args, \*\*kwargs)

Run BSREM with full or pre-split measurements and physics.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*deepinv.utils.TensorList*](https://deepinv.org/api/stubs/deepinv.utils.TensorList.md#deepinv.utils.TensorList) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – Full measurement tensor, or pre-split measurements when `physics` is a [`deepinv.physics.StackedLinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.StackedLinearPhysics.md#deepinv.physics.StackedLinearPhysics).
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.md#deepinv.physics.Physics)) – Full tomography/PET physics or pre-split [`deepinv.physics.StackedLinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.StackedLinearPhysics.md#deepinv.physics.StackedLinearPhysics).
* **Returns:**
  Reconstructed image, and optionally the metrics dictionary when `compute_metrics=True`.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)]

<a id="sphx-glr-backref-deepinv-optim-bsrem"></a>

## Examples using `BSREM`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a 2D slice from the BrainWeb \`&lt;https://github.com/casperdcl/brainweb&gt;\`_ positron emission tomography (PET) dataset. The slice contains five hot lesions, we simulate a sinogram with deepinv.physics.PET and we compare standard PET reconstruction algorithms with methods that support penalized objective functions.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet_brainweb_2d_thumb.png)

[2D PET reconstruction with the Brainweb dataset](https://deepinv.org/auto_examples/physics/demo_pet_brainweb_2d.md)

  <div class="sphx-glr-thumbnail-title">2D PET reconstruction with the Brainweb dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a volume from the BrainWeb \`&lt;https://github.com/casperdcl/brainweb&gt;\`_ positron emission tomography (PET) dataset. We compare standard PET reconstruction algorithms with methods that support penalized objective functions.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet_brainweb_3d_thumb.png)

[3D PET reconstruction with the Brainweb dataset](https://deepinv.org/auto_examples/physics/demo_pet_brainweb_3d.md)

  <div class="sphx-glr-thumbnail-title">3D PET reconstruction with the Brainweb dataset</div>
</div>
<!-- thumbnail-parent-div-close --></div>
