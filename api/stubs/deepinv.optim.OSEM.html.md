# OSEM

### *class* deepinv.optim.OSEM(data_fidelity=None, prior=None, lambda_reg=1.0, g_param=None, sigma_denoiser=None, num_subsets=2, eps=1e-6, max_iter=100, crit_conv='residual', thres_conv=1e-5, early_stop=False, custom_metrics=None, custom_init=None, unfold=False, trainable_params=None, cost_fn=None, params_algo=None, \*\*kwargs)

Bases: [`BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim)

Ordered-Subsets Expectation-Maximization (OSEM) algorithm for Poisson inverse problems.

OSEM was proposed in <sup>[1](#footcite-hudsonacceleratedimagereconstruction1994)</sup>
to accelerate MLEM <sup>[2](#footcite-sheppmaximumlikelihoodreconstruction1982)</sup> by
splitting the measurement into ordered subsets.
Note that MLEM is a special case of OSEM with only one subset.
At each iteration, the algorithm performs multiplicative updates over all subsets
of the form:

$$
x_{k,l+1} = \frac{x_{k,l}}{A_l^T \mathbf{1}} \odot A_l^T \left(\frac{y_l}{A_l x_{k,l} + b_l}\right),

$$

where $A_l$ and $y_l$ are the corresponding physics and measurement
subset, and $b_l$ is an optional additive background and `l` is the subset index.

#### NOTE
Only [`deepinv.physics.Tomography`](https://deepinv.org/api/stubs/deepinv.physics.Tomography.html.md#deepinv.physics.Tomography), [`deepinv.physics.TomographyWithAstra`](https://deepinv.org/api/stubs/deepinv.physics.TomographyWithAstra.html.md#deepinv.physics.TomographyWithAstra)
and [`deepinv.physics.PET`](https://deepinv.org/api/stubs/deepinv.physics.PET.html.md#deepinv.physics.PET) are currently supported for OSEM.

#### NOTE
The user can provide either the full measurement tensor `y` and full tomography
`physics`, or pre-split measurements passed as a [`deepinv.utils.TensorList`](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList)
and pre-split physics passed as a [`deepinv.physics.StackedLinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.StackedLinearPhysics.html.md#deepinv.physics.StackedLinearPhysics).
See [`deepinv.physics.split_physics()`](https://deepinv.org/api/stubs/deepinv.physics.split_physics.html.md#deepinv.physics.split_physics) and [`deepinv.physics.split_measurements()`](https://deepinv.org/api/stubs/deepinv.physics.split_measurements.html.md#deepinv.physics.split_measurements).

See [`deepinv.optim.optim_iterators.OSEMIteration`](https://deepinv.org/api/stubs/deepinv.optim.optim_iterators.OSEMIteration.html.md#deepinv.optim.optim_iterators.OSEMIteration) for the details of one
iteration.

A regularization can be included by specifying a `prior`.
This uses One-Step-Late (OS-MAP-OSL) <sup>[3](#footcite-greenuseemalgorithm1990)</sup>,
similar to in [`deepinv.optim.MLEM`](https://deepinv.org/api/stubs/deepinv.optim.MLEM.html.md#deepinv.optim.MLEM).

#### NOTE
By default, the algorithm is initialized with a tensor of ones with the same
shape as $A^T y$. This can be overridden using `custom_init`.

* **Parameters:**
  * **num_subsets** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of ordered subsets used when splitting a full
    physics. It must be positive and is ignored when a pre-split physics is
    provided. With one subset, OSEM is equivalent to [`deepinv.optim.MLEM`](https://deepinv.org/api/stubs/deepinv.optim.MLEM.html.md#deepinv.optim.MLEM).
    Default: `2`.
  * **data_fidelity** ([*deepinv.optim.DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity) *]*) – data fidelity term.
    If `None`, defaults to [`deepinv.optim.PoissonLikelihood`](https://deepinv.org/api/stubs/deepinv.optim.PoissonLikelihood.html.md#deepinv.optim.PoissonLikelihood).
  * **prior** ([*deepinv.optim.Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior) *]*) – prior term. If `None`, no prior is used.
    Default: `None`.
  * **lambda_reg** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – regularization parameter $\lambda$. Default: `1.0`.
  * **g_param** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – parameter for the prior. Default: `None`.
  * **sigma_denoiser** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – same as `g_param`. If both `g_param` and `sigma_denoiser` are provided, `g_param` is used. Default: `None`.
  * **eps** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – positive value used to clamp denominators in the
    multiplicative update. Default: `1e-6`.
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of OSEM epochs. Default: `100`.
  * **crit_conv** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – convergence criterion, either `"residual"` or `"cost"`.
    Default: `"residual"`.
  * **thres_conv** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – convergence threshold. Default: `1e-5`.
  * **early_stop** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, the algorithm stops when the convergence criterion is met.
    Default: `False`.
  * **custom_metrics** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – dictionary of custom metrics to compute at each epoch.
    Default: `None`.
  * **custom_init** (*Callable*) – custom initialization function. OSEM passes the
    split measurements and stacked subset physics to this function. Default: `None`.
  * **unfold** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to unfold the algorithm or not. Default: `False`.
  * **trainable_params** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list)) – parameters to train if `unfold` is `True`. Default: `None`.
  * **cost_fn** (*Callable*) – custom cost function used for metrics and convergence.
    OSEM calls it with a [`deepinv.optim.StackedPhysicsDataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.StackedPhysicsDataFidelity.html.md#deepinv.optim.StackedPhysicsDataFidelity),
    split measurements, and stacked subset physics. Default: `None`.
  * **params_algo** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – optionally provide the algorithm parameters directly.

<hr />

* **References:**

* <a id='footcite-hudsonacceleratedimagereconstruction1994'>**[1]**</a> H.M. Hudson and R.S. Larkin. Accelerated image reconstruction using ordered subsets of projection data. *IEEE Transactions on Medical Imaging*, 13(4):601–609, December 1994. [doi:10.1109/42.363108](https://doi.org/10.1109/42.363108).
* <a id='footcite-sheppmaximumlikelihoodreconstruction1982'>**[2]**</a> Lawrence A Shepp and Yehuda Vardi. Maximum likelihood reconstruction for emission tomography. *IEEE Transactions on Medical Imaging*, 1(2):113–122, 1982.
* <a id='footcite-greenuseemalgorithm1990'>**[3]**</a> Peter J. Green. On Use of the Em Algorithm for Penalized Likelihood Estimation. *Journal of the Royal Statistical Society: Series B (Methodological)*, 52(3):443–452, 1990. [doi:10.1111/j.2517-6161.1990.tb01798.x](https://doi.org/10.1111/j.2517-6161.1990.tb01798.x).

#### forward(y, physics, \*args, \*\*kwargs)

Run OSEM with full or pre-split measurements and physics.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*deepinv.utils.TensorList*](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – Full measurement tensor, or pre-split measurements when `physics` is a [`deepinv.physics.StackedLinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.StackedLinearPhysics.html.md#deepinv.physics.StackedLinearPhysics).
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Full tomography/PET physics or pre-split [`deepinv.physics.StackedLinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.StackedLinearPhysics.html.md#deepinv.physics.StackedLinearPhysics).
* **Returns:**
  Reconstructed image, and optionally the metrics dictionary when `compute_metrics=True`.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)]

<a id="sphx-glr-backref-deepinv-optim-osem"></a>

## Examples using `OSEM`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct an image from them.">  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 2D</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct a volume from them.">  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 3D</div>
</div>
<!-- thumbnail-parent-div-close --></div>
