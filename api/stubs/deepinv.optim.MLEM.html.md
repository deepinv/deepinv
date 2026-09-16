# MLEM

### *class* deepinv.optim.MLEM(data_fidelity=None, prior=None, lambda_reg=1.0, g_param=None, sigma_denoiser=None, eps=1e-6, max_iter=100, crit_conv='residual', thres_conv=1e-5, early_stop=False, custom_metrics=None, custom_init=None, unfold=False, trainable_params=None, cost_fn=None, params_algo=None, \*\*kwargs)

Bases: [`BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim)

Maximum Likelihood Expectation Maximization (MLEM) algorithm for Poisson inverse problems.

This algorithm was originally proposed for deconvolution by Richardson and Lucy <sup>[1](#footcite-richardsonbayesianbasediterativemethod1972)</sup><sup>[2](#footcite-lucyiterativetechniquerectification1974)</sup> and was later
adapted to tomographic reconstruction by Shepp and Vardi <sup>[3](#footcite-sheppmaximumlikelihoodreconstruction1982)</sup>.
It is also widely used in Non-Negative Matrix Factorization (NMF) problems where it is known as the Lee and Seung multiplicative update algorithm <sup>[4](#footcite-leeseungalgorithmsnonnegativematrix2000)</sup>.

The algorithm is traditionally derived from the Expectation-Maximization (EM) framework with specific latent variables.
Alternatively, it can be seen as a Majorization-Minimization (MM) algorithm where each iteration consists in constructing a surrogate function that majorizes the Poisson negative log-likelihood and then minimizing this surrogate function.
At each iteration, the algorithm performs a multiplicative update of the form:

$$
x_{k+1} = \frac{x_k}{A^T \mathbf{1}} \odot A^T \left(\frac{y}{A x_k + b}\right)

$$

where $A$ is the forward operator, $y$ is the observed data,
$b$ is an optional additive background (useful in PET), $\mathbf{1}$ is a tensor of ones,
and $\odot$ denotes element-wise multiplication.

The algorithm can be used with a prior term (e.g., for MAP-EM variants) or without
(standard MLEM). See [`deepinv.optim.optim_iterators.MLEMIteration`](https://deepinv.org/api/stubs/deepinv.optim.optim_iterators.MLEMIteration.html.md#deepinv.optim.optim_iterators.MLEMIteration) for the details of the iteration.

The MLEM algorithm minimizes the Poisson negative log-likelihood data-fidelity. The `data_fidelity` argument
can be used to measure progress during optimization (e.g., for early stopping or metrics computation), but it is
not used as the objective function to minimize. Only the Poisson negative log-likelihood is minimized regardless
of the provided `data_fidelity` argument.

A regularization can be included via the `prior` argument, which will lead to a MAP-EM variant of the MLEM algorithm.
Our implementation is based on the One-Step-Late (OSL) heuristic of Green <sup>[5](#footcite-greenuseemalgorithm1990)</sup>.
It leads to the following update rule:

$$
x_{k+1} = \frac{x_k}{A^T \mathbf{1} + \lambda \nabla g(x_k)} \odot A^T \left(\frac{y}{A x_k + b}\right)

$$

where $g$ is the prior function and $\lambda$ is the regularization parameter.

In the case of a non-differentiable prior, the gradient term $\nabla g(x_k)$ is replaced by a subgradient:

$$
x_{k+1} = \frac{x_k}{A^T \mathbf{1} + \lambda \partial g(x_k)} \odot A^T \left(\frac{y}{A x_k + b}\right)

$$

where $\partial g(x_k)$ is a subgradient of $g$ at point $x_k$.

#### NOTE
By default, the algorithm is initialized with a tensor of ones with the same
shape as $A^T y$. This can be overridden using `custom_init`.

* **Parameters:**
  * **data_fidelity** ([*deepinv.optim.DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity) *]*) – data fidelity term.
    If `None`, defaults to [`deepinv.optim.PoissonLikelihood`](https://deepinv.org/api/stubs/deepinv.optim.PoissonLikelihood.html.md#deepinv.optim.PoissonLikelihood).
  * **prior** ([*deepinv.optim.Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior) *]*) – prior term. If `None`, no prior is used.
    Default: `None`.
  * **lambda_reg** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – regularization parameter $\lambda$. Default: `1.0`.
  * **g_param** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – parameter for the prior. Default: `None`.
  * **sigma_denoiser** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – same as `g_param`. If both `g_param` and `sigma_denoiser` are provided, `g_param` is used. Default: `None`.
  * **eps** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – positive value used to clamp denominators in the
    multiplicative update. Default: `1e-6`.
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of iterations. Default: `100`.
  * **crit_conv** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – convergence criterion, either `"residual"` or `"cost"`.
    Default: `"residual"`.
  * **thres_conv** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – convergence threshold. Default: `1e-5`.
  * **early_stop** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, the algorithm stops when the convergence criterion is met.
    Default: `False`.
  * **custom_metrics** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – dictionary of custom metrics to compute at each iteration.
    Default: `None`.
  * **custom_init** (*Callable*) – custom initialization function. Default: `None`.
  * **unfold** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to unfold the algorithm or not. Default: `False`.
  * **trainable_params** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list)) – list of ADMM parameters to be trained if `unfold` is True. To choose between `["lambda", "stepsize", "g_param", "beta"]`. Default: None, which means that all parameters are trainable if `unfold` is True. For no trainable parameters, set to an empty list.
  * **cost_fn** (*Callable*) – Custom user input cost function.
    `cost_fn(x, data_fidelity, prior, cur_params, y, physics)` takes as input
    the current primal variable ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)), the current data-fidelity ([`deepinv.optim.DataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)),
    the current prior ([`deepinv.optim.Prior`](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)), the current parameters (dict), and the measurement ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)).
    Default: `None`.
  * **params_algo** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – optionally, directly provide the ADMM parameters in a dictionary. This will overwrite the parameters in the arguments `stepsize`, `lambda_reg`, `g_param` and `beta`.

<hr />

* **References:**

* <a id='footcite-richardsonbayesianbasediterativemethod1972'>**[1]**</a> William Hadley Richardson. Bayesian-based iterative method of image restoration. *Journal of the Optical Society of America*, 62(1):55, 1972. [doi:10.1364/JOSA.62.000055](https://doi.org/10.1364/JOSA.62.000055).
* <a id='footcite-lucyiterativetechniquerectification1974'>**[2]**</a> L. B. Lucy. An iterative technique for the rectification of observed distributions. *The Astronomical Journal*, 79:745, 1974. [doi:10.1086/111605](https://doi.org/10.1086/111605).
* <a id='footcite-sheppmaximumlikelihoodreconstruction1982'>**[3]**</a> Lawrence A Shepp and Yehuda Vardi. Maximum likelihood reconstruction for emission tomography. *IEEE Transactions on Medical Imaging*, 1(2):113–122, 1982.
* <a id='footcite-leeseungalgorithmsnonnegativematrix2000'>**[4]**</a> Daniel Lee and H. Sebastian Seung. Algorithms for non-negative matrix factorization. In T. Leen, T. Dietterich, and V. Tresp, editors, *Advances in neural information processing systems*, volume 13. MIT Press, 2000.
* <a id='footcite-greenuseemalgorithm1990'>**[5]**</a> Peter J. Green. On Use of the Em Algorithm for Penalized Likelihood Estimation. *Journal of the Royal Statistical Society: Series B (Methodological)*, 52(3):443–452, 1990. [doi:10.1111/j.2517-6161.1990.tb01798.x](https://doi.org/10.1111/j.2517-6161.1990.tb01798.x).

#### forward(y, physics, \*args, \*\*kwargs)

Run MLEM with a sensitivity map computed once for this reconstruction.

<a id="sphx-glr-backref-deepinv-optim-mlem"></a>

## Examples using `MLEM`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to solve Poisson inverse problems using the Maximum-Likelihood Expectation-Maximization (MLEM) algorithm :footcitesheppMaximumLikelihoodReconstruction1982, also known as the Richardson-Lucy algorithm in the deconvolution setting :footciterichardsonBayesianBasedIterativeMethod1972,lucyIterativeTechniqueRectification1974.">  <div class="sphx-glr-thumbnail-title">Poisson Inverse Problems with Maximum-Likelihood Expectation-Maximization (MLEM)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct an image from them.">  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 2D</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct a volume from them.">  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 3D</div>
</div>
<!-- thumbnail-parent-div-close --></div>
