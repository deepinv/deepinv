# BlindRL

### *class* deepinv.optim.BlindRL(x_prior=None, k_prior=None, lambda_reg_x=0.0, lambda_reg_k=0.0, g_param=None, g_param_kernel=None, x_steps=1, k_steps=1, kernel_size=(17, 17), normalize_kernel=True, use_fft=False, eps=1e-15, max_iter=100, crit_conv='residual', thres_conv=1e-5, early_stop=False, custom_metrics=None, init=None, cost_fn=None, params_algo=None, unfold=False, DEQ=None, anderson_acceleration=False, \*\*kwargs)

Bases: [`BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.md#deepinv.optim.BaseOptim)

Blind Richardson-Lucy deconvolution for Poisson inverse problems.

This algorithm alternates multiplicative MLEM updates for the image
$x$ and the blur kernel $h$ under the model

$$
y \sim \operatorname{Poisson}(h * x).

$$

The updates are given by:

> $$
> h^{(k+1)} = \Pi_{\Delta}\left[\frac{h^{(k)}}{(x^{(k)})^\dagger * \mathbf{1}} \odot (x^{(k)})^\dagger * \left(\frac{y}{x^{(k)} * h^{(k)}}\right)\right],
> $$

> and:

> $$
> x^{(k+1)} = \frac{x^{(k)}}{(h^{(k+1)})^\dagger * \mathbf{1}} \odot (h^{(k+1)})^\dagger * \left(\frac{y}{h^{(k+1)} * x^{(k)}}\right).
> $$

where $z^\dagger$ denotes the spatially flipped $z$, such that $z^\dagger *$ is the adjoint of convolution by $z$.
The kernel is constrained to be nonnegative and, by default, normalized to unit
sum by the Pi_{Delta} operation after each kernel update.

Image and kernel priors can be used.
The regularized algorithm is implemented using the the One-Step-Late (OSL) heuristic
of Green <sup>[1](#footcite-greenuseemalgorithm1990)</sup>.
The kernel and image updates then become:

$$
h^{(k+1)} = \Pi_{\Delta}\left[\frac{h^{(k)}}{(x^{(k)})^\dagger * \mathbf{1} + \lambda_h \nabla R_h(h^{(k)})} \odot (x^{(k)})^\dagger * \left(\frac{y}{x^{(k)} * h^{(k)}}\right)\right].
$$

$$
x^{(k+1)} = \frac{x^{(k)}}{(h^{(k+1)})^\dagger * \mathbf{1} + \lambda_x \nabla R_x(x^{(k)})} \odot (h^{(k+1)})^\dagger * \left(\frac{y}{h^{(k+1)} * x^{(k)}}\right),
$$

#### NOTE
The parameter `use_fft` enables to swap standard convolutions used to update
the image and the kernel for FFT based convolutions, which significantly speeds
up the algorithm when using a large image and estimating a large kernel.
The speedup is particularly important for kernels bigger than 30x30.
For small kernels (<15x15) and images (<128x128), it is more efficient to use
standard convolutions.

* **Parameters:**
  * **x_prior** ([*deepinv.optim.Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.md#deepinv.optim.Prior)) – optional image prior. Default: `None`.
  * **k_prior** ([*deepinv.optim.Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.md#deepinv.optim.Prior)) – optional kernel prior. Default: `None`.
  * **lambda_reg_x** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – image regularization parameter. Default: `0.0`.
  * **lambda_reg_k** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – kernel regularization parameter. Default: `0.0`.
  * **g_param** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – parameter for the image prior. Default: `None`.
  * **g_param_kernel** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – parameter for the kernel prior. Default: `None`.
  * **x_steps** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of inner image updates per iteration. Default: `1`.
  * **k_steps** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of inner kernel updates per iteration. Default: `1`.
  * **kernel_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – spatial size of the default uniform
    kernel. An explicit kernel in `init` overrides this size. Default: `(17, 17)`.
  * **normalize_kernel** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to normalize the kernel to unit sum.
    Default: `True`.
  * **use_fft** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to use FFT implementations for image and kernel
    convolutions. Default: `False`.
  * **eps** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – numerical stability constant. Default: `1e-15`.
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of alternating BlindRL iterations. Default: `100`.
  * **init** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – initial image and blur kernel
    `(x0, k0)`. If `None`, `x0` is initialized with `y` and `k0`
    with a uniform kernel of the same size as `kernel_size`.
  * **cost_fn** (*Callable*) – optional cost function. If omitted, the Poisson
    negative log-likelihood plus explicit image and kernel priors is used.
  * **params_algo** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – optionally provide BlindRL parameters directly.

<hr />

* **References:**

* <a id='footcite-greenuseemalgorithm1990'>**[1]**</a> Peter J. Green. On Use of the Em Algorithm for Penalized Likelihood Estimation. *Journal of the Royal Statistical Society: Series B (Methodological)*, 52(3):443–452, 1990. [doi:10.1111/j.2517-6161.1990.tb01798.x](https://doi.org/10.1111/j.2517-6161.1990.tb01798.x).

#### forward(y, x_gt=None, compute_metrics=False, \*\*kwargs)

Run Blind Richardson-Lucy deconvolution.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – blurred image of shape `(B, C, H, W)`.
  * **x_gt** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – optional ground-truth image used to compute metrics. Default: `None`.
  * **compute_metrics** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to compute reconstruction metrics. Default: `False`.
* **Returns:**
  estimated image and blur kernel `(x, k)`. If `compute_metrics` is `True`, return `((x, k), metrics)`.

<a id="sphx-glr-backref-deepinv-optim-blindrl"></a>

## Examples using `BlindRL`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example introduces the Richardson-Lucy algorithm for deconvolution in the blind setting, where both the underlying clean image and the blur kernel are unknown.">![](auto_examples/blind-inverse-problems/images/thumb/sphx_glr_demo_blind_richardsonlucy_thumb.png)

[Blind Richardson-Lucy deconvolution](https://deepinv.org/auto_examples/blind-inverse-problems/demo_blind_richardsonlucy.md)

  <div class="sphx-glr-thumbnail-title">Blind Richardson-Lucy deconvolution</div>
</div>
<!-- thumbnail-parent-div-close --></div>
