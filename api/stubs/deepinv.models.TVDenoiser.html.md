# TVDenoiser

### *class* deepinv.models.TVDenoiser(verbose=False, tau=0.01, rho=1.99, n_it_max=1000, crit=1e-5, x2=None, u2=None, ths=None)

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

Proximal operator of the isotropic Total Variation operator.

This algorithm converges to the unique image $x$ that is the solution of

$$
\underset{x}{\arg\min} \;  \frac{1}{2}\|x-y\|_2^2 + \gamma \|Dx\|_{1,2},
$$

where $D$ maps an image to its gradient field.

The problem is solved with an over-relaxed Chambolle-Pock algorithm, see Condat<sup>[1](#footcite-condat2013primal)</sup>.

Code (and description) adapted from Laurent Condat’s matlab version ([https://lcondat.github.io/software.html](https://lcondat.github.io/software.html)) and
Daniil Smolyakov’s [code](https://github.com/RoundedGlint585/TGVDenoising/blob/master/TGV%20WithoutHist.ipynb).

This algorithm is implemented with warm restart, i.e. the primary and dual variables are kept in memory
between calls to the forward method. This speeds up the computation when using this class in an iterative algorithm.

* **Parameters:**
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Whether to print computation details or not. Default: False.
  * **tau** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Stepsize for the primal update. Default: 0.01.
  * **rho** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Over-relaxation parameter. Default: 1.99.
  * **n_it_max** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Maximum number of iterations. Default: 1000.
  * **crit** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Convergence criterion. Default: 1e-5.
  * **x2** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* *None*) – Primary variable for warm restart. Default: None.
  * **u2** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* *None*) – Dual variable for warm restart. Default: None.
  * **ths** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Regularization parameter $\gamma$. Can also be passed to `forward`. Default: None

#### NOTE
The regularization term $\|Dx\|_{1,2}$ is implicitly normalized by its Lipschitz constant, i.e.
$\sqrt{8}$, see e.g. A. Beck and M. Teboulle, “Fast gradient-based algorithms for constrained total
variation image denoising and deblurring problems”, IEEE T. on Image Processing. 18(11), 2419-2434, 2009.

#### WARNING
For using TV as a prior for Plug and Play algorithms, it is recommended to use the class
[`TVPrior`](https://deepinv.org/api/stubs/deepinv.optim.TVPrior.html.md#deepinv.optim.TVPrior) instead. In particular, it allows to evaluate TV.

<hr />

* **References:**

* <a id='footcite-condat2013primal'>**[1]**</a> Laurent Condat. A primal–dual splitting method for convex optimization involving lipschitzian, proximable and linear composite terms. *Journal of optimization theory and applications*, 158(2):460–479, 2013.

#### forward(y, ths=None, \*\*kwargs)

Computes the proximity operator of the TV norm.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Noisy image. Assumes a tensor of shape (B, C, H, W) (2D data) or (B, C, D, H, W) (3D data).
  * **ths** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Regularization parameter $\gamma$. Takes priority over `ths` passed at initialization.
* **Returns:**
  Denoised image.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### *static* nabla(x)

Applies the finite differences operator associated with tensors of the same shape as x, in either 2D or 3D.

#### *static* nabla_adjoint(x)

Applies the adjoint of the finite difference operator.

#### prox_tau_fx(x, y)

Proximal operator of the function $\frac{1}{2}\|x-y\|_2^2$.

<a id="sphx-glr-backref-deepinv-models-tvdenoiser"></a>

## Examples using `TVDenoiser`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use variational 3D denoisers for denoising a 3D image. We first apply a standard soft-thresholding wavelet denoiser to a 3D brain MRI volume, as well as a 3D TV denoiser. We then extend the wavelet denoiser objective to a redundant dictionary of wavelet bases, which does not admit a closed-form solution. We solve the denoising problem using the Dykstra-like algorithm.">  <div class="sphx-glr-thumbnail-title">3D denoising</div>
</div>
<!-- thumbnail-parent-div-close --></div>
