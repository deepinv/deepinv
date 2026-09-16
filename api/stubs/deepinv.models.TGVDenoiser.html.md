# TGVDenoiser

### *class* deepinv.models.TGVDenoiser(verbose=False, n_it_max=1000, crit=1e-5, x2=None, u2=None, r2=None, ths=None)

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

Proximal operator of (2nd order) Total Generalized Variation operator.

Adapted from Bredies *et al.*<sup>[1](#footcite-bredies2010total)</sup>.

This algorithm converges to the unique image $x$ (and the auxiliary vector field $r$) minimizing

$$
\underset{x, r}{\arg\min} \;  \frac{1}{2}\|x-y\|_2^2 + \lambda_1 \|r\|_{1,2} + \lambda_2 \|J(Dx-r)\|_{1,F}
$$

where $D$ maps an image to its gradient field and $J$ maps a vector field to its Jacobian.
For a large value of $\lambda_2$, the TGV behaves like the TV.
For a small value, it behaves like the $\ell_1$-Frobenius norm of the Hessian.

The problem is solved with an over-relaxed Chambolle-Pock algorithm, see Condat<sup>[2](#footcite-condat2013primal)</sup>.

Code (and description) adapted from Laurent Condat’s matlab version ([https://lcondat.github.io/software.html](https://lcondat.github.io/software.html)) and
Daniil Smolyakov’s [code](https://github.com/RoundedGlint585/TGVDenoising/blob/master/TGV%20WithoutHist.ipynb).

#### NOTE
The regularization term $\|r\|_{1,2} + \|J(Dx-r)\|_{1,F}$ is implicitly normalized by its Lipschitz
constant, i.e. $\sqrt{72}$, see e.g. K. Bredies et al., “Total generalized variation,” SIAM J. Imaging
Sci., 3(3), 492-526, 2010.

* **Parameters:**
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Whether to print computation details or not. Default: False.
  * **n_it_max** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Maximum number of iterations. Default: 1000.
  * **crit** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Convergence criterion. Default: 1e-5.
  * **x2** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* *None*) – Primary variable. Default: None.
  * **u2** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* *None*) – Dual variable. Default: None.
  * **r2** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* *None*) – Auxiliary variable. Default: None.
  * **ths** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Regularization parameter. Can also be passed to `forward`. Default: None

<hr />

* **References:**

* <a id='footcite-bredies2010total'>**[1]**</a> Kristian Bredies, Karl Kunisch, and Thomas Pock. Total generalized variation. *SIAM Journal on Imaging Sciences*, 3(3):492–526, 2010.
* <a id='footcite-condat2013primal'>**[2]**</a> Laurent Condat. A primal–dual splitting method for convex optimization involving lipschitzian, proximable and linear composite terms. *Journal of optimization theory and applications*, 158(2):460–479, 2013.

#### *static* epsilon(I)

Applies the jacobian of a vector field.

#### *static* epsilon_adjoint(G)

Applies the adjoint of the jacobian of a vector field.

#### forward(y, ths=None, \*\*kwargs)

Computes the proximity operator of the TGV norm.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Noisy image. Assumes a tensor of shape (B, C, H, W) (2D data) or (B, C, D, H, W) (3D data).
  * **ths** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Regularization parameter. Takes priority over `ths` passed at initialization.
* **Returns:**
  Denoised image.

#### *static* nabla(x)

Applies the finite differences operator associated with tensors of the same shape as x.

#### *static* nabla_adjoint(x)

Applies the adjoint of the finite difference operator.

<a id="sphx-glr-backref-deepinv-models-tgvdenoiser"></a>

## Examples using `TGVDenoiser`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of the denoisers in DeepInverse. A denoiser is a model that takes in a noisy image and outputs a denoised version of it. Basically, it solves the following problem:">  <div class="sphx-glr-thumbnail-title">Benchmarking pretrained denoisers</div>
</div>
<!-- thumbnail-parent-div-close --></div>
