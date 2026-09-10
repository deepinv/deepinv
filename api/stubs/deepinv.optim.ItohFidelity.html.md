# ItohFidelity

### *class* deepinv.optim.ItohFidelity(sigma=1.0, threshold=1.0)

Bases: [`L2`](https://deepinv.org/api/stubs/deepinv.optim.L2.html.md#deepinv.optim.L2)

Itoh data-fidelity term for spatial unwrapping problems.

This class implements a data-fidelity term based on the $\ell_2$ norm, but applied to the spatial finite differences of the variable and the wrapped differences of the data.
This is based on the Itoh condition for phase unwrapping <sup>[1](#footcite-itoh1982analysis)</sup>.
It is designed to be used in conjunction with the [`deepinv.physics.SpatialUnwrapping`](https://deepinv.org/api/stubs/deepinv.physics.SpatialUnwrapping.html.md#deepinv.physics.SpatialUnwrapping) class for spatial unwrapping tasks.

The data-fidelity term is defined as:

$$
f(x,y) = \frac{1}{2\sigma^2} \| D x - w_{t}(Dy) \|^2
$$

where $D$ denotes the spatial finite differences operator, $w_t$ denotes the wrapping operator, and $\sigma$ denotes the noise level.

* **Parameters:**
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Standard deviation of the noise to be used as a normalisation factor.
  * **threshold** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Threshold value $t$ used in the wrapping operator (default: 1.0).

<hr />

* **Example:**
  ```pycon
  >>> import torch
  >>> from deepinv.physics.spatial_unwrapping import SpatialUnwrapping
  >>> from deepinv.optim.data_fidelity import ItohFidelity
  >>> x = torch.ones(1, 1, 3, 3)
  >>> y = x
  >>> physics = SpatialUnwrapping(threshold=1.0, mode="round")
  >>> fidelity = ItohFidelity(sigma=1.0)
  >>> f = fidelity(x, y, physics)
  >>> print(f)
  tensor([0.])
  ```

<hr />

* **References:**

* <a id='footcite-itoh1982analysis'>**[1]**</a> Kazuyoshi Itoh. Analysis of the phase unwrapping algorithm. *Applied optics*, 21(14):2470–2470, 1982.

#### D(x, \*\*kwargs)

Apply spatial finite differences to the input tensor.

Computes the horizontal and vertical finite differences of the input tensor `x`
using first-order differences along the last two spatial dimensions. The result
is a tensor containing both the horizontal and vertical gradients stacked along
a new dimension.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input tensor of shape (…, H, W), where H and W are spatial dimensions.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) of shape (…, H, W, 2), where the last dimension contains
  the horizontal and vertical finite differences, respectively.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### D_adjoint(x, \*\*kwargs)

Applies the adjoint (transpose) of the spatial finite difference operator to the input tensor.

This function computes the adjoint operation corresponding to spatial finite differences,
typically used in image processing and variational optimization problems. The input `x`
is expected to have its last dimension of size 2, representing the horizontal and vertical
finite differences $(D_h x, D_v x)$.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input tensor of shape (…, 2), where the last dimension contains
  the horizontal and vertical finite differences.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) The result of applying the adjoint finite difference operator, with the
  same shape as the input except for the last dimension (which is removed).
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### WD(x, \*\*kwargs)

Applies spatial finite differences to the input and wraps the result.

This method computes the spatial finite differences of the input tensor $x$ using the $D$ operator,
then applies modular rounding to the result. This is typically used in
applications where periodic boundary conditions or phase wrapping are required.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input tensor to which the spatial finite differences and wrapping are applied.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) The wrapped finite differences of the input tensor.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### fn(x, y, physics, \*args, \*\*kwargs)

Computes the data fidelity term $\datafid{x}{y} = \distance{Dx}{w_{t}(Dy)}$.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the data fidelity is computed.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Data $y$.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – physics model.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) data fidelity $\datafid{x}{y}$.

#### grad(x, y, \*args, \*\*kwargs)

Calculates the gradient of the data fidelity term $\datafidname$ at $x$.

The gradient is computed using the chain rule:

$$
\nabla_x \distance{Dx}{w_{t}(Dy)} = \left. \frac{\partial D}{\partial x} \right|_x^\top \nabla_u \distance{u}{w_{t}(Dy)},
$$

where $\left. \frac{\partial D}{\partial x} \right|_x$ is the Jacobian of $D$ at $x$, and $\nabla_u \distance{u}{w_{t}(Dy)}$ is computed using `grad_d` with $u = Dx$. The multiplication is computed using the [`D_adjoint`](#deepinv.optim.ItohFidelity.D_adjoint) method of the class.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the gradient is computed.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Data $y$.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) gradient $\nabla_x \datafid{x}{y}$, computed in $x$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### prox(x, y, physics=None, \*args, gamma=1.0, \*\*kwargs)

Proximal operator of $\gamma \datafid{x}{y}$

Compute the proximal operator of the fidelity term $\operatorname{prox}_{\gamma \datafidname}$, i.e.

$$
\operatorname{prox}_{\gamma \datafidname}(x) = \underset{u}{\text{argmin}} \frac{\gamma}{2\sigma^2}\|Du-w_{t}(Dy)\|_2^2+\frac{1}{2}\|u-x\|_2^2
$$

using the DCT-based closed-form solution of Ramirez *et al.*<sup>[2](#footcite-ramirez2024phase)</sup> as follows

$$
\hat{x}_{i,j} = \texttt{DCT}^{-1}\left(
\frac{\texttt{DCT}(D^{\top}w_t(Dy) + \frac{\rho}{2} z)_{i,j}}
{ \frac{\rho}{2} + 4 - (2\cos(\pi i / M) + 2\cos(\pi j / N))}
\right)

$$

where $D$ is the finite difference operator and $\texttt{DCT}$ is the discrete cosine transform.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the proximity operator is computed.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Data $y$.
  * **gamma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – stepsize of the proximity operator.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) proximity operator $\operatorname{prox}_{\gamma \datafidname}(x)$.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<hr />

* **References:**

* <a id='footcite-ramirez2024phase'>**[2]**</a> Jhon Ramirez, Henry Arguello, and Jorge Bacca. Phase unwrapping for phase imaging using the plug-and-play proximal algorithm. *Appl. Opt.*, 63(2):535–542, Jan 2024. URL: [https://opg.optica.org/ao/abstract.cfm?URI=ao-63-2-535](https://opg.optica.org/ao/abstract.cfm?URI=ao-63-2-535), [doi:10.1364/AO.504036](https://doi.org/10.1364/AO.504036).

<a id="sphx-glr-backref-deepinv-optim-itohfidelity"></a>

## Examples using `ItohFidelity`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows the use of the deepinv.physics.SpatialUnwrapping forward model and the deepinv.optim.ItohFidelity for unwrapping problems, which occur in modulo imaging, interferometry SAR and other imaging applications. It shows how to generate a wrapped phase image, apply blur and noise, and reconstruct the original phase using both DCT inversion and ADMM optimization.">  <div class="sphx-glr-thumbnail-title">Spatial unwrapping and modulo imaging</div>
</div>
<!-- thumbnail-parent-div-close --></div>
