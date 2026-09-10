# to_multiscale

### *class* deepinv.physics.to_multiscale(physics, img_size, factors=(2, 4, 8), device='cpu', dtype=None)

Bases:

Convert a single-scale physics operator to a multi-scale physics operator

A single-scale physics operator $\forw{\cdot}$ can be converted to a
multi-scale physics operator, i.e., one operating at multiple coarser
scales in addition to the original fine scale.

A downsampled physics operator $A_j$ operating at scale index
$j \geq 0$ can be evaluated by first upsampling the coarse scale
input $x_j$ to the original fine scale and then applying the original
physics operator

$$
A_j\left(x\right) = A\left(U_j\left(x_j\right)\right)
$$

where $U_j$ denotes upsampling by a factor $\alpha_j$
associated to the scale index $j$. For more details, see <sup>[1](#footcite-laurent2025multilevel)</sup>.

#### NOTE
The scale index $j$ is often associated to the scale factor $\alpha_j = 2^j$.

For linear physics operators, the adjoint operator can be computed from the
adjoint of the original physics operator and the adjoint of the upsampling
operator, which is the corresponding downsampling operator.

In certain cases, it is possible to evaluate the downsampled physics
operator without first upsampling the input image at the original scale.
For instance, for blur and inpainting operators, it is possible to compute
the downsampled blur kernels and inpainting masks acting directly on the
coarse input.

#### NOTE
In addition to the generic implementation `MultiScalerPhysics` which
is compatible with arbitrary physics operators, specific
implementations cover linear physics operators
`LinearPhysicsMultiScaler` and specific physics operators such as
`BlurMultiScaler`, `BlurFFTMultiScaler`, and
`InpaintingMultiScaler`. Users are encouraged to write custom
implementations for physics operators that do not have an efficient
implementation yet.

#### NOTE
The function `to_multiscale` is currently only compatible with 2D input images.

* **Parameters:**
  * **physics** ([*Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – single-scale physics that should be made multi-scale
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,*  *...* *]*) – image size in the fine scale
  * **factors** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,*  *...* *]*) – coarse scales associated to coarse scale indices, default: (2, 4, 8)
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – device of the inputs, default: ‘cpu’
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) *,* *None*) – dtype of the inputs, default: None
* **Returns:**
  (PhysicsMultiScaler) the multi-scale physics operator

<hr />

* **References:**

* <a id='footcite-laurent2025multilevel'>**[1]**</a> Nils Laurent, Julián Tachella, Elisa Riccietti, and Nelly Pustelnik. Multilevel plug-and-play image restoration. *IEEE Transactions on Computational Imaging*, 2025.

<a id="sphx-glr-backref-deepinv-physics-to-multiscale"></a>

## Examples using `to_multiscale`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Plug-and-Play (PnP) is known to be challenging to apply to certain inverse problems like inpainting. One way to overcome this is to use a multi-scale approach instead of the standard single-scale approach.">  <div class="sphx-glr-thumbnail-title">Multi-scale Plug-and-Play for Inpainting</div>
</div>
<!-- thumbnail-parent-div-close --></div>
