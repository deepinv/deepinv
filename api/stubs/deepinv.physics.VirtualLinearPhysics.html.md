# VirtualLinearPhysics

### *class* deepinv.physics.VirtualLinearPhysics(, physics, transform, g_params)

Bases: [`LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)

Virtual linear operator

A virtual operator is an operator of the form

$$
A = \tilde{A} T_g
$$

where $\tilde{A}$ is a linear operator and $T_g$ is an invertible transformation with parameters $g$.

Unlike general composition of linear operators, like for [`deepinv.physics.ComposedLinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.ComposedLinearPhysics.html.md#deepinv.physics.ComposedLinearPhysics), the invertibility of $T_g$ allows to compute the pseudo-inverse of $A$ in a computationally efficient closed form, i.e.,

$$
A^\dagger = T_g^{-1} \tilde{A}^\dagger.
$$

Virtual operators are used in [`deepinv.models.EquivariantReconstructor`](https://deepinv.org/api/stubs/deepinv.models.EquivariantReconstructor.html.md#deepinv.models.EquivariantReconstructor). For more details, see Sechaud *et al.*<sup>[1](#footcite-sechaud26equivariant)</sup>.

#### WARNING
The adjoint and pseudo-inverse might be incorrect if the transformation is not invertible, for instance due to boundary effects.

* **Parameters:**
  * **physics** ([*LinearPhysics*](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)) – linear physics operator $\tilde{A}$.
  * **transform** ([*Transform*](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform)) – transformation $T_g$
  * **g_params** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – parameters of the transformation $g$.

<hr />

* **References:**

* <a id='footcite-sechaud26equivariant'>**[1]**</a> Victor Sechaud, Jérémy Scanvic, Quentin Barthélemy, Patrice Abry, and Julián Tachella. Equivariant Splitting: Self-supervised learning from incomplete data. In *The Fourteenth International Conference on Learning Representations (ICLR)*. 2026.

#### A(x, \*\*kwargs)

Apply the virtual operator to an input

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) output of the virtual operator.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### A_adjoint(y, \*\*kwargs)

Apply the adjoint of the virtual operator to an input

* **Parameters:**
  **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input measurement.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) output of the adjoint of the virtual operator.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### A_dagger(y, \*\*kwargs)

Apply the pseudo-inverse of the virtual operator to an input

* **Parameters:**
  **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input measurement.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) output of the pseudo-inverse of the virtual operator.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-physics-virtuallinearphysics"></a>

## Examples using `VirtualLinearPhysics`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Equivariant splitting consists in minimizing a self-supervised loss to train a reconstruction model using measurement data only :footcitesechaud26Equivariant.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Splitting</div>
</div>
<!-- thumbnail-parent-div-close --></div>
