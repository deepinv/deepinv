# PhaseRetrieval

### *class* deepinv.physics.PhaseRetrieval(B, \*\*kwargs)

Bases: [`Physics`](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)

Phase Retrieval base class corresponding to the operator

$$
\forw{x} = |Bx|^2.
$$

The linear operator $B$ is defined by a [`deepinv.physics.LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics) object.

An existing operator can be loaded from a saved .pth file via `self.load_state_dict(save_path)`, in a similar fashion to [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).

* **Parameters:**
  **B** ([*deepinv.physics.forward.LinearPhysics*](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)) – the linear forward operator.

#### A(x, \*\*kwargs)

Applies the forward operator to the input x.

Note here the operation includes the modulus operation.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – signal/image.

#### A_dagger(y, \*\*kwargs)

Computes an initial reconstruction for the image $x$ from the measurements $y$.

We use the spectral methods defined in [`deepinv.optim.phase_retrieval.spectral_methods`](https://deepinv.org/api/stubs/deepinv.optim.phase_retrieval.spectral_methods.html.md#deepinv.optim.phase_retrieval.spectral_methods) to obtain an initial inverse.

* **Parameters:**
  **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) an initial reconstruction for image $x$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### A_vjp(x, v)

Computes the product between a vector $v$ and the Jacobian of the forward operator $A$ at the input x, defined as:

$$
A_{vjp}(x, v) = 2 \overline{B}^{\top} \text{diag}(Bx) v.
$$

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – signal/image.
  * **v** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – vector.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) the VJP product between $v$ and the Jacobian.

#### B_dagger(y)

Computes the linear pseudo-inverse of $B$.

* **Parameters:**
  **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) the reconstruction image $x$.

#### forward(x, \*\*kwargs)

Applies the phase retrieval measurement operator, i.e. $y = \noise{|Bx|^2}$ (with noise $N$ and/or sensor non-linearities).

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,*[*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – signal/image
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) noisy measurements

<a id="sphx-glr-backref-deepinv-physics-phaseretrieval"></a>

## Examples using `PhaseRetrieval`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to create a random phase retrieval operator and generate phaseless measurements from a given image. The example showcases 4 different reconstruction methods to recover the image from the phaseless measurements:">  <div class="sphx-glr-thumbnail-title">Random phase retrieval and reconstruction methods.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to create a Ptychography phase retrieval operator and generate phaseless measurements from a given image.">  <div class="sphx-glr-thumbnail-title">Ptychography phase retrieval</div>
</div>
<!-- thumbnail-parent-div-close --></div>
