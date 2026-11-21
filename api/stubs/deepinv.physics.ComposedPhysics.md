# ComposedPhysics

### *class* deepinv.physics.ComposedPhysics(\*physics, \*\*kwargs)

Bases: [`Physics`](https://deepinv.org/api/stubs/deepinv.physics.Physics.md#deepinv.physics.Physics)

Composes multiple physics operators into a single operator.

The measurements produced by the resulting model are defined as

$$
\noise{\forw{x}} = N_k(A_k(\dots A_2(A_1(x))))
$$

where $A_i(\cdot)$ is the ith physics operator and $N_k(\cdot)$ is the noise of the last operator.

* **Parameters:**
  **physics** (*Iterable* *[*[*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.md#deepinv.physics.Physics) *]*) – variable number of physics to compose.

#### A(x, \*\*kwargs)

Computes forward of composed operator

$$
y = A_k(\dots(A_1(x)))
$$

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – signal/image
* **Returns:**
  measurements
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### update_parameters(\*\*kwargs)

Updates the parameters of each operator in the composed operator.

* **Parameters:**
  **kwargs** ([*dict*](https://docs.python.org/3.12/library/stdtypes.html#dict)) – dictionary of parameters to update.

<a id="sphx-glr-backref-deepinv-physics-composedphysics"></a>

## Examples using `ComposedPhysics`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs projected phase from preprocessed near-field holograms acquired at several propagation distances, see e.g. huhn2022Fast for details on the physics.">![](auto_examples/physics/images/thumb/sphx_glr_demo_fresnel_phase_retrieval_thumb.png)

[Multi-distance Fresnel phase retrieval](https://deepinv.org/auto_examples/physics/demo_fresnel_phase_retrieval.md)

  <div class="sphx-glr-thumbnail-title">Multi-distance Fresnel phase retrieval</div>
</div>
<!-- thumbnail-parent-div-close --></div>
