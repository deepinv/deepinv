# compose

### deepinv.physics.compose(\*physics, \*\*kwargs)

Composes multiple forward operators $A = A_1\circ A_2\circ \dots \circ A_n$.

The measurements produced by the resulting model are [`deepinv.utils.TensorList`](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) objects, where
each entry corresponds to the measurements of the corresponding operator.

* **Parameters:**
  **physics** (*Iterable* *[*[*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics) *|* [*deepinv.physics.LinearPhysics*](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics) *]*) – Physics operators $A_i$ to be composed.

## Examples using `compose`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs projected phase from preprocessed near-field holograms acquired at several propagation distances, see e.g. huhn2022Fast for details on the physics.">![](auto_examples/physics/images/thumb/sphx_glr_demo_fresnel_phase_retrieval_thumb.png)

[Multi-distance Fresnel phase retrieval](https://deepinv.org/auto_examples/physics/demo_fresnel_phase_retrieval.html.md)

  <div class="sphx-glr-thumbnail-title">Multi-distance Fresnel phase retrieval</div>
</div>
<!-- thumbnail-parent-div-close --></div>
