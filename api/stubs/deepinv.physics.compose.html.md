# compose

### deepinv.physics.compose(\*physics, \*\*kwargs)

Composes multiple forward operators $A = A_1\circ A_2\circ \dots \circ A_n$.

The measurements produced by the resulting model are [`deepinv.utils.TensorList`](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) objects, where
each entry corresponds to the measurements of the corresponding operator.

* **Parameters:**
  **physics** (*Iterable* *[*[*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics) *|* [*deepinv.physics.LinearPhysics*](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics) *]*) – Physics operators $A_i$ to be composed.
