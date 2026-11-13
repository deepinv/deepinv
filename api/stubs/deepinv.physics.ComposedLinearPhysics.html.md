# ComposedLinearPhysics

### *class* deepinv.physics.ComposedLinearPhysics(\*physics, \*\*kwargs)

Bases: [`ComposedPhysics`](https://deepinv.org/api/stubs/deepinv.physics.ComposedPhysics.html.md#deepinv.physics.ComposedPhysics), [`LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)

Composes multiple linear physics operators into a single operator.

The measurements produced by the resulting model are defined as

$$
\noise{\forw{x}} = N_k(A_k \dots A_2(A_1(x)))
$$

where $A_i(\cdot)$ is the i-th physics operator and $N_k(\cdot)$ is the noise of the last operator.

* **Parameters:**
  **physics** (*Iterable* *[*[*deepinv.physics.LinearPhysics*](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics) *]*) – variable number of physics to compose.

#### A_adjoint(y, \*\*kwargs)

Computes adjoint of composed operator

$$
x = A_1^{\top} A_2^{\top} \dots A_k^{\top} y
$$

* **Parameters:**
  **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements
* **Returns:**
  signal/image
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
