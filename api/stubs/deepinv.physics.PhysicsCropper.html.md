# PhysicsCropper

### *class* deepinv.physics.PhysicsCropper(physics, crop, device='cpu')

Bases: [`LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)

Cropping for linear physics operators.

Given a linear physics operator $A$, this operator instantiates a new operator $\tilde{A} = A \circ C$ where $C$ is a cropping operator that crops the input tensor.
The adjoint operator is defined as $\tilde{A}^{\top} = C^{\top} \circ A^{\top}$ and $C^{\top}$ is a padding operator that pads the input tensor to the original size.

* **Parameters:**
  * **physics** ([*deepinv.physics.LinearPhysics*](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)) – base linear physics operator.
  * **crop** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – padding to apply to the input tensor, e.g., `(pad_height, pad_width)` or `(pad_z, pad_height, pad_weight)` where `pad_z` is either channel or depth dimension pad.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – cpu or cuda, every registered buffer and module parameters are recursively pushed onto the device during initialization.
