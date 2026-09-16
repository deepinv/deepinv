# StackedPhysics

### *class* deepinv.physics.StackedPhysics(physics_list, \*\*kwargs)

Bases: [`Physics`](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)

Stacks multiple physics operators into a single operator.

The measurements produced by the resulting model are [`deepinv.utils.TensorList`](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) objects, where
each entry corresponds to the measurements of the corresponding operator.

See [Combining Physics](https://deepinv.org/user_guide/physics/intro.html.md#physics-combining) for more information.

* **Parameters:**
  **physics_list** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics) *]*) – list of physics operators to stack.

#### A(x, \*\*kwargs)

Computes forward of stacked operator

$$
y = \begin{bmatrix} A_1(x) \\ A_2(x) \\ \vdots \\ A_n(x) \end{bmatrix}
$$

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – signal/image
* **Returns:**
  measurements
* **Return type:**
  [*TensorList*](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList)

#### noise(y, \*\*kwargs)

Applies noise to the measurements per physics operator in the stacked operator.

* **Parameters:**
  **y** ([*deepinv.utils.TensorList*](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList)) – measurements
* **Returns:**
  noisy measurements
* **Return type:**
  [*TensorList*](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList)

#### sensor(y, \*\*kwargs)

Applies sensor non-linearities to the measurements per physics operator
in the stacked operator.

* **Parameters:**
  **y** ([*deepinv.utils.TensorList*](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList)) – measurements
* **Returns:**
  measurements
* **Return type:**
  [*TensorList*](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList)

#### set_noise_model(noise_model, item=0)

Sets the noise model for the physics operator at index `item`.

* **Parameters:**
  * **noise_model** (*Callable* *,* [*deepinv.physics.NoiseModel*](https://deepinv.org/api/stubs/deepinv.physics.NoiseModel.html.md#deepinv.physics.NoiseModel)) – noise model for the physics operator.
  * **item** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – index of the physics operator

#### update_parameters(\*\*kwargs)

Updates the parameters of the stacked operator.

* **Parameters:**
  **kwargs** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – dictionary of parameters to update.

<a id="sphx-glr-backref-deepinv-physics-stackedphysics"></a>

## Examples using `StackedPhysics`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in physics.">  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we demonstrate remote sensing inverse problems for multispectral satellite imaging.">  <div class="sphx-glr-thumbnail-title">Remote sensing with satellite images</div>
</div>
<!-- thumbnail-parent-div-close --></div>
