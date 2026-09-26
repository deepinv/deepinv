# StackedLinearPhysics

### *class* deepinv.physics.StackedLinearPhysics(physics_list, reduction='sum', \*\*kwargs)

Bases: [`StackedPhysics`](https://deepinv.org/api/stubs/deepinv.physics.StackedPhysics.md#deepinv.physics.StackedPhysics), [`LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.md#deepinv.physics.LinearPhysics)

Stacks multiple linear physics operators into a single operator.

The measurements produced by the resulting model are [`deepinv.utils.TensorList`](https://deepinv.org/api/stubs/deepinv.utils.TensorList.md#deepinv.utils.TensorList) objects, where
each entry corresponds to the measurements of the corresponding operator.

See [Combining Physics](https://deepinv.org/user_guide/physics/intro.md#physics-combining) for more information.

* **Parameters:**
  * **physics_list** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.md#deepinv.physics.Physics) *]*) – list of physics operators to stack.
  * **reduction** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – how to combine tensorlist outputs of adjoint operators into single
    adjoint output. Choose between `sum`, `mean` or `None`.

#### A_adjoint(y, \*\*kwargs)

Computes the adjoint of the stacked operator, defined as

$$
A^{\top}y = \sum_{i=1}^{n} A_i^{\top}y_i.
$$

* **Parameters:**
  **y** ([*deepinv.utils.TensorList*](https://deepinv.org/api/stubs/deepinv.utils.TensorList.md#deepinv.utils.TensorList)) – measurements

<a id="sphx-glr-backref-deepinv-physics-stackedlinearphysics"></a>

## Examples using `StackedLinearPhysics`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Many large-scale imaging problems involve operators that can be naturally decomposed as a stack of multiple sub-operators:">![](auto_examples/distributed/images/thumb/sphx_glr_demo_physics_distributed_thumb.png)

[Distributed Physics Operators](https://deepinv.org/auto_examples/distributed/demo_physics_distributed.md)

  <div class="sphx-glr-thumbnail-title">Distributed Physics Operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Many large-scale imaging problems involve operators that can be naturally decomposed as a stack of multiple sub-operators:">![](auto_examples/distributed/images/thumb/sphx_glr_demo_pnp_distributed_thumb.png)

[Distributed Plug-and-Play (PnP) Reconstruction](https://deepinv.org/auto_examples/distributed/demo_pnp_distributed.md)

  <div class="sphx-glr-thumbnail-title">Distributed Plug-and-Play (PnP) Reconstruction</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In many large-scale imaging problems, the size of the image/volume to reconstruct is very large, making it impossible to train reconstruction networks (in this example, unfolded networks) with a single GPU. The deepinv.distributed framework enables training a model on multiple GPUs, by carefully parallelizing the data fidelity and denoising steps inside the network.">![](auto_examples/distributed/images/thumb/sphx_glr_demo_unrolled_distributed_thumb.png)

[Distributed Training of Unfolded Networks](https://deepinv.org/auto_examples/distributed/demo_unrolled_distributed.md)

  <div class="sphx-glr-thumbnail-title">Distributed Training of Unfolded Networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in here.">![](auto_examples/physics/images/thumb/sphx_glr_demo_physics_tour_thumb.png)

[Tour of forward sensing operators](https://deepinv.org/auto_examples/physics/demo_physics_tour.md)

  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we demonstrate remote sensing inverse problems for multispectral satellite imaging.">![](auto_examples/physics/images/thumb/sphx_glr_demo_remote_sensing_thumb.png)

[Remote sensing with satellite images](https://deepinv.org/auto_examples/physics/demo_remote_sensing.md)

  <div class="sphx-glr-thumbnail-title">Remote sensing with satellite images</div>
</div>
<!-- thumbnail-parent-div-close --></div>
