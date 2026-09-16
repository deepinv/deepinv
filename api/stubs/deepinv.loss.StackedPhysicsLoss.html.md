# StackedPhysicsLoss

### *class* deepinv.loss.StackedPhysicsLoss(losses)

Bases: [`Loss`](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss)

Loss function for stacked physics operators.

Adapted to [`deepinv.physics.StackedPhysics`](https://deepinv.org/api/stubs/deepinv.physics.StackedPhysics.html.md#deepinv.physics.StackedPhysics) physics composed of multiple physics operators.

* **Parameters:**
  **losses** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*deepinv.loss.Loss*](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss) *]*) – list of loss functions for each physics operator.

<hr />

* **Examples:**
  Gaussian and Poisson losses function for a stacked physics operator:
  ```pycon
  >>> import torch
  >>> import deepinv as dinv
  >>> # define two observations, one with Gaussian noise and one with Poisson noise
  >>> physics1 = dinv.physics.Denoising(dinv.physics.GaussianNoise(.1))
  >>> physics2 = dinv.physics.Denoising(dinv.physics.PoissonNoise(.1))
  >>> physics = dinv.physics.StackedLinearPhysics([physics1, physics2])
  >>> loss1 = dinv.loss.SureGaussianLoss(.1)
  >>> loss2 = dinv.loss.SurePoissonLoss(.1)
  >>> loss = dinv.loss.StackedPhysicsLoss([loss1, loss2])
  >>> x = torch.ones(1, 1, 5, 5) # image
  >>> y = physics(x) # noisy measurements
  >>> # define a denoiser model
  >>> model = dinv.models.ArtifactRemoval(dinv.models.MedianFilter(3))
  >>> x_net = model(y, physics)
  >>> l = loss(x_net, x, y, physics, model)
  ```

#### forward(x_net, x, y, physics, model, \*\*kwargs)

Computes the loss as

$$
\sum_i \mathcal{L}_i(x, y_i, \inverse{y}, \physics_i, \model),
$$

where $i$ is the index of the physics operator in the stacked physics.

* **Parameters:**
  * **y** ([*deepinv.utils.TensorList*](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList)) – Measurement.
  * **x_net** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Reconstructed image $\inverse{y}$.

#### *property* name *: [bool](https://docs.python.org/3.9/library/functions.html#bool)*

The name of the loss function. This attribute is deprecated in favor of the class name and it will be removed in a future version.

<a id="sphx-glr-backref-deepinv-loss-stackedphysicsloss"></a>

## Examples using `StackedPhysicsLoss`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In this example we demonstrate remote sensing inverse problems for multispectral satellite imaging.">  <div class="sphx-glr-thumbnail-title">Remote sensing with satellite images</div>
</div>
<!-- thumbnail-parent-div-close --></div>
