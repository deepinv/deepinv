# ReducedResolutionLoss

### *class* deepinv.loss.ReducedResolutionLoss(metric=None, physics=None)

Bases: [`SupLoss`](https://deepinv.org/api/stubs/deepinv.loss.SupLoss.html.md#deepinv.loss.SupLoss)

Reduced resolution loss for blur and downsampling problems.

The reduced resolution loss is defined as

$$
\frac{1}{n}\|y-\inverse{\forw{y}}\|^2
$$

where $\forw{y}$ is the reduced resolution measurement via further degrading, and the measurement $y$ is used a supervisory signal.

#### NOTE
Optionally initialize with physics to fix the reduced resolution operator. If not passed, the loss takes the physics from the forward pass during training.
However, this should only be used with physics that can be used to meaningfully further degrade the measurements
$y$, such as blur or downsampling. The physics must be defined without an `img_size` so it can be applied
to the measurements $y$.

At test time, the model does not perform the reduced resolution measurement.

#### HINT
During training, consider using the `compute_train_metrics=False` option in [`deepinv.Trainer`](https://deepinv.org/api/stubs/deepinv.Trainer.html.md#deepinv.Trainer) to prevent a shape
mismatch during metric computation since the reduced resolution output will smaller than ground truth.

This loss was used in Shocher *et al.*<sup>[1](#footcite-shocher2017zero-shot)</sup> for downsampling tasks, and is named Wald’s protocol <sup>[2](#footcite-wald1997fusion)</sup>
for pan-sharpening tasks.

* **Parameters:**
  * **metric** ([*Metric*](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric) *,* [*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – metric used for computing data consistency,
    which is set as the mean squared error by default.
  * **physics** ([*Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – optional physics to perform reduced resolution measurement. If not specified, take the physics from the forward pass.

<hr />

* **References:**

* <a id='footcite-shocher2017zero-shot'>**[1]**</a> Assaf Shocher, Nadav Cohen, and Michal Irani. “zero-shot” super-resolution using deep internal learning. 2017. [doi:10.48550/arXiv.1712.06087](https://doi.org/10.48550/arXiv.1712.06087).
* <a id='footcite-wald1997fusion'>**[2]**</a> L. Wald, T. Ranchin, and Marc Mangolini. Fusion of satellite images of different spatial resolutions: assessing the quality of resulting images. *Photogrammetric Engineering and Remote Sensing*, 1997.

#### forward(x_net, y, \*args, \*\*kwargs)

Computes the reduced resolution loss.

* **Parameters:**
  * **x_net** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – reconstructions.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurements.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Forward operator associated with the measurements.
  * **model** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Reconstruction function.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) loss.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
