# Phase2PhaseLoss

### *class* deepinv.loss.mri.Phase2PhaseLoss(img_size, dynamic_model=True, metric=None, device='cpu')

Bases: [`SplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.SplittingLoss.html.md#deepinv.loss.SplittingLoss)

Phase2Phase loss for dynamic data.

Implements dynamic measurement splitting loss from Eldeniz *et al.*<sup>[1](#footcite-eldeniz2021phase2phase)</sup> for free-breathing MRI.
This is a special (temporal) case of the generic splitting loss: see [`deepinv.loss.SplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.SplittingLoss.html.md#deepinv.loss.SplittingLoss) for more details.

Splits the dynamic measurements into even time frames (“phases”) at model input and odd phases to use for constructing the loss.
Equally, the physics mask (if it exists) is split as well: the even phases are used for the model (e.g. for data consistency in an unrolled network) and odd phases are used for the reference.
At test time, the full input is passed through the network.

#### WARNING
The model should be adapted before training using the method [`adapt_model`](https://deepinv.org/api/stubs/deepinv.loss.SplittingLoss.html.md#deepinv.loss.SplittingLoss.adapt_model)
to include the splitting mechanism at the input.

#### WARNING
Must only be used for dynamic or sequential measurements, i.e. where data $y$ and `physics.mask` (if it exists) are of 5D shape (B, C, T, H, W).

#### NOTE
Phase2Phase can be used to reconstruct video sequences by setting `dynamic_model=True` and using physics [`deepinv.physics.DynamicMRI`](https://deepinv.org/api/stubs/deepinv.physics.DynamicMRI.html.md#deepinv.physics.DynamicMRI).
It can also be used to reconstructs **static** images, where the k-space measurements is a time-sequence,
where each time step (phase) consists of sampled spokes such that the whole measurement is a set of non-overlapping spokes.
To do this, set `dynamic_model=False` and use physics [`deepinv.physics.SequentialMRI`](https://deepinv.org/api/stubs/deepinv.physics.SequentialMRI.html.md#deepinv.physics.SequentialMRI). See below for example or [Self-supervised MRI reconstruction with Artifact2Artifact](https://deepinv.org/auto_examples/self-supervised-learning/demo_artifact2artifact.html.md#sphx-glr-auto-examples-self-supervised-learning-demo-artifact2artifact-py) for full MRI example.

By default, the error is computed using the MSE metric, however any appropriate metric can be used.

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – size of the tensor to be masked without batch dimension of shape (C, T, H, W)
  * **dynamic_model** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – set `True` if using with a model that inputs and outputs time-data i.e. `x` of shape (B,C,T,H,W). Set `False` if `x` are static images (B,C,H,W).
  * **metric** ([*Metric*](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric) *,* [*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – metric used for computing data consistency, which is set as the mean squared error by default.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – torch device.

<hr />

* **Example:**
  Dynamic MRI with Phase2Phase with a video network:
  ```pycon
  >>> import torch
  >>> from deepinv.models import AutoEncoder, TimeAgnosticNet
  >>> from deepinv.physics import DynamicMRI, SequentialMRI
  >>> from deepinv.loss.mri import Phase2PhaseLoss
  >>>
  >>> x = torch.rand((1, 2, 4, 4, 4)) # B, C, T, H, W
  >>> mask = torch.zeros((1, 2, 4, 4, 4))
  >>> mask[:, :, torch.arange(4), torch.arange(4) % 4, :] = 1 # Create time-varying mask
  >>>
  >>> physics = DynamicMRI(mask=mask)
  >>> loss = Phase2PhaseLoss((2, 4, 4, 4))
  >>> model = TimeAgnosticNet(AutoEncoder(32, 2, 2)) # Example video network
  >>> model = loss.adapt_model(model) # Adapt model to perform Phase2Phase
  >>>
  >>> y = physics(x)
  >>> x_net = model(y, physics, update_parameters=True) # save random mask in forward pass
  >>> l = loss(x_net, y, physics, model)
  >>> print(l.item() > 0)
  True
  ```

  Free-breathing MRI with Phase2Phase with an image network and sequential measurements:
  ```pycon
  >>> physics = SequentialMRI(mask=mask) # mask is B, C, T, H, W
  >>> loss = Phase2PhaseLoss((2, 4, 4, 4), dynamic_model=False) # Process static images x
  >>>
  >>> model = AutoEncoder(32, 2, 2) # Example image reconstruction network
  >>> model = loss.adapt_model(model) # Adapt model to perform Phase2Phase
  >>>
  >>> x = torch.rand((1, 2, 4, 4)) # B, C, H, W
  >>> y = physics(x) # B, C, T, H, W
  >>> x_net = model(y, physics, update_parameters=True)
  >>> l = loss(x_net, y, physics, model)
  >>> print(l.item() > 0)
  True
  ```

<hr />

* **References:**

* <a id='footcite-eldeniz2021phase2phase'>**[1]**</a> Cihat Eldeniz, Weijie Gan, Sihao Chen, Tyler J Fraum, Daniel R Ludwig, Yan Yan, Jiaming Liu, Thomas Vahle, Uday Krishnamurthy, Ulugbek S Kamilov, and others. Phase2phase: respiratory motion-resolved reconstruction of free-breathing magnetic resonance imaging using deep learning without a ground truth for improved liver imaging. *Investigative Radiology*, 56(12):809–819, 2021.

#### adapt_model(model, \*\*kwargs)

Apply Phase2Phase splitting to model input. Also perform time-averaging if a static model is used.

* **Parameters:**
  **model** ([*deepinv.models.Reconstructor*](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor) *,* [*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Reconstruction model.
* **Returns:**
  ([`deepinv.loss.SplittingLoss.SplittingModel`](https://deepinv.org/api/stubs/deepinv.loss.SplittingLoss.html.md#deepinv.loss.SplittingLoss.SplittingModel)) Model modified for evaluation.
* **Return type:**
  [*SplittingModel*](https://deepinv.org/api/stubs/deepinv.loss.SplittingLoss.html.md#deepinv.loss.SplittingLoss.SplittingModel)

#### *static* split(mask, y, physics=None)

Override splitting to actually remove masked pixels. In Phase2Phase, this corresponds to masked phases (i.e. time steps).

* **Parameters:**
  * **mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Phase2Phase mask
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input data
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – forward physics

<a id="sphx-glr-backref-deepinv-loss-mri-phase2phaseloss"></a>

## Examples using `Phase2PhaseLoss`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the self-supervised Artifact2Artifact loss for solving an undersampled sequential MRI reconstruction problem without ground truth.">  <div class="sphx-glr-thumbnail-title">Self-supervised MRI reconstruction with Artifact2Artifact</div>
</div>
<!-- thumbnail-parent-div-close --></div>
