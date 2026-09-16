# Artifact2ArtifactLoss

### *class* deepinv.loss.mri.Artifact2ArtifactLoss(img_size, split_size=2, dynamic_model=True, metric=None, device='cpu')

Bases: [`Phase2PhaseLoss`](https://deepinv.org/api/stubs/deepinv.loss.mri.Phase2PhaseLoss.html.md#deepinv.loss.mri.Phase2PhaseLoss)

Artifact2Artifact loss for dynamic data.

Implements dynamic measurement splitting loss from Liu *et al.*<sup>[1](#footcite-liu2020rare)</sup> for free-breathing MRI.
This is a special case of the generic splitting loss: see [`deepinv.loss.SplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.SplittingLoss.html.md#deepinv.loss.SplittingLoss) for more details.

At model input, choose a random time-chunk from the dynamic measurements (“Artifact…”), and another random chunk for constructing the loss (”…2Artifact”).
Equally, the physics mask (if it exists) is split as well: the input chunk is used for the model (e.g. for data consistency in an unrolled network) and the output chunk is used as the reference.
At test time, the full input is passed through the network.
Note this implementation performs a Monte-Carlo-style version where the network output is only compared to one other chunk per iteration.

#### WARNING
The model should be adapted before training using the method [`adapt_model`](https://deepinv.org/api/stubs/deepinv.loss.SplittingLoss.html.md#deepinv.loss.SplittingLoss.adapt_model)
to include the splitting mechanism at the input.

#### WARNING
Must only be used for dynamic or sequential measurements, i.e. where data $y$ and `physics.mask` (if it exists) are of 5D shape (B, C, T, H, W).

#### NOTE
Artifact2Artifact can be used to reconstruct video sequences by setting `dynamic_model=True` and using physics [`deepinv.physics.DynamicMRI`](https://deepinv.org/api/stubs/deepinv.physics.DynamicMRI.html.md#deepinv.physics.DynamicMRI).
It can also be used to reconstructs **static** images, where the k-space measurements is a time-sequence,
where each time step (phase) consists of sampled spokes such that the whole measurement is a set of non-overlapping spokes.
To do this, set `dynamic_model=False` and use physics [`deepinv.physics.SequentialMRI`](https://deepinv.org/api/stubs/deepinv.physics.SequentialMRI.html.md#deepinv.physics.SequentialMRI). See below for example or [Self-supervised MRI reconstruction with Artifact2Artifact](https://deepinv.org/auto_examples/self-supervised-learning/demo_artifact2artifact.html.md#sphx-glr-auto-examples-self-supervised-learning-demo-artifact2artifact-py) for full MRI example.

By default, the error is computed using the MSE metric, however any appropriate metric can be used.

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – size of the tensor to be masked without batch dimension of shape (C, T, H, W)
  * **split_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – time-length of chunk. Must divide `img_size[1]` exactly. If `tuple`, one is randomly selected each time.
  * **dynamic_model** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – set True if using with a model that inputs and outputs time-data i.e. x of shape (B,C,T,H,W). Set False if x are static images (B,C,H,W).
  * **metric** ([*Metric*](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric) *,* [*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – metric used for computing data consistency, which is set as the mean squared error by default.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – torch device.

<hr />

* **Example:**
  Dynamic MRI with Artifact2Artifact with a video network:
  ```pycon
  >>> import torch
  >>> from deepinv.models import AutoEncoder, TimeAgnosticNet
  >>> from deepinv.physics import DynamicMRI, SequentialMRI
  >>> from deepinv.loss.mri import Artifact2ArtifactLoss
  >>>
  >>> x = torch.rand((1, 2, 4, 4, 4)) # B, C, T, H, W
  >>> mask = torch.zeros((1, 2, 4, 4, 4))
  >>> mask[:, :, torch.arange(4), torch.arange(4) % 4, :] = 1 # Create time-varying mask
  >>>
  >>> physics = DynamicMRI(mask=mask)
  >>> loss = Artifact2ArtifactLoss((2, 4, 4, 4))
  >>> model = TimeAgnosticNet(AutoEncoder(32, 2, 2)) # Example video network
  >>> model = loss.adapt_model(model) # Adapt model to perform Artifact2Artifact
  >>>
  >>> y = physics(x)
  >>> x_net = model(y, physics, update_parameters=True) # save random mask in forward pass
  >>> l = loss(x_net, y, physics, model)
  >>> print(l.item() > 0)
  True
  ```

  Free-breathing MRI with Artifact2Artifact with an image network and sequential measurements:
  ```pycon
  >>> physics = SequentialMRI(mask=mask) # mask is B, C, T, H, W
  >>> loss = Artifact2ArtifactLoss((2, 4, 4, 4), dynamic_model=False) # Process static images x
  >>>
  >>> model = AutoEncoder(32, 2, 2) # Example image reconstruction network
  >>> model = loss.adapt_model(model) # Adapt model to perform Artifact2Artifact
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

* <a id='footcite-liu2020rare'>**[1]**</a> Jiaming Liu, Yu Sun, Cihat Eldeniz, Weijie Gan, Hongyu An, and Ulugbek S Kamilov. Rare: image reconstruction using deep priors learned without groundtruth. *IEEE Journal of Selected Topics in Signal Processing*, 14(6):1088–1099, 2020.

#### *property* name *: [bool](https://docs.python.org/3.9/library/functions.html#bool)*

The name of the loss function. This attribute is deprecated in favor of the class name and it will be removed in a future version.

<a id="sphx-glr-backref-deepinv-loss-mri-artifact2artifactloss"></a>

## Examples using `Artifact2ArtifactLoss`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the self-supervised Artifact2Artifact loss for solving an undersampled sequential MRI reconstruction problem without ground truth.">  <div class="sphx-glr-thumbnail-title">Self-supervised MRI reconstruction with Artifact2Artifact</div>
</div>
<!-- thumbnail-parent-div-close --></div>
