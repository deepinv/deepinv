# TimeAgnosticNet

### *class* deepinv.models.TimeAgnosticNet(backbone_net)

Bases: [`Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor), [`TimeMixin`](https://deepinv.org/api/stubs/deepinv.utils.TimeMixin.html.md#deepinv.utils.TimeMixin)

Time-agnostic network wrapper.

Adapts a static image reconstruction network to process time-varying inputs.
The image reconstruction network then processes the data independently frame-by-frame.

Flattens time dimension into batch dimension at input, and unflattens at output.

<hr />

* **Example:**

```pycon
>>> import torch
>>> from deepinv.models import UNet, TimeAgnosticNet
>>> model = UNet(scales=2)
>>> model = TimeAgnosticNet(model)
>>> y = torch.rand(1, 1, 4, 8, 8) # B,C,T,H,W
>>> x_net = model(y, None)
>>> x_net.shape == y.shape
True
```

* **Parameters:**
  * **backbone_net** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Base network which can only take static inputs (B,C,H,W)
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – cpu or gpu.

#### forward(y, physics, \*\*kwargs)

Reconstructs a signal estimate from measurements y

* **Parameters:**
  * **y** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements `(B,C,T,H,W)`
  * **physics** ([*Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – forward operator acting on dynamic inputs
