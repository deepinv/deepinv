# TimeAveragingNet

### *class* deepinv.models.TimeAveragingNet(backbone_net)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module), [`TimeMixin`](https://deepinv.org/api/stubs/deepinv.utils.TimeMixin.html.md#deepinv.utils.TimeMixin)

Time-averaging network wrapper.

Adapts a static image reconstruction network for time-varying inputs to output static reconstructions.
Average the data across the time dim before passing into network.

#### NOTE
The input physics is assumed to be a temporal physics which produced the temporal measurements y (potentially with temporal mask `mask`).
It must either implement a `to_static` method to remove the time dimension, or already be a static physics (e.g. [`deepinv.physics.MRI`](https://deepinv.org/api/stubs/deepinv.physics.MRI.html.md#deepinv.physics.MRI)).

<hr />

* **Example:**

```pycon
>>> import torch
>>> from deepinv.models import UNet, TimeAveragingNet
>>> model = UNet(scales=2)
>>> model = TimeAveragingNet(model)
>>> y = torch.rand(1, 1, 4, 8, 8) # B,C,T,H,W
>>> x_net = model(y, None)
>>> x_net.shape # B,C,H,W
torch.Size([1, 1, 8, 8])
```

* **Parameters:**
  * **backbone_net** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Base network which can only take static inputs (B,C,H,W)
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – cpu or gpu.

#### forward(y, physics, \*\*kwargs)

Evaluate the network

* **Parameters:**
  * **y** – measurements
  * **physics** – forward operator acting on dynamic inputs
