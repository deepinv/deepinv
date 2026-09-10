# BaseLossScheduler

### *class* deepinv.loss.BaseLossScheduler(\*loss, generator=None)

Bases: [`Loss`](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss)

Base class for loss schedulers.

Wraps a list of losses, and each time forward is called, some of them are selected based on a defined schedule.

* **Parameters:**
  * **\*loss** ([*Loss*](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss)) – loss or multiple losses to be scheduled.
  * **generator** (*Generator*) – torch random number generator, defaults to None

#### adapt_model(model, \*\*kwargs)

Adapt model using all wrapped losses.

Some loss functions require the model forward call to be adapted before the forward pass.

* **Parameters:**
  **model** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – reconstruction model

#### forward(x_net=None, x=None, y=None, physics=None, model=None, epoch=None, \*\*kwargs)

Loss forward pass.

When called, subselect losses based on defined schedule to be used at this pass, and apply to inputs.

* **Parameters:**
  * **x_net** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – model output
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – ground truth
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurement
  * **physics** ([*Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – measurement operator
  * **model** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – reconstruction model
  * **epoch** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – current epoch

#### schedule(epoch)

Return selected losses based on defined schedule, optionally based on current epoch.

* **Parameters:**
  **epoch** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – current epoch number
* **Return list[Loss]:**
  selected (sub)list of losses to be used this time.
* **Return type:**
  [list](https://docs.python.org/3.9/library/stdtypes.html#list)[[*Loss*](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss)]
