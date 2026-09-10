# PDNet_DualBlock

### *class* deepinv.models.PDNet_DualBlock(in_channels=7, out_channels=5, depth=3, bias=True, nf=32, dim=2)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Dual block for the Primal-Dual unfolding model.

First introduced by Adler and Öktem<sup>[1](#footcite-adler2018learned)</sup>.

Dual variables are images of shape (batch_size, in_channels, height, width). The input of each
primal block is the concatenation of the current dual variable with the projected primal variable and
the measurements. The output of each dual block is the current primal variable.

* **Parameters:**
  * **in_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of input channels. Default: 7.
  * **out_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of output channels. Default: 5.
  * **depth** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of convolutional layers in the block. Default: 3.
  * **bias** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to use bias in convolutional layers. Default: True.
  * **nf** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of features in the convolutional layers. Default: 32.
  * **dim** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int)) – Whether to build 2D or 3D network (if str, can be “2”, “2d”, “3D”, etc.)

<hr />

* **References:**

* <a id='footcite-adler2018learned'>**[1]**</a> Jonas Adler and Ozan Öktem. Learned primal-dual reconstruction. *IEEE transactions on medical imaging*, 37(6):1322–1332, 2018.

#### forward(u, Ax_cur, y)

Forward pass of the dual block.

* **Parameters:**
  * **u** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – current dual variable.
  * **Ax_cur** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – projection of the primal variable.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements.

<a id="sphx-glr-backref-deepinv-models-pdnet-dualblock"></a>

## Examples using `PDNet_DualBlock`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="where both the data fidelity and the prior are learned modules, distinct for each iterations.">  <div class="sphx-glr-thumbnail-title">Learned Primal-Dual algorithm for CT scan.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
