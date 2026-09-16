# PDNet_PrimalBlock

### *class* deepinv.models.PDNet_PrimalBlock(in_channels=6, out_channels=5, depth=3, bias=True, nf=32, dim=2)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Primal block for the Primal-Dual unfolding model.

First introduced by Adler and Öktem<sup>[1](#footcite-adler2018learned)</sup>.

Primal variables are images of shape (batch_size, in_channels, height, width). The input of each
primal block is the concatenation of the current primal variable and the backprojected dual variable along
the channel dimension. The output of each primal block is the current primal variable.

* **Parameters:**
  * **in_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of input channels. Default: 6.
  * **out_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of output channels. Default: 5.
  * **depth** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of convolutional layers in the block. Default: 3.
  * **bias** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to use bias in convolutional layers. Default: True.
  * **nf** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of features in the convolutional layers. Default: 32.
  * **dim** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int)) – Whether to build 2D or 3D network (if str, can be “2”, “2d”, “3D”, etc.)

<hr />

* **References:**

* <a id='footcite-adler2018learned'>**[1]**</a> Jonas Adler and Ozan Öktem. Learned primal-dual reconstruction. *IEEE transactions on medical imaging*, 37(6):1322–1332, 2018.

#### forward(x, Atu)

Forward pass of the primal block.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – current primal variable.
  * **Atu** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – backprojected dual variable.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) the current primal variable.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-models-pdnet-primalblock"></a>

## Examples using `PDNet_PrimalBlock`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="where both the data fidelity and the prior are learned modules, distinct for each iterations.">  <div class="sphx-glr-thumbnail-title">Learned Primal-Dual algorithm for CT scan.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
