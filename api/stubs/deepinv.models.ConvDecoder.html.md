# ConvDecoder

### *class* deepinv.models.ConvDecoder(img_size, in_size=(4, 4), layers=7, channels=256)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Convolutional decoder network. Supports 2D and 3D data, depending on img_size & in_size

The architecture was introduced by Darestani and Heckel<sup>[1](#footcite-darestani2021accelerated)</sup>,
and it is well suited as a deep image prior (see [`deepinv.models.DeepImagePrior`](https://deepinv.org/api/stubs/deepinv.models.DeepImagePrior.html.md#deepinv.models.DeepImagePrior)).

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – shape of the output image.
  * **in_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – size of the input vector.
  * **layers** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of layers in the network.
  * **channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of channels in the network.

<hr />

* **References:**

* <a id='footcite-darestani2021accelerated'>**[1]**</a> Mohammad Zalbagi Darestani and Reinhard Heckel. Accelerated mri with un-trained neural networks. *IEEE Transactions on Computational Imaging*, 7:724–733, 2021.

#### forward(x, scale_out=1)

Forward pass through the ConvDecoder network.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input tensor.
  * **scale_out** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Output scaling factor.

<a id="sphx-glr-backref-deepinv-models-convdecoder"></a>

## Examples using `ConvDecoder`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to reconstruct a noisy and incomplete image using the deep image prior.">  <div class="sphx-glr-thumbnail-title">Reconstructing an image using the deep image prior.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
