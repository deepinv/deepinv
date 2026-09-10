# PanNet

### *class* deepinv.models.PanNet(backbone_net=None, hrms_shape=(4, 900, 900), scale_factor=4, highpass_kernel_size=5, device='cpu', \*\*kwargs)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

PanNet architecture for pan-sharpening.

PanNet neural network from Yang *et al.*<sup>[1](#footcite-yang2017pannet)</sup>.

Takes input measurements as a [`deepinv.utils.TensorList`](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) with elements (MS, PAN),
where MS is the low-resolution multispectral image of shape (B, C, H, W) and PAN is the
high-resolution panchromatic image of shape (B, 1, H\*r, W\*r) where r is the pan-sharpening factor.

* **Parameters:**
  * **backbone_net** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Backbone neural network, e.g. ResNet. If `None`, defaults to a simple ResNet.
  * **hrms_shape** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – shape of high-resolution multispectral images (C,H,W), defaults to (4,900,900)
  * **scale_factor** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – pansharpening downsampling ratio HR/LR, defaults to 4
  * **highpass_kernel_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – square kernel size for extracting high-frequency features, defaults to 5
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – torch device, defaults to “cpu”

<hr />

* **References:**

* <a id='footcite-yang2017pannet'>**[1]**</a> Junfeng Yang, Xueyang Fu, Yuwen Hu, Yue Huang, Xinghao Ding, and John Paisley. Pannet: a deep network architecture for pan-sharpening. In *Proceedings of the IEEE international conference on computer vision*, 5449–5457. 2017.

#### create_sampler(direction, hr_shape, noise_gain=0.0)

Helper function for downsampling/upsampling images (useful for reduced-resolution training with Wald’s protocol).

* **Parameters:**
  * **direction** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – down or up
  * **hr_shape** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – HRMS input shape (C,H,W)
  * **noise_gain** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – noise applied to downsampling ONLY, defaults to 0.
* **Return dinv.physics.Physics:**
  deepinv sampler
* **Return type:**
  [*Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)

#### forward(y, physics, \*args, \*\*kwargs)

Evaluate the pansharpening model

* **Parameters:**
  * **y** ([*deepinv.utils.TensorList*](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList)) – (MS,PAN) images
  * **physics** ([*deepinv.physics.Pansharpen*](https://deepinv.org/api/stubs/deepinv.physics.Pansharpen.html.md#deepinv.physics.Pansharpen)) – Pansharpening operator

<a id="sphx-glr-backref-deepinv-models-pannet"></a>

## Examples using `PanNet`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In this example we demonstrate remote sensing inverse problems for multispectral satellite imaging.">  <div class="sphx-glr-thumbnail-title">Remote sensing with satellite images</div>
</div>
<!-- thumbnail-parent-div-close --></div>
