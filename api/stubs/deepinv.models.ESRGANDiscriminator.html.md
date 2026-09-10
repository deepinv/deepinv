# ESRGANDiscriminator

### *class* deepinv.models.ESRGANDiscriminator(img_size, filters=(64, 128, 256, 512), dim=2)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

ESRGAN Discriminator.

The ESRGAN discriminator model was originally proposed by Wang *et al.*<sup>[1](#footcite-wang2018esrgan)</sup>. Implementation taken from
[https://github.com/edongdongchen/EI/blob/main/models/discriminator.py](https://github.com/edongdongchen/EI/blob/main/models/discriminator.py).

See [Imaging inverse problems with adversarial networks](https://deepinv.org/auto_examples/adversarial-learning/demo_gan_imaging.html.md#sphx-glr-auto-examples-adversarial-learning-demo-gan-imaging-py) for how to use this for adversarial training.

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – shape of input image
  * **filter** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – Width (number of filters) at each stage. This can also be used to control the number of stages (or also: the output shape relative to input shapes). Defaults to (64, 128, 256, 512)
  * **dim** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int)) – Whether to build 2D or 3D network (if str, can be “2”, “2d”, “3D”, etc.)

<hr />

* **References:**

* <a id='footcite-wang2018esrgan'>**[1]**</a> Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Yu Qiao, and Chen Change Loy. Esrgan: enhanced super-resolution generative adversarial networks. In *Proceedings of the European conference on computer vision (ECCV) workshops*, 0–0. 2018.

#### forward(x)

Forward pass of discriminator model.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image
