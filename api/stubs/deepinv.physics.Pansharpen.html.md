# Pansharpen

### *class* deepinv.physics.Pansharpen(img_size, filter='bilinear', factor=4, srf='flat', noise_color=None, noise_gray=None, use_brovey=True, device='cpu', padding='circular', normalize=False, eps=1e-6, \*\*kwargs)

Bases: [`StackedLinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.StackedLinearPhysics.html.md#deepinv.physics.StackedLinearPhysics)

Pansharpening forward operator.

The measurements consist of a high resolution grayscale image and a low resolution RGB image, and
are represented using [`deepinv.utils.TensorList`](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList), where the first element is the RGB image and the second
element is the grayscale image.

By default, the downsampling is done with a gaussian filter with standard deviation equal to the downsampling,
however, the user can provide a custom downsampling filter.

It is possible to assign a different noise model to the RGB and grayscale images.

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – size of the high-resolution multispectral input image, must be of shape (C, H, W).
  * **filter** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *None*) – Downsampling filter. It can be ‘gaussian’, ‘bilinear’ or ‘bicubic’ or a
    custom `torch.Tensor` filter. If `None`, no filtering is applied.
  * **factor** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – downsampling factor/ratio.
  * **srf** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list)) – spectral response function of the decolorize operator to produce grayscale from multispectral.
    See [`deepinv.physics.Decolorize`](https://deepinv.org/api/stubs/deepinv.physics.Decolorize.html.md#deepinv.physics.Decolorize) for parameter options. Defaults to `flat` i.e. simply average the bands.
  * **use_brovey** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, use the Brovey method Vivone *et al.*<sup>[1](#footcite-vivone2014critical)</sup>.
    to compute the pansharpening, otherwise use the conjugate gradient method.
  * **noise_color** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – noise model for the RGB image. It defaults to zero noise.
  * **noise_gray** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – noise model for the grayscale image. It defaults to zero noise.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – torch device.
  * **padding** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – options are `'valid'`, `'circular'`, `'replicate'` and `'reflect'`.
    If `padding='valid'` the blurred output is smaller than the image (no padding)
    otherwise the blurred output has the same size as the image.
  * **normalize** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, normalize the downsampling operator to have unit norm.
  * **eps** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – small value to avoid division by zero in the Brovey method.

<hr />

* **Examples:**
  Pansharpen operator applied to a random 32x32 image:
  ```pycon
  >>> from deepinv.physics import Pansharpen
  >>> import torch
  >>> x = torch.randn(1, 3, 32, 32) # Define random 32x32 color image
  >>> physics = Pansharpen(img_size=x.shape[1:], device=x.device)
  >>> x.shape
  torch.Size([1, 3, 32, 32])
  >>> y = physics(x)
  >>> y[0].shape
  torch.Size([1, 3, 8, 8])
  >>> y[1].shape
  torch.Size([1, 1, 32, 32])
  ```

<hr />

* **References:**

* <a id='footcite-vivone2014critical'>**[1]**</a> Gemine Vivone, Luciano Alparone, Jocelyn Chanussot, Mauro Dalla Mura, Andrea Garzelli, Giorgio A Licciardi, Rocco Restaino, and Lucien Wald. A critical comparison among pansharpening algorithms. *IEEE Transactions on Geoscience and Remote Sensing*, 53(5):2565–2586, 2014.

#### A_dagger(y, \*\*kwargs)

If the Brovey method is used, compute the classical Brovey solution, otherwise compute the conjugate gradient solution.

See the review paper Vivone *et al.*<sup>[1](#footcite-vivone2014critical)</sup> for more details.

* **Parameters:**
  **y** ([*deepinv.utils.TensorList*](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList)) – input tensorlist of (MS, PAN)
* **Returns:**
  Tensor of image pan-sharpening using the Brovey method.

<hr />

* **References:**

<a id="sphx-glr-backref-deepinv-physics-pansharpen"></a>

## Examples using `Pansharpen`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in physics.">  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we demonstrate remote sensing inverse problems for multispectral satellite imaging.">  <div class="sphx-glr-thumbnail-title">Remote sensing with satellite images</div>
</div>
<!-- thumbnail-parent-div-close --></div>
