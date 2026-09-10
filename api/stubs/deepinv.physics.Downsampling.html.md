# Downsampling

### *class* deepinv.physics.Downsampling(img_size=None, filter='warn', factor=2, device='cpu', padding='circular', \*\*kwargs)

Bases: [`LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)

Downsampling operator for super-resolution problems.

It is defined as

$$
y = S (h*x)
$$

where $h$ is a low-pass filter and $S$ is a subsampling operator.

* **Parameters:**
  * **filter** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *None*) – Downsampling filter. It can be `'gaussian'`, `'bilinear'`, `'bicubic'`
    , `'sinc'` or a custom `torch.Tensor` filter. If `None`, no filtering is applied. Bicubic downsampling is a sensible default if you are not sure which filter to use.
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]* *,* *None*) – optional size of the high resolution image `(C, H, W)`.
    If `tuple`, use this fixed image size.
    If `None`, override on-the-fly using input data size and `factor` (note that here, `A_adjoint` will
    only produce even img shapes).
  * **factor** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – downsampling factor
  * **padding** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – options are `'valid'`, `'circular'`, `'replicate'` and `'reflect'`.
    If `padding='valid'` the blurred output is smaller than the image (no padding)
    otherwise the blurred output has the same size as the image.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device on which the physics’ buffers will be created. If a buffer is updated via `physics.update_parameters()`, if not None, it will be automatically casted to the device of the replaced buffer, else, use the device of the provided value. To change the device of all buffers, please use `physics.to(device)`.

<hr />

* **Examples:**
  Downsampling operator with a gaussian filter:
  ```pycon
  >>> from deepinv.physics import Downsampling
  >>> x = torch.zeros((1, 1, 32, 32)) # Define black image of size 32x32
  >>> x[:, :, 16, 16] = 1 # Define one white pixel in the middle
  >>> physics = Downsampling(filter = "gaussian", img_size=(1, 32, 32), factor=2)
  >>> y = physics(x)
  >>> y[:, :, 7:10, 7:10] # Display the center of the downsampled image
  tensor([[[[0.0146, 0.0241, 0.0146],
            [0.0241, 0.0398, 0.0241],
            [0.0146, 0.0241, 0.0146]]]])
  ```

<hr />

* **Used in benchmarks:**

- [DIV2K Super Resolution 2x](https://deepinv.org/auto_benchmarks/div2k_super_resolution_2x.html.md#div2k-super-resolution-2x)

#### A(x, filter=None, factor=None, \*\*kwargs)

Applies the downsampling operator to the input image.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image.
  * **filter** (*None* *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *]*) – Filter $h$ to be applied to the input image before downsampling.
    If not `None`, it uses this filter and stores it as the current filter.
  * **factor** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – downsampling factor. If not `None`, use this factor and store it as current factor.

#### WARNING
If `factor` is passed, `filter` must also be passed as a `str` or `Tensor`, in order to update the filter to the new factor.

#### A_adjoint(y, filter=None, factor=None, \*\*kwargs)

Adjoint operator of the downsampling operator.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – downsampled image.
  * **filter** (*None* *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *]*) – Filter $h$ to be applied to the input image before downsampling.
    If not `None`, it uses this filter and stores it as the current filter.
  * **factor** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – downsampling factor. If not `None`, use this factor and store it as current factor.

#### WARNING
If `factor` is passed, `filter` must also be passed as a `str` or `Tensor`, in order to update the filter to the new factor.

#### *static* check_factor(factor)

Check new downsampling factor.

* **Parameters:**
  **factor** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – downsampling factor to be checked and cast to `int`. If [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor),
  it must be 1D and all its elements must be the same, since downsampling only supports one factor per batch.
* **Returns:**
  `int`: factor
* **Return type:**
  [int](https://docs.python.org/3.9/library/functions.html#int)

#### *static* get_filter_parameters(img_size=None, filter=None, factor=None, device='cpu')

Create a filter tensor with specified downsampling factor

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – size of the high resolution image `(C, H, W)`.
  * **filter** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *]*) – Filter name or tensor
    to be applied to the input image before downsampling.
  * **factor** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Downsampling factor to be applied to the input image.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device where the filter tensor will be created or pushed to.

#### prox_l2(z, y, gamma, use_fft=True, \*\*kwargs)

If the padding is circular, it computes the proximal operator with the closed-formula of Zhu *et al.*<sup>[1](#footcite-zhu2014fast)</sup>.

Otherwise, it computes it using the conjugate gradient algorithm which can be slow if applied many times.

<hr />

* **References:**

* <a id='footcite-zhu2014fast'>**[1]**</a> Zhiliang Zhu, Fangda Guo, Hai Yu, and Chen Chen. Fast single image super-resolution via self-example learning and sparse representation. *IEEE Transactions on Multimedia*, 16(8):2178–2190, 2014.

#### update_parameters(filter=None, factor=None, device=None, \*\*kwargs)

Updates the current filter and/or factor.

* **Parameters:**
  * **filter** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *]*) – New filter to be applied to the input image.
  * **factor** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – New downsampling factor to be applied to the input image.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – When\`\`self.filter\`\` is `None` and if `filter` is a `str`, specifies the device where the new filter will be created. When``self.filter` is `None` and `filter` is a `torch.Tensor`, the device is inferred from the provided `filter` tensor. Ignored otherwise.

<a id="sphx-glr-backref-deepinv-physics-downsampling"></a>

## Examples using `Downsampling`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Single-image super-resolution (SISR) is the inverse problem of recovering a high-resolution (HR) image x from a low-resolution (LR) observation y = \\downarrow_s(x), where \\downarrow_s denotes downsampling by factor s.">  <div class="sphx-glr-thumbnail-title">Super-resolution with SRResNet</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in physics.">  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we demonstrate remote sensing inverse problems for multispectral satellite imaging.">  <div class="sphx-glr-thumbnail-title">Remote sensing with satellite images</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Implementation of :footciteromano2017little using as plug-in denoiser the Gradient-Step Denoiser (GSPnP) :footcitehurault2021gradient which provides an explicit prior.">  <div class="sphx-glr-thumbnail-title">Regularization by Denoising (RED) for Super-Resolution.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use vanilla unfolded Plug-and-Play. The DnCNN denoiser and the algorithm parameters (stepsize, regularization parameters) are trained jointly. For simplicity, we show how to train the algorithm on a  small dataset. For optimal results, use a larger dataset.">  <div class="sphx-glr-thumbnail-title">Vanilla Unfolded algorithm for super-resolution</div>
</div>
<!-- thumbnail-parent-div-close --></div>
