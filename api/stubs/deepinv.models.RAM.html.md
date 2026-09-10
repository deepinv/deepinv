# RAM

### *class* deepinv.models.RAM(in_channels=(1, 2, 3), device=None, pretrained=True)

Bases: [`Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor), [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

Reconstruct Anything Model (RAM) foundation model.

RAM Terris *et al.*<sup>[1](#footcite-terris2025reconstruct)</sup> is a convolutional neural network model that has been trained to work on a large variety
of linear image reconstruction tasks and datasets (deblurring, inpainting, denoising, tomography, MRI, etc.).

See [Inference and fine-tune a foundation model](https://deepinv.org/auto_examples/models/demo_foundation_model.html.md#sphx-glr-auto-examples-models-demo-foundation-model-py) for examples on the performance of RAM and how to fine-tune the
foundation model on a specific problem and dataset.

The model works both as a reconstructor or denoiser:

* Reconstructor: RAM takes a [physics operator](https://deepinv.org/user_guide/physics/physics.html.md#physics) `model(y, physics)` with an optional noise model defined in the physics
* Denoiser: RAM takes optional Gaussian and/or Poisson noise levels (optionally set to 0) `model(y, sigma=sigma, gamma=gamma)`

#### NOTE
The physics operator should be normalized (i.e. have unit norm) for best results.
Use [`physics.compute_norm()`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics.compute_norm) to check this.

#### TIP
This model can handle non-uniform `sigma` and `gain` maps, which can be of size `(batch_size, 1, height, width)`.

* **Parameters:**
  * **in_channels** (*Sequence*) – Number of input channels. If a list is provided, the model will have separate heads for each channel.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device to which the model should be moved. If None, the model will be created on the default device.
  * **pretrained** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – If `True`, the model will be initialized with pretrained weights. If `str`, load from file.

<hr />

* **Examples:**
  ```pycon
  >>> import deepinv as dinv
  >>> x = dinv.utils.load_example("butterfly.png")
  >>> physics = dinv.physics.Downsampling(filter="bicubic", noise_model=dinv.physics.GaussianNoise(0.01))
  >>> y = physics(x)
  >>> model = dinv.models.RAM()
  >>> x_hat = model(y, physics) # run model
  >>> dinv.metric.PSNR()(x_hat, x) > 29.75
  tensor([True])
  ```

<hr />

* **References:**

* <a id='footcite-terris2025reconstruct'>**[1]**</a> Matthieu Terris, Samuel Hurault, Maxime Song, and Julián Tachella. Reconstruct anything model: a lightweight foundation model for computational imaging. *arXiv preprint arXiv:2503.08915*, 2025.

#### base_conditioning(x, sigma, gain)

Stacks the sigma and gain value as additional channel dimensions to the input tensor.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input tensor
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Gaussian noise level or noise level map
  * **gain** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Poisson noise gain or Poisson noise map
* **Return torch.Tensor:**
  Input tensor with additional channels for sigma and gain
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### constant2map(value, x)

Converts a constant value to a map of the same size as the input tensor x.

* **Parameters:**
  * **value** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – constant value
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor
* **Return torch.Tensor:**
  a tensor of size (B, 1, W, H) containing constant maps of shapes (W, H) for each value in the batch.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### forward(y, physics=None, sigma=None, gain=None, img_size=None)

Reconstructs a signal estimate from measurements y

#### NOTE
The noise levels `sigma` and `gain` can be kept as `None` if a noise model with [`GaussianNoise`](https://deepinv.org/api/stubs/deepinv.physics.GaussianNoise.html.md#deepinv.physics.GaussianNoise),
[`PoissonNoise`](https://deepinv.org/api/stubs/deepinv.physics.PoissonNoise.html.md#deepinv.physics.PoissonNoise) or [`PoissonGaussianNoise`](https://deepinv.org/api/stubs/deepinv.physics.PoissonGaussianNoise.html.md#deepinv.physics.PoissonGaussianNoise)
is specified in the physics. If both are provided, the `sigma` and `gain` values provided to the model will be used.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – forward operator
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Gaussian noise level or noise level map
  * **gain** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Poisson noise gain or Poisson noise map
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – (optional) size of the image to reconstruct. If None, will be inferred automatically from the physics.
* **Returns:**
  torch.Tensor: reconstructed signal estimate
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### forward_unet(x0, sigma=None, gain=None, physics=None, y=None)

Forward pass of the UNet model.

* **Parameters:**
  * **x0** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – init image
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Gaussian noise level or noise level map
  * **gain** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Poisson noise gain or Poisson noise map
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – physics measurement operator
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements

#### get_pad(img_size)

Get padding amount for model input.

* **Parameters:**
  **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – model input image shape.
* **Return tuple[int, int, int]:**
  padding amounts for channel dim and spatial dims.
* **Return type:**
  [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[int](https://docs.python.org/3.9/library/functions.html#int), [int](https://docs.python.org/3.9/library/functions.html#int), [int](https://docs.python.org/3.9/library/functions.html#int)]

#### obtain_sigma_gain(physics, sigma, gain, rescale_val, device='cpu')

Defines the sigma and gain values to be used in the model.

If a noise model is specified in the physics, the sigma and gain values will be taken from the noise model.
Else, the sigma and gain values will be set to the thresholds defined in the model (if not provided).

* **Parameters:**
  * **physics** ([*deepinv.physics.LinearPhysics*](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)) – Physics operator
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Gaussian noise level. If None, will be set to the threshold.
  * **gain** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Poisson noise level. If None, will be set to the threshold.
  * **rescale_val** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Rescale value to apply to the sigma and gain values.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – Device to which the sigma and gain values should be moved.

#### realign_input(x, physics, y, sigma)

Realign the input x based on the measurements y and the physics model.
Applies the proximity operator of the L2 norm with respect to the physics model.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input tensor
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Physics model
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurements
* **Return torch.Tensor:**
  Realigned input tensor
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-models-ram"></a>

## Examples using `RAM`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use DeepInverse with your own dataset.">  <div class="sphx-glr-thumbnail-title">Bring your own dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using a pretrained model in one line.">  <div class="sphx-glr-thumbnail-title">Use a pretrained model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to get started with DeepInverse in under 5 minutes.">  <div class="sphx-glr-thumbnail-title">5 minute quickstart tutorial</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates blind image deblurring using the pretrained kernel estimation network from the paper :footcitecarbajal2023blind. The network estimates spatially-varying blur kernels from a blurred image, which are then used in a space-varying blur physics model to reconstruct the sharp image using a non-blind deblurring algorithm.">  <div class="sphx-glr-thumbnail-title">Blind deblurring with kernel estimation network</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use Spyrit linear models and measurements with DeepInverse. Here we use the HadamSplit2d linear model from Spyrit.">  <div class="sphx-glr-thumbnail-title">Single-pixel imaging with Spyrit</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to denoise low-intensity STED fluorescence microscopy images of live-cell mitochondria using the pretrained foundation model deepinv.models.RAM. We load real Abberior STED microscopy data from :footciteosunavargas2025denoising, process it in batches, and visualize the results both with deepinv.utils.plot and with the interactive 3D viewer deepinv.utils.plot_napari.">  <div class="sphx-glr-thumbnail-title">Low-intensity STED fluorescence microscopy denoising</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In blind inverse problems, some parameters of the physics are unknown at test time. Running non-blind models requires knowing these parameters, which are often hard to estimate. For example, a denoiser needs the noise level \\sigma, a deblurring model needs the blur kernel.">  <div class="sphx-glr-thumbnail-title">Blind inverse problems with no reference metrics</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model :footciteterris2025reconstruct to solve inverse problems.">  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a full-resolution multispectral image from raw snapshot mosaiced measurements using various reconstruction algorithms.">  <div class="sphx-glr-thumbnail-title">Multispectral demosaicing from raw sensor data</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Equivariant splitting consists in minimizing a self-supervised loss to train a reconstruction model using measurement data only :footcitesechaud26Equivariant.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised (blind) denoising of a low-field MRI scan without ground truth data.">  <div class="sphx-glr-thumbnail-title">Low-field MRI denoising without ground truth</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate scan-specific self-supervised learning, that is, learning to reconstruct MRI scans from a single accelerated sample without ground truth.">  <div class="sphx-glr-thumbnail-title">Scan-specific zero-shot SSDU for MRI</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the use of our deepinv.transform module for use in solving imaging problems. These can be used for:">  <div class="sphx-glr-thumbnail-title">Image transforms for equivariance & augmentations</div>
</div>
<!-- thumbnail-parent-div-close --></div>
