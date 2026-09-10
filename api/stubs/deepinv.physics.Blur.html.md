# Blur

### *class* deepinv.physics.Blur(filter=None, padding='valid', use_fft=False, device=torch.device('cpu'), \*\*kwargs)

Bases: [`LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)

Blur operator.

This forward operator performs

$$
y = w*x
$$

where $*$ denotes convolution and $w$ is a filter.

* **Parameters:**
  * **filter** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Tensor of size (b, 1, h, w) or (b, c, h, w) in 2D; (b, 1, d, h, w) or (b, c, d, h, w) in 3D,
    containing the blur filter, e.g., [`deepinv.physics.functional.gaussian_blur()`](https://deepinv.org/api/stubs/deepinv.physics.functional.gaussian_blur.html.md#deepinv.physics.functional.gaussian_blur). If `None`, a filter must be passed to the physics before calling it . (default is `None`)
  * **padding** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – options are `'valid'`, `'circular'`, `'replicate'` and `'reflect'`.
    If `padding='valid'` the blurred output is smaller than the image (no padding)
    otherwise the blurred output has the same size as the image. (default is `'valid'`).
    Only `padding='valid'` and  `padding = 'circular'` are implemented in 3D.
  * **use_fft** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to use FFT-based convolutions. If `True`, it uses FFT-based convolutions which can be faster for large kernels.
    If `False`, it uses the standard convolution functions from `torch.nn.functional`.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device on which the physics’ buffers will be created. When a buffer is updated via `physics.update_parameters()`, if the current buffer is None, use the device of the incoming value, else, the incoming value is casted to the device of the current buffer. To change the device of all buffers, please use `physics.to(device)`.

#### NOTE
This class makes it possible to change the filter at runtime by passing a new filter to the forward method, e.g.,
`y = physics(x, w)`. The new filter $w$ is stored as the current filter.

#### NOTE
This class performs a true convolution, not a cross-correlation as the standard convolution functions in `torch.nn.functional`.
It is recommended to use `use_fft=True` for large filters, and `use_fft=False` for small filters, since FFT-based convolutions can be faster for large kernels but slower for small kernels.

<hr />

* **Examples:**
  Blur operator with a basic averaging filter applied to a 16x16 black image with
  a single white pixel in the center:
  ```pycon
  >>> from deepinv.physics import Blur
  >>> x = torch.zeros((1, 1, 16, 16)) # Define black image of size 16x16
  >>> x[:, :, 8, 8] = 1 # Define one white pixel in the middle
  >>> w = torch.ones((1, 1, 2, 2)) / 4 # Basic 2x2 averaging filter
  >>> physics = Blur(filter=w)
  >>> y = physics(x)
  >>> y[:, :, 7:10, 7:10] # Display the center of the blurred image
  tensor([[[[0.2500, 0.2500, 0.0000],
            [0.2500, 0.2500, 0.0000],
            [0.0000, 0.0000, 0.0000]]]])
  ```

<hr />

* **Used in benchmarks:**

- [DIV2K Gaussian Deblurring](https://deepinv.org/auto_benchmarks/div2k_gaussian_deblurring.html.md#div2k-gaussian-deblurring)

#### A(x, filter=None, \*\*kwargs)

Applies the filter to the input image.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image.
  * **filter** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Filter $w$ to be applied to the input image.
    If not `None`, it uses this filter instead of the one defined in the class, and
    the provided filter is stored as the current filter.
* **Raises:**
  [**ValueError**](https://docs.python.org/3.9/library/exceptions.html#ValueError) – if the input tensor does not have 4 or 5 dimensions.

#### A_adjoint(y, filter=None, \*\*kwargs)

Adjoint operator of the blur operator.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – blurred image.
  * **filter** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Filter $w$ to be applied to the input image.
    If not `None`, it uses this filter instead of the one defined in the class, and
    the provided filter is stored as the current filter.
* **Raises:**
  [**ValueError**](https://docs.python.org/3.9/library/exceptions.html#ValueError) – if the input tensor does not have 4 or 5 dimensions.

<a id="sphx-glr-backref-deepinv-physics-blur"></a>

## Examples using `Blur`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train various networks using adversarial training for deblurring problems. We demonstrate running training and inference using a conditional GAN (i.e. DeblurGAN), CSGM, AmbientGAN and UAIR implemented in the library, and how to simply train your own GAN by using deepinv.training.AdversarialTrainer. These examples can also be easily extended to train more complicated GANs such as CycleGAN.">  <div class="sphx-glr-thumbnail-title">Imaging inverse problems with adversarial networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.physics.Physics together with automatic differentiation to estimate unknown parameters of your forward operator.">  <div class="sphx-glr-thumbnail-title">Calibrating physics operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Many large-scale imaging problems involve operators that can be naturally decomposed as a stack of multiple sub-operators:">  <div class="sphx-glr-thumbnail-title">Distributed Physics Operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Many large-scale imaging problems involve operators that can be naturally decomposed as a stack of multiple sub-operators:">  <div class="sphx-glr-thumbnail-title">Distributed Plug-and-Play (PnP) Reconstruction</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In many large-scale imaging problems, the size of the image/volume to reconstruct is very large, making it impossible to train reconstruction networks (in this example, unfolded networks) with a single GPU. The deepinv.distributed framework enables training a model on multiple GPUs, by carefully parallelizing the data fidelity and denoising steps inside the network.">  <div class="sphx-glr-thumbnail-title">Distributed Training of Unfolded Networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of 2D blur operators in DeepInverse. In particular, we show how to use DiffractionBlurs (Fresnel diffraction), motion blurs and space varying blurs.">  <div class="sphx-glr-thumbnail-title">Tour of blur operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Real-world blurry images have decorrelated opposite boundaries, unlike images synthetically blurred using circular filters. This makes the use of spectral deconvolution methods (inverse filtering, Wiener filtering) impractical and prone to ringing artifacts. Liu-Jia padding &lt;deepinv.physics.functional.liu_jia_pad&gt; :footciteliu2008reducing is a pre-processing step that pads the input image to make it have smooth circular boundaries, while preserving the original spectral content as much as possible.">  <div class="sphx-glr-thumbnail-title">Spectral Methods for Non-Circular Deblurring with Liu-Jia Padding</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of 3D blur operators in the library. In particular, we show how to use Diffraction Blurs (Fresnel diffraction) to simulate fluorescence microscopes.">  <div class="sphx-glr-thumbnail-title">3D diffraction PSF</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in physics.">  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows the use of the deepinv.physics.SpatialUnwrapping forward model and the deepinv.optim.ItohFidelity for unwrapping problems, which occur in modulo imaging, interferometry SAR and other imaging applications. It shows how to generate a wrapped phase image, apply blur and noise, and reconstruct the original phase using both DCT inversion and ADMM optimization.">  <div class="sphx-glr-thumbnail-title">Spatial unwrapping and modulo imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Some unfolded architectures rely on a least-squares solver &lt;deepinv.optim.linear.least_squares&gt; to compute the proximal step w.r.t. the data-fidelity term (e.g., deepinv.optim.optim_iterators.ADMMIteration or deepinv.optim.optim_iterators.HQSIteration):  ">  <div class="sphx-glr-thumbnail-title">Reducing the memory and computational complexity of unfolded network training</div>
</div>
<!-- thumbnail-parent-div-close --></div>
