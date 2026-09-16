# SpaceVaryingBlur

### *class* deepinv.physics.SpaceVaryingBlur(filters=None, multipliers=None, padding='valid', use_fft=False, mask_first=True, device='cpu', \*\*kwargs)

Bases: [`LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)

Space varying blur via product-convolution.

If `mask_first=True` (default), this linear operator performs

$$
y = \sum_{k=1}^K h_k \star (w_k \odot x)
$$

whereas if `mask_first=False`, the multipliers are applied after the convolutions, i.e.

$$
y = \sum_{k=1}^K w_k \odot (h_k \star x)
$$

where $\star$ is a convolution, $\odot$ is a Hadamard product,  $w_k$ are multipliers $h_k$ are filters.

#### TIP
A comparison between both models can be found in Denis *et al.*<sup>[1](#footcite-denis2011fast)</sup>.

* **Parameters:**
  * **filters** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Filters $h_k$. Tensor of size `(B, C, K, h, w)` where
    `B` is the batch size, `C` the number of channels, `K` the number of filters, `h` and `w` the filter height and width which
    should be smaller or equal than the image $x$ height and width respectively.
  * **multipliers** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Multipliers $w_k$. Tensor of size `(B, C, K, H, W)` where
    `B` is the batch size, `C` the number of channels, `K` the number of multipliers, `H` and `W` the image $x$ height and width.
    If `mask_first=False` and `padding='valid'`, the spatial size of the multipliers should match the size of the
    convolution output instead, i.e. `(B, C, K, H-h+1, W-w+1)`.
  * **padding** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – options = `'valid'`, `'circular'`, `'replicate'`, `'reflect'`.
    If `padding = 'valid'` the blurred output is smaller than the image (no padding),
    otherwise the blurred output has the same size as the image.
  * **use_fft** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to use FFT-based convolutions. If `True`, it uses FFT-based convolutions which can be faster for large kernels.
  * **mask_first** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether the multipliers $w_k$ are applied before (`True`, default) or after
    (`False`) the convolutions.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device on which the physics’ buffers will be created. If a buffer is updated via `physics.update_parameters()`, if not None, it will be automatically casted to the device of the replaced buffer, else, use the device of the provided value. To change the device of all buffers, please use `physics.to(device)`.

<hr />

* **Examples:**
  We show how to instantiate a spatially varying blur operator.
  ```pycon
  >>> from deepinv.physics.generator import DiffractionBlurGenerator, ProductConvolutionBlurGenerator
  >>> from deepinv.physics.blur import SpaceVaryingBlur
  >>> from deepinv.utils.plotting import plot
  >>> psf_size = 32
  >>> img_size = (256, 256)
  >>> delta = 16
  >>> psf_generator = DiffractionBlurGenerator((psf_size, psf_size))
  >>> pc_generator = ProductConvolutionBlurGenerator(psf_generator=psf_generator, img_size=img_size)
  >>> params_pc = pc_generator.step(1)
  >>> physics = SpaceVaryingBlur(**params_pc)
  >>> dirac_comb = torch.zeros(img_size).unsqueeze(0).unsqueeze(0)
  >>> dirac_comb[0,0,::delta,::delta] = 1
  >>> psf_grid = physics(dirac_comb)
  >>> plot(psf_grid, titles="Space varying impulse responses")
  ```

<hr />

* **References:**

* <a id='footcite-denis2011fast'>**[1]**</a> L. Denis, E. Thiebaut, and F. Soulez. Fast model of space-variant blurring and its application to deconvolution in astronomy. In *2011 18th IEEE International Conference on Image Processing*, 2817–2820. Brussels, Belgium, September 2011. IEEE. [doi:10.1109/ICIP.2011.6116257](https://doi.org/10.1109/ICIP.2011.6116257).

#### A(x, filters=None, multipliers=None, padding=None, \*\*kwargs)

Applies the space varying blur operator to the input image.

It can receive new parameters  $w_k$, $h_k$ and padding to be used in the forward operator, and stored
as the current parameters.

* **Parameters:**
  * **filters** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Filters $h_k$. Tensor of size (b, c, K, h, w). $b \in \{1, B\}$ and $c \in \{1, C\}$, $h\leq H$ and $w\leq W$.
  * **multipliers** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Multipliers $w_k$. Tensor of size (b, c, K, H, W). $b \in \{1, B\}$ and $c \in \{1, C\}$
  * **padding** – options = `'valid'`, `'circular'`, `'replicate'`, `'reflect'`.
    If `padding = 'valid'` the blurred output is smaller than the image (no padding),
    otherwise the blurred output has the same size as the image.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – cpu or cuda

#### A_adjoint(y, filters=None, multipliers=None, padding=None, \*\*kwargs)

Applies the adjoint operator.

It can receive new parameters $w_k$, $h_k$ and padding to be used in the forward operator, and stored
as the current parameters.

* **Parameters:**
  * **filters** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Filters $h_k$. Tensor of size (b, c, K, h, w). $b \in \{1, B\}$ and $c \in \{1, C\}$, $h\leq H$ and $w\leq W$.
  * **multipliers** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Multipliers $w_k$. Tensor of size (b, c, K, H, W). $b \in \{1, B\}$ and $c \in \{1, C\}$
  * **padding** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – options = `'valid'`, `'circular'`, `'replicate'`, `'reflect'`.
    If `padding = 'valid'` the blurred output is smaller than the image (no padding),
    otherwise the blurred output has the same size as the image.

#### update_parameters(filters=None, multipliers=None, padding=None, \*\*kwargs)

Updates the current parameters.

* **Parameters:**
  * **filters** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Filters $h_k$. Tensor of size (b, c, K, h, w). $b \in \{1, B\}$ and $c \in \{1, C\}$, $h\leq H$ and $w\leq W$.
  * **multipliers** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Multipliers $w_k$. Tensor of size (b, c, K, H, W). $b \in \{1, B\}$ and $c \in \{1, C\}$
  * **padding** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – options = `'valid'`, `'circular'`, `'replicate'`, `'reflect'`.

<a id="sphx-glr-backref-deepinv-physics-spacevaryingblur"></a>

## Examples using `SpaceVaryingBlur`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates blind image deblurring using the pretrained kernel estimation network from the paper :footcitecarbajal2023blind. The network estimates spatially-varying blur kernels from a blurred image, which are then used in a space-varying blur physics model to reconstruct the sharp image using a non-blind deblurring algorithm.">  <div class="sphx-glr-thumbnail-title">Blind deblurring with kernel estimation network</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of 2D blur operators in DeepInverse. In particular, we show how to use DiffractionBlurs (Fresnel diffraction), motion blurs and space varying blurs.">  <div class="sphx-glr-thumbnail-title">Tour of blur operators</div>
</div>
<!-- thumbnail-parent-div-close --></div>
