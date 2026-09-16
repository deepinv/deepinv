# ComplexDenoiserWrapper

### *class* deepinv.models.ComplexDenoiserWrapper(denoiser, mode='real_imag', \*args, \*\*kwargs)

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

Complex-valued wrapper for a real-valued denoiser $\denoisername(\cdot, \sigma)$.

This class lifts any real-valued denoiser to the complex domain by applying it separately to a chosen *pair* of real representations of the complex input and recombining the outputs.

Let the input be $x \in \mathbb{C}^{B\times C\times H\times W}$ and a noise level $\sigma > 0$ (scalar or batch of size $B$).
The underlying denoiser (given by `denoiser`) $\denoisername$ acts on real tensors only. Two processing modes are supported:

<hr />

1. `'real_imag'` mode

We decompose

$$
x = x_{\mathrm{real}} + i x_{\mathrm{imag}}.
$$

The denoiser is applied on the real and imaginary parts (same $\sigma$ broadcast across both halves).
The complex reconstruction is

$$
\hat x = \denoisername(x_{\mathrm{real}}, \sigma) + i \, \denoisername(x_{\mathrm{imag}}, \sigma).
$$

If the provided input tensor is real (i.e. `torch.is_complex(x)` is `False`), it is interpreted as $x_{\mathrm{real}}$ with $x_{\mathrm{imag}}=0$ and the output is returned as
$\denoisername(x_{\mathrm{real}},\sigma) + i 0$ (complex dtype ensured).

<hr />

1. `'abs_angle'` mode

We use the polar decomposition

$$
x = m \exp(i\phi), \qquad m = |x|,\; \phi = \mathrm{arg}(x) \in (-\pi,\pi].
$$

The denoiser is applied on the magnitude and phase parts (same $\sigma$ broadcast across both halves).
The reconstructed complex output is

$$
\hat x = \denoisername(m, \sigma) \exp \big(i\, \denoisername(\phi, \sigma)\big).
$$

Note that the phase estimate $\denoisername(\phi,\sigma)$ is **clipped** back to $(-\pi,\pi]$.

#### NOTE
This wrapper can only process complex inputs that are compatible with the underlying real-valued denoiser.
For example, if the wrapped `denoiser` supports only single-channel (grayscale) real images, then the
corresponding complex input must also be single-channel.

<hr />

* **Examples:**
  ```pycon
  >>> import deepinv as dinv
  >>> import torch
  >>> from deepinv.models import ComplexDenoiserWrapper, DRUNet
  >>> denoiser = DRUNet()
  >>> complex_denoiser = ComplexDenoiserWrapper(denoiser, mode="real_imag")
  >>> y = torch.randn(2, 3, 32, 32, dtype=torch.complex64)  # complex input
  >>> sigma = 0.1
  >>> with torch.no_grad():
  ...     denoised = complex_denoiser(y, sigma)
  >>> print(denoised.dtype)
  torch.complex64
  ```
* **Parameters:**
  * **denoiser** ([*deepinv.models.Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)) – Real-valued denoiser $\denoisername$ to wrap.
  * **mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Either `'real_imag'` or `'abs_angle'`. Default `'real_imag'`.
* **Raises:**
  [**ValueError**](https://docs.python.org/3.9/library/exceptions.html#ValueError) – If an unsupported mode string is provided.
* **Returns:**
  Complex denoised output $\hat x$ with same spatial shape as the input.

#### forward(x, sigma)

Applies the complex-valued denoiser. If a real tensor is provided, it is treated as a complex tensor with zero imaginary part.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – complex-valued input images.
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *or* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – noise level.
* **Returns:**
  Denoised images, with the same shape as the input and will always be in complex dtype.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-models-complexdenoiserwrapper"></a>

## Examples using `ComplexDenoiserWrapper`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to create a random phase retrieval operator and generate phaseless measurements from a given image. The example showcases 4 different reconstruction methods to recover the image from the phaseless measurements:">  <div class="sphx-glr-thumbnail-title">Random phase retrieval and reconstruction methods.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
