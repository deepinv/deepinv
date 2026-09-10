# CompressiveSpectralImaging

### *class* deepinv.physics.CompressiveSpectralImaging(img_size, mask=None, mode='ss', shear_dir='h', device='cpu', rng=None, \*\*kwargs)

Bases: [`LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)

Compressive Hyperspectral Imaging operator.

Coded-aperture snapshot spectral imaging (CASSI) operator, which is a popular
approach for hyperspectral imaging.

The CASSI operator performs a combination of masking (“coded aperture”), shearing,
and flattening in the channel dim.
We provide two specific popular CASSI models: single-disperser (i.e. only spatial encoding)
and spatial-spectral encoding:

$$
y =
\begin{cases}
    \Sigma_{c=1}^{C} S^{-1} MSx & \text{if mode='spatial-spectral'} \\
    \Sigma_{c=1}^{C} SMx & \text{if mode='single-disperser'}
\end{cases}
$$

where $M$ is a binary mask (the “coded aperture”), $S$ is a pixel shear in the 2D
channel-height of channel-width plane and $C$ is number of channels.
Note that the output size of the single-disperser mode has the `H` or `W` dim extended by `C-1` pixels.

For more details see e.g. the paper Choi *et al.*<sup>[1](#footcite-choi2017high)</sup>.

The implementation is a type of linear physics as it is not completely decomposable due to edge effects and different scaling.

<hr />

* **Examples:**
  ```pycon
  >>> from deepinv.physics import CompressiveSpectralImaging
  >>> physics = CompressiveSpectralImaging(img_size=(7, 32, 32))
  >>> x = torch.rand(1, 7, 32, 32) # 7-band image
  >>> y = physics(x)
  >>> y.shape
  torch.Size([1, 1, 32, 32])
  ```
* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – image size, must be of form (C,H,W) where C is number of bands.
  * **mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – coded-aperture mask. If `None`, generate random mask using
    [`deepinv.physics.generator.BernoulliSplittingMaskGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.BernoulliSplittingMaskGenerator.html.md#deepinv.physics.generator.BernoulliSplittingMaskGenerator) with masking ratio
    of 0.5, if mask is `float`, sets mask ratio to this. If `Tensor`, set mask to this,
    must be of shape `(B,C,H,W)`.
  * **mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – ‘sd’ for SD-CASSI i.e. single disperser (only spatial encoding) or
    ‘ss’ for SS-CASSI i.e. spatial-spectral encoding. Defaults to ‘ss’. See above for details.
  * **shear_dir** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – “h” for shear in H-C plane or “w” for shear in W-C plane where C is channel dim, defaults to “h”
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – torch device, only used if `mask` is `None` or `float`
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – torch random generator, only used if `mask` is `None` or `float`

<hr />

* **References:**

* <a id='footcite-choi2017high'>**[1]**</a> Inchang Choi, Daniel S. Jeon, Giljoo Nam, Diego Gutierrez, and Min H. Kim. High-quality hyperspectral reconstruction using a spectral prior. *ACM Trans. Graph.*, 2017.

#### A(x, mask=None, \*\*kwargs)

Applies the CASSI forward operator.

If a mask is provided, it updates the class attribute `mask` on the fly.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image
  * **mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – CASSI mask
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) output measurements
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### A_adjoint(y, mask=None, \*\*kwargs)

Applies the CASSI adjoint operator.

If a mask is provided, it updates the class attribute `mask` on the fly.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input measurements
  * **mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – CASSI mask
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) output image
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### crop(x)

Crop image on bottom or on right.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input padded image

#### flatten(x)

Average over channel dimension

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image of shape B,C,H,W

#### pad(x)

Pad image on bottom or on right.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image

#### shear(x, un=False)

Efficient pixel shear in channel-spatial plane

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image of shape (B,C,H,W)
  * **un** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, unshear in opposite direction.

#### unflatten(y)

Repeat over channel dimension

* **Parameters:**
  **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image of shape B,C,H,W

<a id="sphx-glr-backref-deepinv-physics-compressivespectralimaging"></a>

## Examples using `CompressiveSpectralImaging`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In this example we demonstrate remote sensing inverse problems for multispectral satellite imaging.">  <div class="sphx-glr-thumbnail-title">Remote sensing with satellite images</div>
</div>
<!-- thumbnail-parent-div-close --></div>
