# HyperSpectralUnmixing

### *class* deepinv.physics.HyperSpectralUnmixing(M=None, E=15, C=64, device='cpu', \*\*kwargs)

Bases: [`LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)

Hyperspectral Unmixing operator.

Hyperspectral Unmixing (HU) analyzes data captured by a hyperspectral sensor,
which captures light across a high number of bands (vs. a regular camera which captures light in three bands (RGB)).
As an analogy, imagine the problem of unmixing paint in a pixel. The paint at a pixel is likely a mixture of various basic colors.
Unmixing separates the overall color (spectrum) of the pixel into the amounts (abundances) of each base color (endmember) used to create the mixture.

Please see the survey Bioucas-Dias *et al.*<sup>[1](#footcite-bioucas2012hyperspectral)</sup> for details.

Hyperspectral mixing is modelled using a Linear Mixing Model (LMM).

$$
\mathbf{y}= \mathbf{M}\cdot\mathbf{x} + \mathbf{\epsilon}
$$

where $\mathbf{y}$ is the resulting image of shape `(B, C, H, W)`. LMM assumes each pixel $\mathbf{y}_i$’s spectrum
is a linear combination of the spectra of pure materials (endmembers) in the scene, represented by a matrix $\mathbf{M}$ of shape $(E,C)$,
weighted by their fractional abundances in the pixel $x_i$ of shape $(B, E, H, W)$ where $\epsilon$ represents measurement noise.

The HU inverse problem aims to recover the abundance vector $\mathbf{x}$ for each pixel in the image, essentially separating the mixed signals.
If the endmember matrix $\mathbf{M}$ is unknown, then this must be estimated too.

* **Parameters:**
  * **M** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Matrix of endmembers of shape $(E,C)$. Overrides `E` and `C` parameters.
    If `None`, then a random normalized matrix is simulated from a uniform distribution. Default `None`.
  * **E** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of endmembers (e.g. number of materials). Ignored if `M` is set.  Default: `15`.
  * **C** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of hyperspectral bands. Ignored if `M` is set. Default: `64`.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – torch device, cpu or gpu.

<hr />

* **Examples:**
  Hyperspectral mixing of a 128x128 image with 64 channels and 15 endmembers:
  ```pycon
  >>> from deepinv.physics import HyperSpectralUnmixing
  >>> E, C = 15, 64 # n. endmembers and n. channels
  >>> B, H, W = 4, 128, 128 # batch size and image size
  >>> physics = HyperSpectralUnmixing(E=E, C=C)
  >>> x = torch.rand((B, E, H, W)) # sample set of abundances
  >>> y = physics(x) # resulting mixed image
  >>> print(x.shape, y.shape, physics.M.shape)
  torch.Size([4, 15, 128, 128]) torch.Size([4, 64, 128, 128]) torch.Size([15, 64])
  ```

<hr />

* **References:**

* <a id='footcite-bioucas2012hyperspectral'>**[1]**</a> José M Bioucas-Dias, Antonio Plaza, Nicolas Dobigeon, Mario Parente, Qian Du, Paul Gader, and Jocelyn Chanussot. Hyperspectral unmixing overview: geometrical, statistical, and sparse regression-based approaches. *IEEE journal of selected topics in applied earth observations and remote sensing*, 5(2):354–379, 2012.

#### A(x, M=None, \*\*kwargs)

Applies the endmembers matrix to the input abundances x.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input abundances.
  * **M** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – optional new endmembers matrix $\mathbf{M}$ to be applied to the input abundances.

#### A_adjoint(y, M=None, \*\*kwargs)

Applies the transpose endmember matrix to the image y.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image.
  * **M** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – optional new endmembers matrix $\mathbf{M}$ to be applied to the input image.

#### A_dagger(y, M=None, \*\*kwargs)

Applies the pseudoinverse endmember matrix to the image y.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image.
  * **M** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – optional new endmembers matrix $\mathbf{M}$ to be applied to the input image.

#### update_parameters(M=None, \*\*kwargs)

Updates the current endmembers matrix.

* **Parameters:**
  **M** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – New endmembers matrix to be applied to the input abundances.

<a id="sphx-glr-backref-deepinv-physics-hyperspectralunmixing"></a>

## Examples using `HyperSpectralUnmixing`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In this example we demonstrate remote sensing inverse problems for multispectral satellite imaging.">  <div class="sphx-glr-thumbnail-title">Remote sensing with satellite images</div>
</div>
<!-- thumbnail-parent-div-close --></div>
