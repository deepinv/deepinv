# RadioInterferometry

### *class* deepinv.physics.RadioInterferometry(img_size, samples_loc, dataWeight=None, k_oversampling=2, interp_points=7, real_projection=True, device='cpu', \*\*kwargs)

Bases: [`LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)

Radio Interferometry measurement operator.

The operator handles ungridded measurements using the non-uniform FFT (NUFFT), which is based in Kaiser-Bessel
kernel interpolation. This particular implementation relies on the [torchkbnufft](https://github.com/mmuckley/torchkbnufft) package.

The forward operator is defined as $A:x \mapsto y$,
where $A$ can be decomposed as $A = GFZ \in \mathbb{C}^{m \times n}$.
There, $G \in \mathbb{C}^{m \times d}$ is a sparse interpolation matrix,
encoding the non-uniform Fourier transform,
$F \in \mathbb{C}^{d\times d}$ is the 2D Discrete orthonormal Fourier Transform,
$Z \in \mathbb{R}^{d\times n}$ is a zero-padding operator,
incorporating the correction for the convolution performed through the operator $G$.

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – Size of the target image, e.g., (H, W).
  * **samples_loc** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Normalized sampling locations in the Fourier domain (Size: N x 2).
  * **dataWeight** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Data weighting for the measurements (Size: N). Default is `torch.tensor([1.0])` (i.e. no weighting).
  * **interp_points** (*Union* *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* *Sequence* *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]* *]*) – Number of neighbors to use for interpolation in each dimension. Default is `7`.
  * **k_oversampling** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Oversampling of the k space grid, should be between `1.25` and `2`. Default is `2`.
  * **real_projection** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Apply real projection after the adjoint NUFFT. Default is `True`.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – Device where the operator is computed. Default is `cpu`.

#### WARNING
If the `real_projection` parameter is set to `False`, the output of the adjoint will have a complex type rather than a real typed.

#### NOTE
This class requires the `torchkbnufft` package to be installed. Install with `pip install torchkbnufft`.

#### A(x, \*\*kwargs)

Applies the weighted NUFFT operator to the input image.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) containing the measurements
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### A_adjoint(y, \*\*kwargs)

Applies the adjoint of the weighted NUFFT operator.

* **Parameters:**
  **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input measurements
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) containing the reconstructed image
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-physics-radiointerferometry"></a>

## Examples using `RadioInterferometry`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In this example, we investigate a simple 2D Radio Interferometry (RI) imaging task with deepinverse. The following example and data are taken from :footciteaghabiglou2024r2d2. If you are interested in RI imaging problem and would like to see more examples or try the state-of-the-art algorithms, please check BASPLib.">  <div class="sphx-glr-thumbnail-title">Radio interferometric imaging with deepinverse</div>
</div>
<!-- thumbnail-parent-div-close --></div>
