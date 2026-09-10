# MultiCoilMRI

### *class* deepinv.physics.MultiCoilMRI(mask=None, coil_maps=None, img_size=(320, 320), three_d=False, device=torch.device('cpu'), \*\*kwargs)

Bases: [`MRIMixin`](https://deepinv.org/api/stubs/deepinv.utils.MRIMixin.html.md#deepinv.utils.MRIMixin), [`LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)

Multi-coil 2D or 3D MRI operator.

The linear operator operates in 2D slices or 3D volumes and is defined as:

$$
y_n = \text{diag}(p) F \text{diag}(s_n) x
$$

for $n=1,\dots,N$ coils, where $y_n$ are the measurements from the cth coil, $\text{diag}(p)$ is the acceleration mask, $F$ is the Fourier transform and $\text{diag}(s_n)$ is the nth coil sensitivity.

The data `x` should be of shape (B,C,H,W) or (B,C,D,H,W) where C=2 is the channels (real and imaginary) and D is optional dimension for 3D MRI.
Then, the resulting measurements `y` will be of shape (B,C,N,(D,)H,W) where N is the coils dimension.

#### NOTE
We provide various random mask generators (e.g. Cartesian undersampling) that can be used directly with this physics. See e.g. [`deepinv.physics.generator.mri.RandomMaskGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.RandomMaskGenerator.html.md#deepinv.physics.generator.RandomMaskGenerator).
If mask or coil maps are not passed, a mask and maps full of ones is used (i.e. no acceleration).

#### NOTE
You can also simulate basic `birdcage coil sensitivity maps <https://mriquestions.com/birdcage-coil.html>` by passing instead an integer to `coil_maps`
using `MultiCoilMRI(coil_maps=N, img_size=x.shape)` (note this requires installing the `sigpy` library).

#### NOTE
This physics is directly compatible with FastMRI data using [`deepinv.datasets.FastMRISliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.FastMRISliceDataset.html.md#deepinv.datasets.FastMRISliceDataset).
The dataset loads pairs of RSS images and multicoil kspace `(x, y)` where `x = MultiCoilMRI().A_adjoint(y, rss=True, crop=True)`.

* **Parameters:**
  * **mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – binary sampling mask which should have shape (H,W), (C,H,W), (B,C,H,W), or (B,C,…,H,W). If None, generate mask of ones with `img_size`.
  * **coil_maps** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – either `Tensor`, integer, or `None`. If complex valued (i.e. of complex dtype) coil sensitivity maps which should have shape (H,W), (N,H,W), (B,N,H,W) or (B,N,…,H,W).
    If None, generate flat coil maps of ones with `img_size`. If integer, simulate birdcage coil maps with integer number of coils (this requires `sigpy` installed).
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – if `mask` or `coil_maps` not specified, flat `mask` or `coil_maps` of ones are created using `img_size`,
    where `img_size` can be of any shape specified above. If `mask` or `coil_maps` provided, `img_size` is ignored.
  * **three_d** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, calculate Fourier transform in 3D for 3D data (i.e. data of shape (B,C,D,H,W) where D is depth).
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – specify which device you want to use (i.e, cpu or gpu).

<hr />

* **Examples:**
  Multi-coil MRI operator:
  ```pycon
  >>> from deepinv.physics import MultiCoilMRI
  >>> seed = torch.manual_seed(0) # Random seed for reproducibility
  >>> x = torch.randn(1, 2, 2, 2) # Define random 2x2 image B,C,H,W
  >>> physics = MultiCoilMRI(img_size=x.shape) # Define coil map of ones
  >>> physics(x).shape # B,C,N,H,W
  torch.Size([1, 2, 1, 2, 2])
  >>> coil_maps = torch.randn(1, 5, 2, 2, dtype=torch.complex64) # Define 5-coil sensitivity maps
  >>> physics.update(coil_maps=coil_maps) # Update coil maps on the fly
  >>> physics(x).shape
  torch.Size([1, 2, 5, 2, 2])
  ```

#### A(x, mask=None, coil_maps=None, \*\*kwargs)

Applies linear operator.

Optionally update MRI mask or coil sensitivity maps on the fly.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – image with shape `(B,2,...,H,W)`.
  * **mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – optionally set the mask on-the-fly.
  * **coil_maps** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – optionally set the mask on-the-fly.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) multi-coil kspace measurements with shape `(B,2,N,...,H,W)` where `N` is coil dimension.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### A_adjoint(y, mask=None, coil_maps=None, rss=False, crop=False, \*\*kwargs)

Applies adjoint linear operator.

Optionally update MRI mask or coil sensitivity maps on the fly.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – multi-coil kspace measurements with shape [B,2,N,…,H,W] where N is coil dimension.
  * **mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – optionally set the mask on-the-fly.
  * **coil_maps** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – optionally set the mask on-the-fly.
  * **rss** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – perform root-sum-square reconstruction.
    This option is provided to match the original data of [`deepinv.datasets.FastMRISliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.FastMRISliceDataset.html.md#deepinv.datasets.FastMRISliceDataset),
    such that `x = MultiCoilMRI().A_adjoint(y, rss=True)`.
  * **crop** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, crop last 2 dims of x to last 2 dims of img_size.
    This option is provided to match the original data of [`deepinv.datasets.FastMRISliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.FastMRISliceDataset.html.md#deepinv.datasets.FastMRISliceDataset),
    such that `x = MultiCoilMRI().A_adjoint(y, crop=True)`.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) image with shape `(B,2,...,H,W)` if not rss else `(B,1,...,H,W)`
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### A_dagger(y, mask=None, coil_maps=None, \*\*kwargs)

Computes least squares solution to the MRI inverse problem, as proposed in [SENSE: Sensitivity encoding for fast MRI](https://doi.org/10.1002/(SICI)1522-2594(199911)42:5%3C952::AID-MRM16%3E3.0.CO;2-S).

By default uses conjugate gradient solver. Overwrite default solver arguments by passing `kwargs`. See [`deepinv.optim.linear.least_squares()`](https://deepinv.org/api/stubs/deepinv.optim.linear.least_squares.html.md#deepinv.optim.linear.least_squares) for details.

The MRI mask or coil sensitivity maps are updated if passed as inputs to the function.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – multi-coil kspace measurements with shape [B,2,N,…,H,W] where N is coil dimension.
  * **mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – optionally set the mask on-the-fly.
  * **coil_maps** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – optionally set the mask on-the-fly.
  * **kwargs** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – kwargs to pass to base [`deepinv.physics.LinearPhysics.A_dagger()`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics.A_dagger).
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) image with shape `(B,2,...,H,W)`
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### *static* check_coil_maps(coil_maps, three_d)

Check coil maps dimensions.

* **Parameters:**
  **coil_maps** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – coil sensitivity maps
* **Return torch.Tensor:**
  checked coil sensitivity maps
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### *static* estimate_coil_maps(y, calib_size=24, use_cupy=False, espirit_crop=0.95)

Estimate coil sensitivity maps using ESPIRiT.

This was proposed in [ESPIRiT — An Eigenvalue Approach to Autocalibrating Parallel MRI: Where SENSE meets GRAPPA](https://onlinelibrary.wiley.com/doi/10.1002/mrm.24751).

Note this uses a suboptimal undifferentiable unbatched implementation provided by `sigpy`.

Optionally use `cupy` to accelerate on GPU, only if `cupy` is installed and a GPU is available.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – multi-coil kspace measurements with shape [B,2,N,…,H,W] where N is coil dimension.
  * **calib_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – optional square auto-calibration size in pixels, used by `sigpy`.
  * **use_cupy** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to attempt to use cupy for GPU acceleration.
  * **espirit_crop** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – optionally set crop argument of ESPIRiT algorithm, defaults to 0.95.
* **Returns:**
  torch.Tensor of coil maps of complex dtype and shape [B,N,…,H,W]
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### noise(x, \*\*kwargs)

Incorporates noise into the measurements $\tilde{y} = N(y)$ and takes the mask into account.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – clean measurements
  * **noise_level** (*None* *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – optional noise level parameter
* **Returns:**
  noisy measurements
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### simulate_birdcage_csm(n_coils)

Simulate birdcage coil sensitivity maps. Requires library `sigpy`.

* **Parameters:**
  **n_coils** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of coils N
* **Return torch.Tensor:**
  coil maps of complex dtype of shape (N,H,W)
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### update_parameters(mask=None, coil_maps=None, check_mask=True, check_coil_maps=True, \*\*kwargs)

Update MRI subsampling mask and coil sensitivity maps.

* **Parameters:**
  * **mask** ([*torch.nn.parameter.Parameter*](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – MRI mask
  * **coil_maps** ([*torch.nn.parameter.Parameter*](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – MRI coil sensitivity maps
  * **check_mask** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – check mask dimensions before updating
  * **check_coil_maps** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – check coil maps dimensions before updating

<a id="sphx-glr-backref-deepinv-physics-multicoilmri"></a>

## Examples using `MultiCoilMRI`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various input/output functions provided by DeepInverse for handling medical and scientific imaging formats. We demonstrate loading and plotting from DICOM, NIfTI, ISMRMRD, PyTorch, NumPy and raster data sources.">  <div class="sphx-glr-thumbnail-title">Loading scientific images</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate scan-specific self-supervised learning, that is, learning to reconstruct MRI scans from a single accelerated sample without ground truth.">  <div class="sphx-glr-thumbnail-title">Scan-specific zero-shot SSDU for MRI</div>
</div>
<!-- thumbnail-parent-div-close --></div>
