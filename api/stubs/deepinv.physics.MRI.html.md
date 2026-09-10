# MRI

### *class* deepinv.physics.MRI(mask=None, img_size=(320, 320), three_d=False, device='cpu', \*\*kwargs)

Bases: [`MRIMixin`](https://deepinv.org/api/stubs/deepinv.utils.MRIMixin.html.md#deepinv.utils.MRIMixin), [`DecomposablePhysics`](https://deepinv.org/api/stubs/deepinv.physics.DecomposablePhysics.html.md#deepinv.physics.DecomposablePhysics)

Single-coil accelerated 2D or 3D magnetic resonance imaging.

The linear operator operates in 2D slices or 3D volumes and is defined as

$$
y = MFx
$$

where $M$ applies a mask (subsampling operator), and $F$ is the 2D or 3D discrete Fourier Transform.
This operator has a simple singular value decomposition, so it inherits the structure of
[`deepinv.physics.DecomposablePhysics`](https://deepinv.org/api/stubs/deepinv.physics.DecomposablePhysics.html.md#deepinv.physics.DecomposablePhysics) and thus have a fast pseudo-inverse and prox operators.

The complex images $x$ and measurements $y$ should be of size (B, C,…, H, W) with C=2, where the first channel corresponds to the real part
and the second channel corresponds to the imaginary part. The `...` is an optional depth dimension for 3D MRI data.

A fixed mask can be set at initialisation, or a new mask can be set either at forward (using `physics(x, mask=mask)`) or using `update`.

#### NOTE
We provide various random mask generators (e.g. Cartesian undersampling) that can be used directly with this physics. See e.g. [`deepinv.physics.generator.mri.RandomMaskGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.RandomMaskGenerator.html.md#deepinv.physics.generator.RandomMaskGenerator)
If mask is not passed, a mask full of ones is used (i.e. no acceleration).

#### NOTE
This physics is directly compatible with FastMRI data using [`deepinv.datasets.FastMRISliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.FastMRISliceDataset.html.md#deepinv.datasets.FastMRISliceDataset).
The dataset loads pairs of magnitude images and kspace `(x, y)` where `x = MRI().A_adjoint(y, mag=True, crop=True)`.

* **Parameters:**
  * **mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – binary mask, where 1s represent sampling locations, and 0s otherwise.
    The mask size can either be (H,W), (C,H,W), (B,C,H,W), (B,C,…,H,W) where H, W are the image height and width, C is channels (which should be 2) and B is batch size.
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – if mask not specified, flat mask of ones is created using `img_size`, where `img_size` can be of any shape specified above. If mask provided, `img_size` is ignored.
  * **three_d** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, calculate Fourier transform in 3D for 3D data (i.e. data of shape (B,C,D,H,W) where D is depth).
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – cpu or gpu.

<hr />

* **Examples:**
  Single-coil accelerated MRI operator with subsampling mask:
  ```pycon
  >>> from deepinv.physics import MRI
  >>> seed = torch.manual_seed(0) # Random seed for reproducibility
  >>> x = torch.randn(1, 2, 2, 2) # Define random 2x2 image
  >>> mask = 1 - torch.eye(2) # Define subsampling mask
  >>> physics = MRI(mask=mask) # Define mask at initialisation
  >>> physics(x)
  tensor([[[[ 0.0000, -1.4290],
            [ 0.4564, -0.0000]],

           [[ 0.0000,  1.8622],
            [ 0.0603, -0.0000]]]])
  >>> physics = MRI(img_size=x.shape) # No subsampling
  >>> physics(x)
  tensor([[[[ 2.2908, -1.4290],
            [ 0.4564, -0.1814]],

           [[ 0.3744,  1.8622],
            [ 0.0603, -0.6209]]]])
  >>> physics.update(mask=mask) # Update mask on the fly
  >>> physics(x)
  tensor([[[[ 0.0000, -1.4290],
            [ 0.4564, -0.0000]],

           [[ 0.0000,  1.8622],
            [ 0.0603, -0.0000]]]])
  ```

#### A_adjoint(y, mask=None, mag=False, crop=False, \*\*kwargs)

Adjoint operator.

Optionally perform crop and magnitude to match FastMRI data.

By default, crop and magnitude are not performed.
By setting `mag=crop=True`, the outputs will be consistent with [`deepinv.datasets.FastMRISliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.FastMRISliceDataset.html.md#deepinv.datasets.FastMRISliceDataset).

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input kspace of shape (B,C,…,H,W)
  * **mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – optionally set mask on-the-fly.
  * **mag** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – perform complex magnitude.
    This option is provided to match the original data of [`deepinv.datasets.FastMRISliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.FastMRISliceDataset.html.md#deepinv.datasets.FastMRISliceDataset),
    such that `x = MRI().A_adjoint(y, mag=True)`.
  * **crop** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, crop last 2 dims of x to last 2 dims of img_size.
    This option is provided to match the original data of [`deepinv.datasets.FastMRISliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.FastMRISliceDataset.html.md#deepinv.datasets.FastMRISliceDataset),
    such that `x = MRI().A_adjoint(y, crop=True)`.

#### noise(x, \*\*kwargs)

Incorporates noise into the measurements $\tilde{y} = N(y)$

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – clean measurements
* **Return torch.Tensor:**
  noisy measurements

#### update_parameters(mask=None, check_mask=True, \*\*kwargs)

Update MRI subsampling mask.

* **Parameters:**
  * **mask** ([*torch.nn.parameter.Parameter*](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – MRI mask
  * **check_mask** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – check mask dimensions before updating

<a id="sphx-glr-backref-deepinv-physics-mri"></a>

## Examples using `MRI`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This examples shows you how to use DeepInverse with your own physics.">  <div class="sphx-glr-thumbnail-title">Bring your own physics</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to get started with DeepInverse in under 5 minutes.">  <div class="sphx-glr-thumbnail-title">5 minute quickstart tutorial</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model :footciteterris2025reconstruct to solve inverse problems.">  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in physics.">  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the self-supervised Artifact2Artifact loss for solving an undersampled sequential MRI reconstruction problem without ground truth.">  <div class="sphx-glr-thumbnail-title">Self-supervised MRI reconstruction with Artifact2Artifact</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a reconstruction network for an MRI inverse problem on a fully self-supervised way, i.e., using measurement data only.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Imaging for MRI.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
