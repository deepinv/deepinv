# DynamicMRI

### *class* deepinv.physics.DynamicMRI(mask=None, img_size=(320, 320), three_d=False, device='cpu', \*\*kwargs)

Bases: [`MRI`](https://deepinv.org/api/stubs/deepinv.physics.MRI.html.md#deepinv.physics.MRI), [`TimeMixin`](https://deepinv.org/api/stubs/deepinv.utils.TimeMixin.html.md#deepinv.utils.TimeMixin)

Single-coil accelerated dynamic magnetic resonance imaging.

The linear operator operates in 2D+t videos and is defined as

$$
y_t = M_t Fx_t
$$

where $M_t$ applies a time-varying mask, and $F$ is the 2D discrete Fourier Transform.
This operator has a simple singular value decomposition, so it inherits the structure of
[`deepinv.physics.DecomposablePhysics`](https://deepinv.org/api/stubs/deepinv.physics.DecomposablePhysics.html.md#deepinv.physics.DecomposablePhysics) and thus have a fast pseudo-inverse and prox operators.

The complex images $x$ and measurements $y$ should be of size (B, 2, T, H, W) where the first channel corresponds to the real part
and the second channel corresponds to the imaginary part.

A fixed mask can be set at initialisation, or a new mask can be set either at forward (using `physics(x, mask=mask)`) or using `update`.

#### NOTE
We provide various random mask generators (e.g. Cartesian undersampling) that can be used directly with this physics. See e.g. [`deepinv.physics.generator.mri.RandomMaskGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.RandomMaskGenerator.html.md#deepinv.physics.generator.RandomMaskGenerator)

* **Parameters:**
  * **mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – binary mask, where 1s represent sampling locations, and 0s otherwise.
    The mask size can either be (H,W), (T,H,W), (C,T,H,W) or (B,C,T,H,W) where H, W are the image height and width, T is time-steps, C is channels (typically 2) and B is batch size.
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – if mask not specified, flat mask of ones is created using `img_size`, where `img_size` can be of any shape specified above. If mask provided, `img_size` is ignored.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – cpu or gpu.

<hr />

* **Examples:**
  Single-coil accelerated 2D+t MRI operator:
  ```pycon
  >>> from deepinv.physics import DynamicMRI
  >>> seed = torch.manual_seed(0) # Random seed for reproducibility
  >>> x = torch.randn(1, 2, 2, 2, 2) # Define random video of shape (B,C,T,H,W)
  >>> mask = torch.rand_like(x) > 0.75 # Define random 4x subsampling mask
  >>> physics = DynamicMRI(mask=mask) # Physics with given mask
  >>> physics.update(mask=mask) # Alternatively set mask on-the-fly
  >>> physics(x)
  tensor([[[[[-0.0000,  0.7969],
             [-0.0000, -0.0000]],

            [[-0.0000, -1.9860],
             [-0.0000, -0.4453]]],


           [[[ 0.0000,  0.0000],
             [-0.8137, -0.0000]],

            [[-0.0000, -0.0000],
             [-0.0000,  1.1135]]]]])
  ```

#### A_adjoint(y, mask=None, mag=False, \*\*kwargs)

Adjoint operator.

Optionally perform magnitude to reduce channel dimension.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input kspace of shape `(B,2,T,H,W)`
  * **mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – optionally set mask on-the-fly, see class docs for shapes allowed.
  * **mag** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – perform complex magnitude.

#### check_mask(mask=None, \*\*kwargs)

Updates MRI mask and verifies mask shape to be B,C,T,H,W.

:param torch.nn.parameter.Parameter, float MRI subsampling mask.

#### noise(x, \*\*kwargs)

Incorporates noise into the measurements $\tilde{y} = N(y)$

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – clean measurements
* **Return torch.Tensor:**
  noisy measurements

#### to_static(mask=None, device='cpu')

Convert dynamic MRI to static MRI by removing time dimension.

* **Parameters:**
  **mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – new static MRI mask. If None, existing mask is flattened (summed) along the time dimension.
* **Return MRI:**
  static MRI physics
* **Return type:**
  [*MRI*](https://deepinv.org/api/stubs/deepinv.physics.MRI.html.md#deepinv.physics.MRI)

<a id="sphx-glr-backref-deepinv-physics-dynamicmri"></a>

## Examples using `DynamicMRI`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This examples shows you how to use DeepInverse with your own physics.">  <div class="sphx-glr-thumbnail-title">Bring your own physics</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the self-supervised Artifact2Artifact loss for solving an undersampled sequential MRI reconstruction problem without ground truth.">  <div class="sphx-glr-thumbnail-title">Self-supervised MRI reconstruction with Artifact2Artifact</div>
</div>
<!-- thumbnail-parent-div-close --></div>
