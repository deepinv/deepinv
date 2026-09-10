# SequentialMRI

### *class* deepinv.physics.SequentialMRI(mask=None, img_size=(320, 320), three_d=False, device='cpu', \*\*kwargs)

Bases: [`DynamicMRI`](https://deepinv.org/api/stubs/deepinv.physics.DynamicMRI.html.md#deepinv.physics.DynamicMRI)

Single-coil accelerated magnetic resonance imaging using sequential sampling.

Let $M$ be a subsampling mask with given acceleration.
$M_t$ is a time-varying mask with the sequential sampling pattern e.g. non-overlapping lines or spokes, such that $S=\bigcup_t S_t$.
The sequential MRI operator then simulates a time sequence of k-space samples:

$$
y_t = M_t F x
$$

where $F$ is the 2D discrete Fourier Transform, the image $x$ is of shape (B, 2, H, W) and measurements $y$ is of shape (B, 2, T, H, W)
where the first channel corresponds to the real part and the second channel corresponds to the imaginary part.

This operator has a simple singular value decomposition, so it inherits the structure of [`deepinv.physics.DecomposablePhysics`](https://deepinv.org/api/stubs/deepinv.physics.DecomposablePhysics.html.md#deepinv.physics.DecomposablePhysics)
and thus have a fast pseudo-inverse and prox operators.

A fixed mask can be set at initialisation, or a new mask can be set either at forward (using `physics(x, mask=mask)`)
or using `update`.

#### NOTE
We provide various random mask generators (e.g. Cartesian undersampling) that can be used directly with this physics. See e.g. [`deepinv.physics.generator.mri.RandomMaskGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.RandomMaskGenerator.html.md#deepinv.physics.generator.RandomMaskGenerator)

* **Parameters:**
  * **mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – binary mask $S_t,t=1\ldots T$, where 1s represent sampling locations, and 0s otherwise.
    The mask size can either be (H,W), (T,H,W), (C,T,H,W) or (B,C,T,H,W) where H, W are the image height and width, T is time-steps, C is channels (typically 2) and B is batch size.
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – if mask not specified, flat mask of ones is created using `img_size`, where `img_size` can be of any shape specified above. If mask provided, `img_size` is ignored.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – cpu or gpu.

<hr />

* **Examples:**
  Single-coil accelerated sequential MRI operator:
  ```pycon
  >>> from deepinv.physics import SequentialMRI
  >>> x = torch.randn(1, 2, 2, 2) # Define random image of shape (B,C,H,W)
  >>> mask = torch.zeros(1, 2, 3, 2, 2) # Empty demo time-varying mask with 3 frames
  >>> physics = SequentialMRI(mask=mask) # Physics with given mask
  >>> physics.update(mask=mask) # Alternatively set mask on-the-fly
  >>> physics(x).shape # MRI sequential samples
  torch.Size([1, 2, 3, 2, 2])
  ```

#### A_adjoint(y, mask=None, keep_time_dim=False, \*\*kwargs)

Computes the adjoint of the forward operator $\tilde{x} = A^{\top}y$.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor
  * **mask** ([*torch.nn.parameter.Parameter*](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – input mask
  * **keep_time_dim** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, adjoint is calculated frame-by-frame. Used for visualisation. If `False`, flatten the time dimension before calculating.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) output tensor
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-physics-sequentialmri"></a>

## Examples using `SequentialMRI`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the self-supervised Artifact2Artifact loss for solving an undersampled sequential MRI reconstruction problem without ground truth.">  <div class="sphx-glr-thumbnail-title">Self-supervised MRI reconstruction with Artifact2Artifact</div>
</div>
<!-- thumbnail-parent-div-close --></div>
