# SequentialMultiCoilMRI

### *class* deepinv.physics.SequentialMultiCoilMRI(\*args, transform=None, transform_params=None, \*\*kwargs)

Bases: [`DynamicMultiCoilMRI`](https://deepinv.org/api/stubs/deepinv.physics.DynamicMultiCoilMRI.html.md#deepinv.physics.DynamicMultiCoilMRI)

Multi-coil MRI with sequential (“time-resolved”) sampling.

Models sequential sampling over time with an optional transform per time step.

Let $M$ be a subsampling mask with given acceleration.
$M_t$ is a time-varying mask with the sequential sampling pattern e.g. non-overlapping lines or spokes, such that $M=\bigcup_t M_t$.
The sequential MRI operator then simulates a time sequence of k-space samples:

$$
y_{n,t} = M_t F \operatorname{diag}(s_n) \mathcal{T}_t x,
$$

where $F$ is the 2D or 3D discrete Fourier Transform, $s_n$ is the n-th coil map, and
$\mathcal{T}_t$ is a transform per time-step.
$x$ is a 2D image / 3D volume of shape (B, 2, (D,) H, W), and $y$ is the kspace of shape (B, 2, N, T, (D,) H, W)
where the first channel corresponds to the real part and the second channel corresponds to the imaginary part,
`N` is number of coils, `T` is number of timesteps, and `D` is optional depth dimension for 3D data.

When a transform is used, this physics can be used for motion-compensated MRI reconstruction.

#### TIP
See [Dynamic & time-resolved imaging](https://deepinv.org/user_guide/physics/physics.html.md#dynamic-imaging) for how to use `transform` and `transform_params` for motion-compensated MRI simulation and reconstruction.

#### NOTE
The transforms are applied over time by looping. In a future version, we will parallelise transforms across time.
If this feature is critical for your application, please open a feature request issue on GitHub.

* **Parameters:**
  * **mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – sequential mask with shape `(B,2,T,H,W)` or `(B,2,T,D,H,W)`.
  * **coil_maps** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – complex coil maps with shape `(B,N,H,W)` or `(B,N,D,H,W)`.
  * **img_size** ([*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple)) – image size used when `mask` or `coil_maps` is not specified.
  * **three_d** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – if `True`, apply a 3D spatial Fourier transform.
  * **transform** ([*deepinv.transform.Transform*](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform)) – optional transform applied per time-step.
  * **transform_params** ([*dict*](https://docs.python.org/3.12/library/stdtypes.html#dict)) – optional transform parameters, each of shape `(1,T,...)` for one trajectory shared across the batch,
    or `(B,T,...)` for one trajectory per batch element, in which case `transform` must be built with `index_params_into_batch=True`.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.12/library/stdtypes.html#str)) – computation device.

<hr />

* **Example:**

```pycon
>>> import torch
>>> from deepinv.physics import SequentialMultiCoilMRI
>>> x = torch.randn(1, 2, 8, 8)  # Static image (B,2,H,W)
>>> mask = torch.zeros(1, 2, 3, 8, 8)
>>> mask[:, :, 0, :, 1] = mask[:, :, 1, :, 3] = mask[:, :, 2, :, 6] = 1
>>> coil_maps = torch.ones(1, 4, 8, 8, dtype=torch.complex64)
>>> physics = SequentialMultiCoilMRI(mask=mask, coil_maps=coil_maps)
>>> physics(x).shape
torch.Size([1, 2, 4, 3, 8, 8])
```

#### A_adjoint(y, keep_time_dim=False, blind=False, \*\*kwargs)

Apply the adjoint operator.

When a transform is used, this is known as motion-compensated reconstruction.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – temporal kspace of shape `(B,2,N,T,(D,)H,W)`.
  * **keep_time_dim** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – if `True`, return time dim in the adjoint (for visualisation) instead of summing over time.
  * **blind** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – if `False` (default), perform adjoint by inversing the motion.
    If `True`, perform a ‘blind’ adjoint i.e. call the multicoil adjoint without applying motion.
* **Returns:**
  adjoint image of shape `(B,2,(D,)H,W)`, or `(B,2,T,(D,)H,W)` if `keep_time_dim=True`.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-physics-sequentialmulticoilmri"></a>

## Examples using `SequentialMultiCoilMRI`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Many imaging modalities are not acquired all at once but sequentially in &quot;shots&quot;, e.g. kspace lines in MRI or projections in CT. Since the time between shots is non-negligible, the subject may move during scanning, and reconstructions that assume a static subject will be motion-corrupted. Modelling a transform per shot allows us to simulate this motion corruption and, given the motion parameters, to compensate for it during reconstruction.">![](auto_examples/physics/images/thumb/sphx_glr_demo_motion_mri_thumb.png)

[Motion simulation and motion-compensated reconstruction](https://deepinv.org/auto_examples/physics/demo_motion_mri.html.md)

  <div class="sphx-glr-thumbnail-title">Motion simulation and motion-compensated reconstruction</div>
</div>
<!-- thumbnail-parent-div-close --></div>
