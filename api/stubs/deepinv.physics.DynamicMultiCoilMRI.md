# DynamicMultiCoilMRI

### *class* deepinv.physics.DynamicMultiCoilMRI(\*args, transform=None, transform_params=None, \*\*kwargs)

Bases: [`MultiCoilMRI`](https://deepinv.org/api/stubs/deepinv.physics.MultiCoilMRI.md#deepinv.physics.MultiCoilMRI), [`TimeMixin`](https://deepinv.org/api/stubs/deepinv.utils.TimeMixin.md#deepinv.utils.TimeMixin)

Dynamic multi-coil MRI for 2D+t or 3D+t (i.e. 4D) data.

The linear operator operates in 2D+t or 3D+t videos and is defined as:

$$
y_{n,t} = \operatorname{diag}(p_t) F \operatorname{diag}(s_n) \mathcal{T}_t x_t
$$

for $n=1,\dots,N$ coils and $t=1,\dots,T$ time steps, where
$p_t$ is the sampling mask at time $t$, $F$ is the spatial
Fourier transform, $s_n$ is the sensitivity of the nth coil, and
$\mathcal{T}_t$ is an optional `transform` per time-step, e.g. to model motion.

The input image has shape `(B, 2, T, (D,) H, W)` and the output k-space has
shape `(B, 2, N, T, (D,) H, W)`, where `N` is the number of coils, `T` is
number of timesteps and `D` is optional depth dim (for 3D).

* **Parameters:**
  * **mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – dynamic mask with shape (T,H,W), (C,T,H,W), (B,C,T,H,W), or (B,C,T,…,H,W)
  * **coil_maps** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – complex coil maps with shape `(B,N,H,W)` or `(B,N,T,H,W)` (time-varying maps), see [`deepinv.physics.MultiCoilMRI`](https://deepinv.org/api/stubs/deepinv.physics.MultiCoilMRI.md#deepinv.physics.MultiCoilMRI) for details.
  * **img_size** ([*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple)) – image size used when `mask` or `coil_maps` is not specified, see [`deepinv.physics.MultiCoilMRI`](https://deepinv.org/api/stubs/deepinv.physics.MultiCoilMRI.md#deepinv.physics.MultiCoilMRI) for details.
  * **three_d** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – if `True`, assume 3D data (`(..., D,H,W)`) and apply a 3D spatial Fourier transform.
  * **transform** ([*deepinv.transform.Transform*](https://deepinv.org/api/stubs/deepinv.transform.Transform.md#deepinv.transform.Transform)) – optional transform applied per time-step.
  * **transform_params** ([*dict*](https://docs.python.org/3.12/library/stdtypes.html#dict)) – optional transform parameters, each of shape `(1,T,...)` for one trajectory shared across the batch,
    or `(B,T,...)` for one trajectory per batch element, in which case `transform` must be built with `index_params_into_batch=True`.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.12/library/stdtypes.html#str)) – computation device.

<hr />

* **Example:**

```pycon
>>> import torch
>>> from deepinv.physics import DynamicMultiCoilMRI
>>> x = torch.randn(1, 2, 3, 8, 8)  # (B,2,T,H,W)
>>> mask = torch.ones_like(x)
>>> coil_maps = torch.ones(1, 4, 8, 8, dtype=torch.complex64)
>>> physics = DynamicMultiCoilMRI(mask=mask, coil_maps=coil_maps)
>>> physics(x).shape
torch.Size([1, 2, 4, 3, 8, 8])
```

#### to_static(device='cpu')

Convert dynamic multi-coil MRI to static multi-coil MRI.

This conversion is performed by removing time dimension. The mask is built by retaining all the sampled locations across time, as

$$
\tilde{M} = \bigcup_t M_t = \operatorname{max}_t M_t

$$

In the case of time-varying coil sensitivity maps, these are averaged across time.

* **Parameters:**
  **device** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str)) – device to convert to
* **Return MultiCoilMRI physics:**
  equivalent temporal dimension free MulicoilMRI physics
* **Return type:**
  [*MultiCoilMRI*](https://deepinv.org/api/stubs/deepinv.physics.MultiCoilMRI.md#deepinv.physics.MultiCoilMRI)

#### update_parameters(\*\*kwargs)

Update MRI parameters and transform parameters.

If kwargs contains any `transform` params (e.g. `theta` for [`deepinv.transform.Rotate`](https://deepinv.org/api/stubs/deepinv.transform.Rotate.md#deepinv.transform.Rotate)),
then the `transform_params` dict is updated instead.

* **Parameters:**
  **kwargs** – MRI parameters, see [`deepinv.physics.MultiCoilMRI.update_parameters()`](https://deepinv.org/api/stubs/deepinv.physics.MultiCoilMRI.md#deepinv.physics.MultiCoilMRI.update_parameters), and/or transform parameters.

<a id="sphx-glr-backref-deepinv-physics-dynamicmulticoilmri"></a>

## Examples using `DynamicMultiCoilMRI`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Many imaging modalities are not acquired all at once but sequentially in &quot;shots&quot;, e.g. kspace lines in MRI or projections in CT. Since the time between shots is non-negligible, the subject may move during scanning, and reconstructions that assume a static subject will be motion-corrupted. Modelling a transform per shot allows us to simulate this motion corruption and, given the motion parameters, to compensate for it during reconstruction.">![](auto_examples/physics/images/thumb/sphx_glr_demo_motion_mri_thumb.png)

[Motion simulation and motion-compensated reconstruction](https://deepinv.org/auto_examples/physics/demo_motion_mri.md)

  <div class="sphx-glr-thumbnail-title">Motion simulation and motion-compensated reconstruction</div>
</div>
<!-- thumbnail-parent-div-close --></div>
