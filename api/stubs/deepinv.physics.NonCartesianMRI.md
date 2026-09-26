# NonCartesianMRI

### *class* deepinv.physics.NonCartesianMRI(img_size, num_shots=100, num_samples_per_shot=500, trajectory='radial', tilt='uniform', in_out=False, coil_maps=None, backend='finufft', normalize=False, device='cpu', \*\*kwargs)

Bases: [`MultiCoilMRI`](https://deepinv.org/api/stubs/deepinv.physics.MultiCoilMRI.md#deepinv.physics.MultiCoilMRI), [`MRIMixin`](https://deepinv.org/api/stubs/deepinv.utils.MRIMixin.md#deepinv.utils.MRIMixin)

Non-Cartesian (multi-coil) MRI via `mri-nufft`.

This physics wraps non-uniform FFT forward and adjoint operators provided by the `mri-nufft` [library](https://mind-inria.github.io/mri-nufft/index.html), and models non-Cartesian MRI sequences such as
radial or spiral sampling.

The physics also supports other `mri-nufft` functionality such as density compensation, which is provided in `A_dagger(density_compensate=True)`.

#### TIP
This operator is differentiable via the autograd function.

We assume that `x` is of shape `(B,2,H,W)` and kspace `y` are `(B,2,N,S)` where `N` = coils and `S` = num shots \* num samples per shot.

#### NOTE
Only supports 2D acquisition for now. For 3D/stacked physics, please open a feature request issue on GitHub.

#### NOTE
This physics supports batching, along as the backend accepts it. See [mri-nufft backend docs](https://mind-inria.github.io/mri-nufft/backend.html).

#### TIP
This is a thin wrapper of `mri-nufft`. Learn more about their [extensive MRI support](https://mind-inria.github.io/mri-nufft/index.html), such as more advanced trajectories,
trajectory estimation, various coil map estimation algorithms or off-resonance correction.

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – reconstructed image size `(H, W)` (no channel dim).
  * **num_shots** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of sampling shots `Nc` (e.g. spokes), default to 100
  * **num_samples_per_shot** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of samples per shot `Ns`, default to 500
  * **trajectory** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – `radial` or `spiral`, passed to `mri-nufft`, default to ‘radial’
  * **tilt** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – tilt of the shots, options include those listed in `mrinufft.trajectories.utils.initialize_tilt` docs, or `golden`/`grasp` for golden-angle tilt. Default to ‘uniform’
  * **in_out** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to start sampling from the center or not, default `False`.
  * **coil_maps** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *,* *None*) – complex coil sensitivity maps of shape `(H,W)`, `(N,H,W)` or `(B,N,H,W)`. `int` `N` simulates `N` birdcage maps (requires `sigpy`). `None` = single-coil (flat map) (default).
  * **backend** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – mri-nufft backend. Use `finufft` for CPU (default), `cufinufft` for CUDA. Set to `mps` to use finufft on Apple MPS, which avoids a torch threading clash with `libomp`.
  * **normalize** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether normalise by empirical norm, default `False`.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – physics device, default `'cpu'`.

#### A(x, \*\*kwargs)

MRI-NUFFT forward operator.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – image of shape B,2,H,W
* **Returns:**
  [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), multicoil kspace of shape B,2,N,S, where N is coil dim, and S is shots \* samples dim
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### A_adjoint(y, rss=False, \*\*kwargs)

MRI-NUFFT adjoint operator.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – multi-coil kspace measurements with shape B,2,N,S where N is coil dimension.
  * **rss** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – perform root-sum-square reconstruction over coils and take magnitude.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) image of shape `(B,2,H,W)` if not rss else `(B,1,H,W)`
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### A_dagger(y, density_compensate=False, rss=False, \*\*kwargs)

Computes the solution in $x$ to $y = Ax$ using a least squares solver. A faster approximation can be obtained by setting `density_compensate=True`,
which computes the filtered (i.e. density-compensated) backprojection (i.e. adjoint).

#### WARNING
The density-compensated adjoint is not the exact linear pseudo-inverse of the NUFFT problem, but it is a good approximation.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – multi-coil kspace measurements with shape B,2,N,S where N is coil dimension.
  * **density_compensation** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – fast approximation to the pseudo-inverse: Voronoi density compensation by multiplying density in the adjoint.
  * **rss** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – perform root-sum-square reconstruction over coils and take magnitude.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) image of shape `(B,2,H,W)` if not rss else `(B,1,H,W)`

#### estimate_coil_maps(y, method='low_frequency', \*\*kwargs)

Estimate coil sensitivity maps from non-Cartesian kspace via `mri-nufft`.

Unlike [`deepinv.physics.MultiCoilMRI.estimate_coil_maps()`](https://deepinv.org/api/stubs/deepinv.physics.MultiCoilMRI.md#deepinv.physics.MultiCoilMRI.estimate_coil_maps) which uses ACS region,
non-Cartesian estimation reconstructs low-frequency per-coil images
See `mrinufft.extras.get_smaps` for details.

#### NOTE
The estimation uses `mri-nufft` and is performed on CPU.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – multi-coil kspace `(B,2,N,S)`.
  * **method** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – `mri-nufft` smaps method, either `low_frequency` or `espirit`.
* **Returns:**
  complex coil maps `(B,N,H,W)`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### noise(x, \*\*kwargs)

Bypass MultiCoilMRI Cartesian masked noise

<a id="sphx-glr-backref-deepinv-physics-noncartesianmri"></a>

## Examples using `NonCartesianMRI`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs raw non-Cartesian multicoil kspace data from the FastMRI breast dataset solomonFastMRI2025, for mammography.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_mrinufft_breast_thumb.png)

[Reconstruct accelerated non-Cartesian breast MRI acquisition data](https://deepinv.org/auto_examples/external-libraries/demo_mrinufft_breast.md)

  <div class="sphx-glr-thumbnail-title">Reconstruct accelerated non-Cartesian breast MRI acquisition data</div>
</div>
<!-- thumbnail-parent-div-close --></div>
