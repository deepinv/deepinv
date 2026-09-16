# PET

### *class* deepinv.physics.PET(img_size, voxel_size=(2, 2, 2), fwhm_data_mm=4.0, scanner=None, radial_trim=3, gain=1.0, normalize=False, normalize_counts=False, device='cpu', views=None, background=None, attenuation=None, \*\*kwargs)

Bases: [`LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)

Non time-of-flight Positron emission tomography (PET) physics model.

This operator relies on the `parallelproj` library by Schramm and Thielemans<sup>[1](#footcite-schramm2024parallelproj)</sup>.

The PET forward model is defined as

$$
y \sim \gamma \mathcal{P}\left(\frac{c \circ H(g*x) + b}{\gamma}\right)
$$

where $H \in \mathbb{R}_{+}^{m \times n}$ is the projection operator,
$g \in \mathbb{R}_{+}^{n}$ is a Gaussian blur kernel, $x\in\mathbb{R}_{+}^{n}$
is the emission image, $b \in \mathbb{R}_{+}^{m}$ is the (expected) background,
$\mathcal{P}$ denotes Poisson noise with gain $\gamma > 0$,
$c=\exp(-H\mu)\in \mathbb{R}_{+}^{m}$ is an (optional) attenuation term
with $\mu \in \mathbb{R}_{+}^{n}$ an attenuation map (typically obtained through an auxiliary CT scan).

The operator **can be used on 2D images or 3D volumes**.

The operator relies on parameters `background` and `attenuation` that can be updated through the
[`physics.update`](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics.update) method or when evaluating
[`physics.A`](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics.A) or [`physics.A_adjoint`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics.A_adjoint).

#### NOTE
This operator requires the `parallelproj` package to be installed.
This in turn requires [installing deepinv via pixi or conda](https://deepinv.org/index.html.md#install),
but not pypi/uv (as `parallelproj` is not currently available on pypi).

If you are working on a conda environment, you can install `parallelproj` as

```default
conda install -c conda-forge parallelproj
```

If you are working on a pixi installation, simply do

```default
pixi install -e full
```

which installs all optional dependencies.

Check the `parallelproj` documentation for more details: [https://parallelproj.readthedocs.io/en/stable/](https://parallelproj.readthedocs.io/en/stable/).

#### TIP
Check out the [2D](https://deepinv.org/auto_examples/physics/demo_pet2d.html.md#sphx-glr-auto-examples-physics-demo-pet2d-py) and
[3D](https://deepinv.org/auto_examples/physics/demo_pet3d.html.md#sphx-glr-auto-examples-physics-demo-pet3d-py) examples to get started with this operator.

#### NOTE
This operator currently only supports sinogram non-ToF data.
To use this operator with listmode data and/or ToF data,
you can easily swap out the projector `self.proj` for the appropriate listmode or ToF projector.
See `parallelproj` [docs](https://parallelproj.readthedocs.io/) for more details.

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – shape of the input 2D `(H, W)` or 3D volumes `(D, H, W)`.
  * **voxel_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – voxel size in mm. Default is 2 x 2 x 2 mm.
  * **fwhm_data_mm** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – full width at half maximum (FWHM) of the Gaussian blur $g$. It has a crucial impact on the maximum achievable resolution,
    which is typically a fraction of the FWHM.
  * **scanner** (*None* *,* [*parallelproj.pet_scanners.ModularizedPETScannerGeometry*](https://parallelproj.readthedocs.io/en/stable/api_pet_scanners.html#parallelproj.pet_scanners.ModularizedPETScannerGeometry)) – Scanner configuration. If None, a default demo scanner from parallelproj is used.
  * **radial_trim** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – radial trim of rays on the sides of the volume to improve efficiency.
  * **gain** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – gain factor $\gamma$ for the Poisson noise model.
  * **normalize** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True` the forward operator is normalized such that $\|A\|=1$.
  * **normalize_counts** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `False` the $\gamma$ normalization term in front of the Poisson noise is removed,
    so that the measurements $y$ are true integer counts.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *|* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – device to run the computations on, e.g. `"cpu"` or `"cuda"`
  * **views** (*None* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – one-dimensional tensor of integer indices selecting the PET sinogram views to project. If `None`, all views are projected.
  * **background** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – background sinogram $b$, i.e. the expected number of background events in each LOR, with shape `(num_lors,)`
  * **attenuation** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – attenuation map. Can be provided either in **image space** as $\mu$
    (linear attenuation coefficients, shape `(H,W)` for 2D or `(D,H,W)` for 3D — typically from an auxiliary CT scan),
    or in **sinogram/projection space** as $c=\exp(-H\mu)$. The space is inferred automatically
    by comparing the spatial dimensions of the tensor against `img_size`: if they match, image space is assumed
    and the attenuation is projected; otherwise, sinogram space is assumed and the tensor is used directly.
    Providing the attenuation in image space allows computing gradients with respect to it efficiently.

<hr />

* **Example:**

Simulate 2D PET measurements

```pycon
>>> from deepinv.physics import PET
>>> import torch
>>> img_size = (64, 64)
>>> physics = PET(img_size=img_size)
>>> x = torch.rand((1, 1) + img_size)
>>> background = torch.ones_like(physics.A(x))
>>> attenuation = torch.rand((1, 1,) + img_size)
>>> y = physics(x, attenuation=attenuation, background=background)
>>> y.shape
torch.Size([1, 1, 539, 272])
```

<hr />

* **References:**

* <a id='footcite-schramm2024parallelproj'>**[1]**</a> Georg Schramm and Kris Thielemans. Parallelproj—an open-source framework for fast calculation of projections in tomography. *Frontiers in Nuclear Medicine*, 3:1324562, 2024.

#### A(x, add_background=False, background=None, attenuation=None, \*\*kwargs)

Apply the linear operator $Ax=c \circ H(g*x)$ to a signal $x$

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image or volume of shape `(B,1,H,W)` for 2D or `(B,1,D,H,W)` for 3D where `B` is the batch size.
  * **add_background** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – whether to add background $b$. By default, no background is added.
  * **background** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – If not `None`, update the background $b$ of the operator.
  * **attenuation** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – If not `None`, update the attenuation $c$ of the operator.
    The space (image or sinogram) is inferred automatically from the tensor shape.
* **Returns:**
  sinogram of shape `(B,1,N,N/2,R^2)` where `N` is the number of detectors per ring and `R` is the number of rings.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### A_adjoint(y, attenuation=None, background=None, \*\*kwargs)

Apply the adjoint of the linear operator $A^{\top}y$ where $A=c \circ H(g*\cdot)$ to a sinogram $y$

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input sinogram of shape `(B,1,N,N/2,R^2)` where `N` is the number of detectors per ring and `R` is the number of rings.
  * **attenuation** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – If not `None`, update the attenuation $c$ of the operator
  * **background** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – If not `None`, update the background $b$ of the operator

#### forward(x, attenuation=None, background=None, \*\*kwargs)

Generate PET measurements.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image or volume
  * **attenuation** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – If not `None`, update the attenuation $c$ of the operator
  * **background** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – If not `None`, update the background $b$ of the operator

#### generate_background(expected_background)

Generate a random PET background based on the expected background.

* **Parameters:**
  **expected_background** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Expected background.

#### plot_geometry()

Plot the scanner geometry.

#### update_parameters(attenuation=None, background=None, \*\*kwargs)

Update the background and/or attenuation parameters.

The space of the attenuation tensor is inferred automatically: if the last
`len(img_size)` dimensions match `img_size`, the tensor is treated as an
image-space attenuation map $\mu$ and projected; otherwise it is treated as
a sinogram-space attenuation $c=\exp(-H\mu)$ and used directly.

* **Parameters:**
  * **attenuation** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – If not `None`, update the attenuation $c$ of the operator. Can be in
    image space (shape matching `img_size`) or sinogram space.
  * **background** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – If not `None`, update the background $b$.

<a id="sphx-glr-backref-deepinv-physics-pet"></a>

## Examples using `PET`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct an image from them.">  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 2D</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct a volume from them.">  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 3D</div>
</div>
<!-- thumbnail-parent-div-close --></div>
