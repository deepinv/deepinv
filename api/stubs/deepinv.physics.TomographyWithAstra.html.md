# TomographyWithAstra

### *class* deepinv.physics.TomographyWithAstra(img_size=None, angles=180, n_detector_pixels=None, angular_range=(0, 180), detector_spacing=1.0, pixel_spacing=1.0, bounding_box=None, geometry_type='parallel', geometry_parameters=MappingProxyType({'source_radius': 80.0, 'detector_radius': 20.0}), geometry_vectors=None, object_geometry=None, projection_geometry=None, is_2d=None, normalize=None, device=torch.device('cuda'), \*\*kwargs)

Bases: [`LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)

Computed Tomography operator with [astra-toolbox](https://astra-toolbox.com/) backend.
It is more memory efficient than the [`deepinv.physics.Tomography`](https://deepinv.org/api/stubs/deepinv.physics.Tomography.html.md#deepinv.physics.Tomography) operator and support 3D geometries.
See documentation of [`deepinv.physics.functional.XrayTransform`](https://deepinv.org/api/stubs/deepinv.physics.functional.XrayTransform.html.md#deepinv.physics.functional.XrayTransform) for more
information on the `astra` wrapper.

Mathematically, it is described as a ray transform
$A$ which linearly integrates an object $x$ along straight
lines

$$
y = \forw{x}

$$

where $y$ is the set of line integrals, called sinogram in 2D, or
radiographs in 3D. An object is typically scanned using a surrounding circular
trajectory. Given different acquisition systems, the lines along which
the integrals are computed follow different geometries:

* parallel. (2D and 3D)
  : Per view, all rays intersecting the object are parallel. In 2D, all rays live on the same plane, perpendicular
    to the axis of rotation.
* fanbeam. (2D)
  : Per view, all rays come from a single source and intersect the object at a certain angle. The detector consists of a 1d line of cells. Similar to
    the 2D “parallel”, all rays live on the same plane, perpendicular to the axis of rotation.
* conebeam. (3D)
  : Per view, all rays come from a single source. The detector consists of a 2D grid of cells. Apart from the central plane, the set of rays coming onto
    a line of cells live on a tilted plane.

#### NOTE
The pseudo-inverse is computed using the filtered back-projection
algorithm with a Ramp filter, and its equivalent for conebeam 3D, the
Feldkamp-Davis-Kress algorithm. This is not the exact linear pseudo-inverse
of the Ray Transform, but it is a good approximation which is robust to noise.

#### NOTE
In the default configuration, reconstruction cells and detector cells are
set to have isotropic unit lengths. The geometry is set to 2D parallel
and matches the default configuration of the [`deepinv.physics.Tomography`](https://deepinv.org/api/stubs/deepinv.physics.Tomography.html.md#deepinv.physics.Tomography) operator with
`circle=False`.

#### NOTE
When the acquisition geometry is already described by `astra` geometry objects,
the operator can be built directly from them with [`from_astra_geometry`](#deepinv.physics.TomographyWithAstra.from_astra_geometry).

#### WARNING
By default, `normalize` is set to `True` if not specified. Initializing the operator without specifying the normalization behavior will issue a warning. Note that normalizing the operator affects the reconstruction dynamics, which may not always be suitable for real-world applications.

#### WARNING
Due to computational efficiency reasons, the projector and backprojector
implemented in `astra` are not matched. The projector is typically ray-driven,
while the backprojector is pixel-driven. The adjoint of the forward Ray Transform
is approximated by rescaling the backprojector.

#### WARNING
The [`deepinv.physics.functional.XrayTransform`](https://deepinv.org/api/stubs/deepinv.physics.functional.XrayTransform.html.md#deepinv.physics.functional.XrayTransform) used in [`deepinv.physics.TomographyWithAstra`](#deepinv.physics.TomographyWithAstra) sequentially processes batch elements, which can make the 2D parallel beam operator significantly slower than its native torch counterpart with [`deepinv.physics.Tomography`](https://deepinv.org/api/stubs/deepinv.physics.Tomography.html.md#deepinv.physics.Tomography) (though still more memory-efficient).

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,*  *...* *]* *,* *None*) – Shape of the object grid, either a 2 or 3-element tuple, for respectively 2D or 3D. If `None`, `object_geometry` and `is_2d` must be specified.
  * **angles** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of angular positions sampled uniformly in `angular_range` or a Tensor containing angular positions in degrees. (default: 180)
  * **n_detector_pixels** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,*  *...* *]* *,* *None*) – In 2D, specify an integer for a single line of detector cells. In 3D, specify a 2-element tuple for (row,col) shape of the detector.  If `None` and `projection_geometry` is specified, `is_2d` must be specified. (default: None)
  * **angular_range** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *]*) – Angular range, defaults to `(0, 180)`.
  * **detector_spacing** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *]*) – In 2D the width of a detector cell. In 3D a 2-element tuple specifying the (vertical, horizontal) dimensions of a detector cell. (default: 1.0)
  * **pixel_spacing** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,*  *...* *]*) – In 2D, the (x,y) dimensions of a pixel in the reconstructed image. In 3D, the (x,y,z) dimensions of a voxel. Scalar value is interpreted as the same dimension along all axes (default: 1.0)
  * **bounding_box** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,*  *...* *]* *,* *None*) – Axis-aligned bounding-box of the reconstruction area [min_x, max_x, min_y, max_y, …]. Optional argument, if specified, overrides argument `object_spacing`. (default: None)
  * **geometry_type** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – The type of geometry among `'parallel'`, `'fanbeam'` in 2D and `'parallel'` and `'conebeam'` in 3D. (default: `'parallel'`)
  * **geometry_parameters** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *]*) – 

    Contains extra parameters specific to certain geometries. When `geometry_type='fanbeam'` or  `'conebeam'`, the dictionary should contains the keys
    - `"source_radius"`: the distance between the x-ray source and the rotation axis, denoted $D_{s0}$, (default: 80.),
    - `"detector_radius"`: the distance between the x-ray detector and the rotation axis, denoted $D_{0d}$. (default: 20.)
  * **geometry_vectors** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* *None*) – 

    Alternative way to describe a 3D geometry. It is a torch.Tensor of shape [num_angles, 12], where for each angular position of index `i` the row consists of a vector of size (12,) with
    - `(sx, sy, sz)`: the position of the source,
    - `(dx, dy, dz)`: the center of the detector,
    - `(ux, uy, uz)`: the horizontal unit vector of the detector,
    - `(vx, vy, vz)`: the vertical unit vector of the detector.

    When specified, `geometry_vectors` overrides `detector_spacing`, `angles` and `geometry_parameters`. It is particularly useful to build the geometry for the [Walnut-CBCT dataset](https://zenodo.org/records/2686726), where the acquisition parameters are provided via such vectors.
  * **object_geometry** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict) *,* *None*) – Pre-created `astra` volume geometry, as returned by `astra.create_vol_geom`. If specified, overrides `img_size`, `pixel_spacing` and `bounding_box`.
  * **projection_geometry** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict) *,* *None*) – Pre-created `astra` projection geometry, as returned by `astra.create_proj_geom`. If specified, overrides `angles`, `n_detector_pixels`, and `geometry_parameters`.
  * **is_2d** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, the operator is 2D, otherwise it is 3D. If `object_geometry` and `projection_geometry` are not specified, this argument is ignored and inferred from the `img_size` argument.
  * **normalize** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True` [`A()`](#deepinv.physics.TomographyWithAstra.A) and [`A_adjoint()`](#deepinv.physics.TomographyWithAstra.A_adjoint) are normalized so that the operator has unit norm. (default: `True`)
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *|* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – The operator only supports CUDA computation. (default: `torch.device('cuda')`)

<hr />

* **Examples:**
  Tomography operator with a 2D `'fanbeam'` geometry, 10 uniformly sampled angles in `[0, 360]`, a detector line of 5 cells with length 2., a source-radius of 20.0 and a detector_radius of 20.0 for 5x5 image:
  ```pycon
  >>> import torch
  >>> if torch.cuda.is_available():
  ...     from deepinv.physics import TomographyWithAstra
  ...     x = torch.randn(1, 1, 5, 5, device='cuda') # Define random 5x5 image
  ...     physics = TomographyWithAstra(
  ...             img_size=(5,5),
  ...             angles=10,
  ...             angular_range=(0, 360),
  ...             n_detector_pixels=5,
  ...             detector_spacing=2.0,
  ...             geometry_type='fanbeam',
  ...             geometry_parameters={
  ...                 'source_radius': 20.,
  ...                 'detector_radius': 20.
  ...             },
  ...             normalize=False
  ...     )
  ...     sinogram = physics(x)
  ...     print(sinogram.shape)
  ... else:
  ...     print(torch.Size([1, 1, 10, 5]))
  torch.Size([1, 1, 10, 5])
  ```

  Tomography operator with a 3D `'conebeam'` geometry, 10 uniformly sampled angles in `[0, 360]`, a detector grid of 5x5 cells of size (2.,2.), a source-radius of 20.0 and a detector_radius of 20.0 for a 5x5x5 volume:
  ```pycon
  >>> if torch.cuda.is_available():
  ...     x = torch.randn(1, 1, 5, 5, 5, device='cuda')  # Define random 5x5x5 volume
  ...     angles = torch.linspace(0, 360, steps=4)[:-1]
  ...     physics = TomographyWithAstra(
  ...            img_size=(5,5,5),
  ...            angles = angles,
  ...            n_detector_pixels=(5,5),
  ...            pixel_spacing=(1.0,1.0,1.0),
  ...            detector_spacing=(2.0,2.0),
  ...            geometry_type='conebeam',
  ...            geometry_parameters={
  ...                 'source_radius': 20.,
  ...                 'detector_radius': 20.
  ...            },
  ...            normalize=False
  ...     )
  ...     sinogram = physics(x)
  ...     print(sinogram.shape)
  ... else:
  ...     print(torch.Size([1, 1, 5, 3, 5]))
  torch.Size([1, 1, 5, 3, 5])
  ```

#### NOTE
This class requires the `astra-toolbox` package to be installed. Install with `pip install astra-toolbox`.

#### A(x, \*\*kwargs)

Forward projection.

In 2D, the output is a sinogram of shape [B,C,A,N],
with A the number of angular positions, and N the number of detector cells.
In 3D, the output is a stack of sinograms of shape [B,C,V,A,N], with A the
number of angular positions, and (V,N) the shape of the 2D detector grid,
where V is the number of rows of the detector and N the number of columns.
:param torch.Tensor x: input of shape [B,C,…,H,W]
:return: projection of shape [B,C,…,A,N]

#### A_adjoint(y, \*\*kwargs)

Approximation of the adjoint.

In 2D, expected input is a sinogram of
shape [B,C,A,N], with A the number of angular positions, and N the number
of detector cells. In 3D, expected input is a stack of sinograms of shape [B,C,V,A,N],
with A the number of angular positions, and (V,N) the shape of the 2D detector grid,
where V is the number of rows of the detector and N the number of columns.

* **Parameters:**
  **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input of shape [B,C,…,A,N]
* **Returns:**
  scaled back-projection of shape [B,C,…,H,W]
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### A_dagger(y, fbp=False, \*\*kwargs)

Computes the solution in $x$ to $y = Ax$ using a least squares solver. A faster approximation can be obtained by setting `fbp=True`, which computes the filtered back-projection of the measurements, or the Feldkamp-Davis-Kress algorithm (FDK) in cone-beam 3D.

#### WARNING
The filtered back-projection algorithm is not the exact linear pseudo-inverse of the Radon transform, but it is a good approximation that is robust to noise.

* **Parameters:**
  **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input of shape [B,C,…,A,N]
* **Returns:**
  filtered back-projection of shape [B,C,…,H,W]
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### *property* angles *: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [None](https://docs.python.org/3.9/library/constants.html#None)*

Astra projection geometry angles tensor in degrees, or `None` for vector geometries.

#### *property* angular_range *: [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[float](https://docs.python.org/3.9/library/functions.html#float), [float](https://docs.python.org/3.9/library/functions.html#float)] | [None](https://docs.python.org/3.9/library/constants.html#None)*

The angular range represented by the X-ray transform in degrees.

#### *property* bounding_box *: [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[float](https://docs.python.org/3.9/library/functions.html#float), ...]*

The reconstruction bounding box represented by the X-ray transform.

#### *property* detector_spacing *: [float](https://docs.python.org/3.9/library/functions.html#float) | [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[float](https://docs.python.org/3.9/library/functions.html#float), [float](https://docs.python.org/3.9/library/functions.html#float)]*

The detector-cell spacing represented by the X-ray transform.

#### fbp_weighting(sinogram)

Scales the computation by the inverse number of views and
object-to-detector cell ratio.

In conebeam 3D, compute FDK weights to correct inflated distances due to
tilted rays. Given coordinate $(x,y)$  of a detector cell, the corresponding
weight is $\omega(x,y) = \frac{D_{s0}}{\sqrt{D_{sd}^2 + x^2 + y^2}}$.

* **Parameters:**
  **sinogram** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Sinogram of shape [B,C,…,A,N].
* **Returns:**
  Weighted sinogram.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### *classmethod* from_astra_geometry(object_geometry, projection_geometry, is_2d, normalize=None, device=torch.device('cuda'), \*\*kwargs)

Build the operator from pre-created `astra` geometries.

Alternative constructor for the cases where `astra` geometries are already available.

#### NOTE
Both geometries must be 3D, even for a 2D acquisition.
For 2D, use one slice and set `is_2d=True`.

* **Parameters:**
  * **object_geometry** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – An `astra` volume geometry, as returned by `astra.create_vol_geom`.
  * **projection_geometry** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – An `astra` projection geometry, as returned by `astra.create_proj_geom`.
  * **is_2d** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Whether the geometries describe a 2D slice or a 3D volume.
  * **normalize** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True` [`A()`](#deepinv.physics.TomographyWithAstra.A) and [`A_adjoint()`](#deepinv.physics.TomographyWithAstra.A_adjoint) are normalized so that the operator has unit norm. (default: `True`)
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *|* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – The operator only supports CUDA computation. (default: `torch.device('cuda')`)
* **Returns:**
  ([`deepinv.physics.TomographyWithAstra`](#deepinv.physics.TomographyWithAstra)) the tomography operator.
* **Return type:**
  [*TomographyWithAstra*](#deepinv.physics.TomographyWithAstra)

#### *property* geometry_type *: [str](https://docs.python.org/3.9/library/stdtypes.html#str)*

The geometry type represented by the X-ray transform.

#### *property* geometry_vectors *: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [None](https://docs.python.org/3.9/library/constants.html#None)*

Astra projection geometry vectors, or `None` for angle geometries.

#### *property* pixel_spacing *: [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[float](https://docs.python.org/3.9/library/functions.html#float), ...]*

The reconstruction-cell spacing represented by the X-ray transform.

<a id="sphx-glr-backref-deepinv-physics-tomographywithastra"></a>

## Examples using `TomographyWithAstra`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the Astra tomography toolbox with deepinv, a popular toolbox for tomography with GPU acceleration.">  <div class="sphx-glr-thumbnail-title">Low-dose CT with ASTRA backend and Total-Variation (TV) prior</div>
</div>
<!-- thumbnail-parent-div-close --></div>
