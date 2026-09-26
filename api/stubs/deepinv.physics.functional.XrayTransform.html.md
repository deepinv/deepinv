# XrayTransform

### *class* deepinv.physics.functional.XrayTransform(projection_geometry, object_geometry, is_2d=False)

Bases: [`object`](https://docs.python.org/3.9/library/functions.html#object)

X-ray Transform operator with `astra-toolbox` backend.

Uses the [astra-toolbox](https://astra-toolbox.com/) to implement a ray-driven forward projector
and a pixel-driven backprojector ([`XrayTransform.T`](#deepinv.physics.functional.XrayTransform.T)).
This class leverages the GPULink functionality of `astra` to share the underlying
CUDA memory between torch Tensors and CUDA-based arrays use in `astra`. The
functionality is only implemented for 3D arrays, thus the underlying transforms
are all 3D operators. For 2D transforms, the object is set to a flat volume with only 1 voxel depth.

#### NOTE
This transform does not handle batched and multi-channel inputs. It is
handled by a custom [`torch.autograd.Function`](https://docs.pytorch.org/docs/stable/autograd.html#torch.autograd.Function) that wraps the [`XrayTransform`](#deepinv.physics.functional.XrayTransform).
To handle standard PyTorch pipelines, [`XrayTransform`](#deepinv.physics.functional.XrayTransform) is instantiated inside a [`deepinv.physics.TomographyWithAstra`](https://deepinv.org/api/stubs/deepinv.physics.TomographyWithAstra.html.md#deepinv.physics.TomographyWithAstra) operator.

* **Parameters:**
  * **projection_geometry** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *Any* *]*) – Dictionary containing the parameters of the projection geometry in the format produced by `astra.create_proj_geom()`. It is passed to the `astra.create_projector()` function to instantiate the projector.
  * **object_geometry** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *Any* *]*) – Dictionary containing the parameters of the object geometry in the format produced by `astra.create_vol_geom()`. It is passed to the `astra.create_projector()` function to instantiate the projector.
  * **is_2d** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Specifies if the geometry is flat (2D) or describe a real 3D reconstruction setup.

#### NOTE
This class requires the `astra-toolbox` package to be installed. Install with `pip install astra-toolbox`.

#### *property* T

Implements and returns the adjoint of the transform operator.

#### *property* detector_cell_area *: [float](https://docs.python.org/3.9/library/functions.html#float)*

The surface in physical units of a detector pixel.

#### *property* detector_cell_u_length *: [float](https://docs.python.org/3.9/library/functions.html#float)*

The horizontal length of a detector cell.

#### *property* detector_cell_v_length *: [float](https://docs.python.org/3.9/library/functions.html#float)*

The vertical length of a detector cell.

#### *property* detector_radius *: [float](https://docs.python.org/3.9/library/functions.html#float)*

The distance between the center of the detector and the axis of rotation.

#### *property* domain_shape *: [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)*

The shape of the input volume.

#### *property* magnification_factor *: [float](https://docs.python.org/3.9/library/functions.html#float)*

The magnification factor induced by the fan/cone geometry.

#### *property* object_cell_volume *: [float](https://docs.python.org/3.9/library/functions.html#float)*

The volume in physical units of a voxel.

#### *property* range_shape *: [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)*

The shape of the output projection.

#### *property* source_radius *: [float](https://docs.python.org/3.9/library/functions.html#float)*

The distance between the source and the axis of rotation.
