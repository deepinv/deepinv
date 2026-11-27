# PtychographyGeometry

### *class* deepinv.physics.PtychographyGeometry(wavelength, object_detector_distance, detector_shape, detector_pixel_size)

Bases: [`ABC`](https://docs.python.org/3.12/library/abc.html#abc.ABC)

Physical geometry of a ptychography setup.

* **Parameters:**
  * **wavelength** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – Illumination wavelength $\lambda$, in metres.
  * **object_detector_distance** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – Object-to-detector distance $z$,
    in metres.
  * **detector_shape** ([*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.12/library/functions.html#int) *,* [*int*](https://docs.python.org/3.12/library/functions.html#int) *]*) – Number of detector pixels `(height, width)`.
  * **detector_pixel_size** ([*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.12/library/functions.html#float) *,* [*float*](https://docs.python.org/3.12/library/functions.html#float) *]*) – Effective detector pixel size
    `(dy, dx)` $\Delta_d$, in metres.

#### *abstract property* object_pixel_size *: [tuple](https://docs.python.org/3.12/library/stdtypes.html#tuple)[[float](https://docs.python.org/3.12/library/functions.html#float), [float](https://docs.python.org/3.12/library/functions.html#float)]*

Pixel size in the object plane.

#### positions_to_shifts(positions, origin='center')

Convert scan positions in metres to integer pixel shifts, by dividing by
[`object_pixel_size`](#deepinv.physics.PtychographyGeometry.object_pixel_size) and rounding.

* **Parameters:**
  * **positions** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Scan positions in metres, of shape `(N, 2)`
    ordered `(row, column)`.
  * **origin** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Reference position subtracted before
    conversion, in metres. `"center"` subtracts the mean position, so shifts
    are centred on the scan. Pass a `(2,)` tensor to use another reference point.
* **Returns:**
  Integer shifts of shape `(N, 2)`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-physics-ptychographygeometry"></a>

## Examples using `PtychographyGeometry`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Ptychography is a coherent imaging technique for recovering a complex-valued object, including both its amplitude and phase, from intensity-only measurements. A localized illumination, called the probe, is scanned across overlapping regions of the object. At each position, a detector records the intensity after the exit wave has propagated from the object plane. The overlap provides redundant measurements of the same object regions, making it possible to recover phase information that is not measured directly by the detector.">![](auto_examples/physics/images/thumb/sphx_glr_demo_ptychography_thumb.png)

[Ptychography phase retrieval](https://deepinv.org/auto_examples/physics/demo_ptychography.md)

  <div class="sphx-glr-thumbnail-title">Ptychography phase retrieval</div>
</div>
<!-- thumbnail-parent-div-close --></div>
