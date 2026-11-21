# FarFieldPtychographyGeometry

### *class* deepinv.physics.FarFieldPtychographyGeometry(wavelength, object_detector_distance, detector_shape, detector_pixel_size)

Bases: [`PtychographyGeometry`](https://deepinv.org/api/stubs/deepinv.physics.PtychographyGeometry.html.md#deepinv.physics.PtychographyGeometry)

Fraunhofer ptychography geometry.

The object-plane pixel size is determined from the detector sampling by

$$
\Delta_o = \frac{\lambda z}{N \Delta_d},
$$

where $N$ is the number of detector pixels along that dimension.

* **Parameters:**
  * **wavelength** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – Illumination wavelength $\lambda$, in metres.
  * **object_detector_distance** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – Object-to-detector distance $z$,
    in metres.
  * **detector_shape** ([*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.12/library/functions.html#int) *,* [*int*](https://docs.python.org/3.12/library/functions.html#int) *]*) – Number of detector pixels `(height, width)`,
    i.e. $N$ along each spatial dimension. Must match the spatial shape of the
    probe and diffraction patterns.
  * **detector_pixel_size** ([*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.12/library/functions.html#float) *,* [*float*](https://docs.python.org/3.12/library/functions.html#float) *]*) – Effective detector pixel size
    `(dy, dx)` $\Delta_d$, in metres.

<hr />

* **Examples:**
  Object-plane pixel size from the detector sampling:
  ```pycon
  >>> import torch
  >>> from deepinv.physics import FarFieldPtychographyGeometry
  >>> geometry = FarFieldPtychographyGeometry(
  ...     wavelength=1e-9, object_detector_distance=1.0,
  ...     detector_shape=(100, 100), detector_pixel_size=(1e-6, 1e-6),
  ... )
  >>> geometry.object_pixel_size
  (1e-05, 1e-05)
  ```

  Convert scan positions from metres to pixel shifts, centred on the scan:
  ```pycon
  >>> positions = torch.tensor([[0.0, 0.0], [0.0, 2e-5]])  # 20 um apart in x
  >>> geometry.positions_to_shifts(positions)
  tensor([[ 0, -1],
          [ 0,  1]], dtype=torch.int32)
  ```

<a id="sphx-glr-backref-deepinv-physics-farfieldptychographygeometry"></a>

## Examples using `FarFieldPtychographyGeometry`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Ptychography is a coherent imaging technique for recovering a complex-valued object, including both its amplitude and phase, from intensity-only measurements. A localized illumination, called the probe, is scanned across overlapping regions of the object. At each position, a detector records the intensity after the exit wave has propagated from the object plane. The overlap provides redundant measurements of the same object regions, making it possible to recover phase information that is not measured directly by the detector.">![](auto_examples/physics/images/thumb/sphx_glr_demo_ptychography_thumb.png)

[Ptychography phase retrieval](https://deepinv.org/auto_examples/physics/demo_ptychography.html.md)

  <div class="sphx-glr-thumbnail-title">Ptychography phase retrieval</div>
</div>
<!-- thumbnail-parent-div-close --></div>
