# NearFieldPtychographyGeometry

### *class* deepinv.physics.NearFieldPtychographyGeometry(wavelength, object_detector_distance, detector_shape, detector_pixel_size)

Bases: [`PtychographyGeometry`](https://deepinv.org/api/stubs/deepinv.physics.PtychographyGeometry.html.md#deepinv.physics.PtychographyGeometry)

Near-field ptychography geometry using same-grid propagation.

Fresnel transfer-function and angular-spectrum propagation keep the sampling
grid, so the object-plane pixel size equals the detector pixel size,
$\Delta_o = \Delta_d$.

* **Parameters:**
  * **wavelength** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – Illumination wavelength $\lambda$, in metres.
  * **object_detector_distance** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – Object-to-detector propagation distance
    $z$, in metres.
  * **detector_shape** ([*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.12/library/functions.html#int) *,* [*int*](https://docs.python.org/3.12/library/functions.html#int) *]*) – Number of detector pixels `(height, width)`.
  * **detector_pixel_size** ([*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.12/library/functions.html#float) *,* [*float*](https://docs.python.org/3.12/library/functions.html#float) *]*) – Effective detector pixel size
    `(dy, dx)` $\Delta_d$, in metres, equal to the object-plane pixel size.
