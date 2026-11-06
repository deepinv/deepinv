# Ptychography

### *class* deepinv.physics.Ptychography(img_size=None, probe=None, shifts=None, device='cpu', geometry=None, \*\*kwargs)

Bases: [`PhaseRetrieval`](https://deepinv.org/api/stubs/deepinv.physics.PhaseRetrieval.md#deepinv.physics.PhaseRetrieval)

Ptychography forward operator

$$
\forw{x} = \left| Bx \right|^2,
$$

where $B$ is the linear operator
[`deepinv.physics.PtychographyLinearOperator`](https://deepinv.org/api/stubs/deepinv.physics.PtychographyLinearOperator.md#deepinv.physics.PtychographyLinearOperator).

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple)) – Shape `(1, H, W)` of the input object.
  * **probe** (*None* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Probe of shape `(1, H_p, W_p)`. Its
    spatial shape determines the diffraction-pattern shape and may be
    smaller than the object. If `None`, a disk probe is generated.
  * **shifts** (*None* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Scan positions of shape `(n_img, 2)` in
    pixels, in `(row, column)` order. If `None`, shifts are generated with
    [`deepinv.physics.phase_retrieval.generate_shifts()`](https://deepinv.org/api/stubs/deepinv.physics.phase_retrieval.generate_shifts.md#deepinv.physics.phase_retrieval.generate_shifts).
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.12/library/stdtypes.html#str)) – Device `"cpu"` or `"cuda"`.
  * **geometry** (*None* *,* [*deepinv.physics.phase_retrieval.PtychographyGeometry*](https://deepinv.org/api/stubs/deepinv.physics.PtychographyGeometry.md#deepinv.physics.PtychographyGeometry)) – Optional
    experimental geometry defining the object-plane pixel size and detector
    sampling. Currently only [`deepinv.physics.phase_retrieval.FarFieldPtychographyGeometry`](https://deepinv.org/api/stubs/deepinv.physics.FarFieldPtychographyGeometry.md#deepinv.physics.FarFieldPtychographyGeometry)
    is supported. If `None`, shifts and probe are in pixels.

<hr />

* **Examples:**

```pycon
>>> from deepinv.physics import FarFieldPtychographyGeometry, Ptychography
>>> import torch
>>> img_size = (1, 64, 64)  # object shape
>>> detector_shape = (32, 32)  # probe and diffraction-pattern shape
>>> geometry = FarFieldPtychographyGeometry(
...     wavelength=632.8e-9,
...     object_detector_distance=5e-2,
...     detector_shape=detector_shape,
...     detector_pixel_size=(36e-6, 36e-6),
... )
>>> physics = Ptychography(img_size=img_size, geometry=geometry)
>>> physics.geometry is geometry
True
>>> x = torch.randn(img_size, dtype=torch.cfloat)
>>> y = physics(x)  # Apply the Ptychography forward operator
>>> print(y.shape)  # 25 probe positions by default
torch.Size([1, 25, 32, 32])
```

#### *property* geometry *: [PtychographyGeometry](https://deepinv.org/api/stubs/deepinv.physics.PtychographyGeometry.md#deepinv.physics.PtychographyGeometry) | [None](https://docs.python.org/3.12/library/constants.html#None)*

Physical geometry associated with the linear ptychography operator.

<a id="sphx-glr-backref-deepinv-physics-ptychography"></a>

## Examples using `Ptychography`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Ptychography is a coherent imaging technique for recovering a complex-valued object, including both its amplitude and phase, from intensity-only measurements. A localized illumination, called the probe, is scanned across overlapping regions of the object. At each position, a detector records the intensity after the exit wave has propagated from the object plane. The overlap provides redundant measurements of the same object regions, making it possible to recover phase information that is not measured directly by the detector.">![](auto_examples/physics/images/thumb/sphx_glr_demo_ptychography_thumb.png)

[Ptychography phase retrieval](https://deepinv.org/auto_examples/physics/demo_ptychography.md)

  <div class="sphx-glr-thumbnail-title">Ptychography phase retrieval</div>
</div>
<!-- thumbnail-parent-div-close --></div>
