# PtychographyLinearOperator

### *class* deepinv.physics.PtychographyLinearOperator(img_size, probe=None, shifts=None, device='cpu', geometry=None, \*\*kwargs)

Bases: [`LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)

Linear part of the ptychography forward model.

At each scan position, a probe-sized patch of the object is multiplied by the
probe and propagated to the detector, so the object can be larger than the probe.

$$
B = \left[ \begin{array}{c} B_1 \\ B_2 \\ \vdots \\ B_{n_{\text{img}}} \end{array} \right],
B_l = P \text{diag}(p) T_l, \quad l = 1, \dots, n_{\text{img}},
$$

where $p$ is the probe, $T_l$ selects the object region at scan
position $l$, and $P$ propagates the wave to the detector.
Under the far-field (Fraunhofer) approximation, $P=F$, the 2D Fourier transform.

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple)) – Shape `(1, H, W)` of the input object.
  * **probe** (*None* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Probe of shape `(1, H_p, W_p)`, where
    `H_p <= H` and `W_p <= W`.
    Each diffraction pattern has spatial shape `(H_p, W_p)`. If `None`,
    a disk probe is generated with [`deepinv.physics.phase_retrieval.build_probe()`](https://deepinv.org/api/stubs/deepinv.physics.phase_retrieval.build_probe.html.md#deepinv.physics.phase_retrieval.build_probe)
    using the detector shape when `geometry` is provided, or `img_size`
    otherwise.
  * **shifts** (*None* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Scan positions of shape `(n_img, 2)` in
    pixels, in `(row, column)` order. If `None`,
    shifts are generated with [`deepinv.physics.phase_retrieval.generate_shifts()`](https://deepinv.org/api/stubs/deepinv.physics.phase_retrieval.generate_shifts.html.md#deepinv.physics.phase_retrieval.generate_shifts)
    with `n_img=25`.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.12/library/stdtypes.html#str)) – Device `"cpu"` or `"cuda"`.
  * **geometry** (*None* *,* [*deepinv.physics.phase_retrieval.PtychographyGeometry*](https://deepinv.org/api/stubs/deepinv.physics.PtychographyGeometry.html.md#deepinv.physics.PtychographyGeometry)) – Optional
    experimental geometry defining the object-plane pixel size and detector
    sampling. Currently only [`deepinv.physics.phase_retrieval.FarFieldPtychographyGeometry`](https://deepinv.org/api/stubs/deepinv.physics.FarFieldPtychographyGeometry.html.md#deepinv.physics.FarFieldPtychographyGeometry)
    is supported.

#### A(x, \*\*kwargs)

Compute the detector fields $Bx$: extract the patch at each scan
position, multiply by the probe, and apply the 2D FFT.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Object of shape `(batch, 1, H, W)`.
* **Returns:**
  Detector fields of shape `(batch, n_img, H_p, W_p)`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### A_adjoint(y, \*\*kwargs)

Apply the adjoint $B^{    op}$ to detector fields `y`.

* **Parameters:**
  **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Detector fields of shape `(batch, n_img, H_p, W_p)`.
* **Returns:**
  Object of shape `(batch, 1, H, W)`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### extract_patch(x, shifts=None)

Extract the probe-sized object patch $T_l x$ seen at each scan position.

Pixels of a patch that fall outside the object are set to zero.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Object of shape `(batch, 1, H, W)`.
  * **shifts** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Scan positions of shape `(n, 2)` in pixels, in
    `(row, column)` order. Defaults to all scan positions of the operator.
* **Returns:**
  Patches of shape `(batch, n, H_p, W_p)`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### get_overlap_img(shifts)

Total probe intensity at each object pixel, $\sum_l T_l^{\top}|p|^2$,
used to normalise the probe.

* **Parameters:**
  **shifts** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Scan positions of shape `(n, 2)` in pixels.
* **Returns:**
  Overlap image of shape `(1, H, W)`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### place_patch(patches, shifts=None)

Place probe-sized patches back on the object, $\sum_l T_l^{\top} z_l$.

This is the adjoint of [`extract_patch()`](#deepinv.physics.PtychographyLinearOperator.extract_patch): each patch is put back at its
scan position, and overlapping patches are summed.

* **Parameters:**
  * **patches** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Patches of shape `(batch, n, H_p, W_p)`.
  * **shifts** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Scan positions of shape `(n, 2)` in pixels, in
    `(row, column)` order. Defaults to all scan positions of the operator.
* **Returns:**
  Object of shape `(batch, 1, H, W)`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### shift(x, x_shift, y_shift, pad_zeros=True)

Shift `x` by `x_shift` rows and `y_shift` columns.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input tensor.
  * **x_shift** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – Shift along rows (dimension `-2`).
  * **y_shift** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – Shift along columns (dimension `-1`).
  * **pad_zeros** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – If True, pads shifted regions with zeros.
* **Returns:**
  Shifted tensor.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
