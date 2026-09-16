# PtychographyLinearOperator

### *class* deepinv.physics.PtychographyLinearOperator(img_size, probe=None, shifts=None, device='cpu', \*\*kwargs)

Bases: [`LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)

Forward linear operator for phase retrieval in ptychography.

Models multiple applications of the shifted probe and Fourier transform on an input image.

This operator performs multiple 2D Fourier transforms on the probe function applied to the shifted input image according to specific offsets, and concatenates them.
The probe function is applied element by element to the input image.

$$
B = \left[ \begin{array}{c} B_1 \\ B_2 \\ \vdots \\ B_{n_{\text{img}}} \end{array} \right],
B_l = F \text{diag}(p) T_l, \quad l = 1, \dots, n_{\text{img}},
$$

where $F$ is the 2D Fourier transform, $\text{diag}(p)$ is associated with the probe $p$ and $T_l$ is a 2D shift.

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – Shape of the input image (height, width).
  * **probe** (*None* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – A tensor of shape `img_size` representing the probe function. If `None`, a disk probe is generated with [`deepinv.physics.phase_retrieval.build_probe()`](https://deepinv.org/api/stubs/deepinv.physics.phase_retrieval.build_probe.html.md#deepinv.physics.phase_retrieval.build_probe) with disk shape and radius 10.
  * **shifts** (*None* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – A 2D array of shape `(N, 2)` corresponding to the `N` shift positions for the probe. If `None`, shifts are generated with [`deepinv.physics.phase_retrieval.generate_shifts()`](https://deepinv.org/api/stubs/deepinv.physics.phase_retrieval.generate_shifts.html.md#deepinv.physics.phase_retrieval.generate_shifts) with `N=25`.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device “cpu” or “gpu”.

#### A(x, \*\*kwargs)

Applies the forward operator to the input image `x` by shifting the probe,
multiplying element-wise, and performing a 2D Fourier transform.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input image tensor.
* **Returns:**
  Concatenated Fourier transformed tensors after applying shifted probes.

#### A_adjoint(y, \*\*kwargs)

Applies the adjoint operator to `y`.

* **Parameters:**
  **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Transformed image data tensor of size (batch_size, n_img, height, width).
* **Returns:**
  Reconstructed image tensor.

#### get_overlap_img(shifts)

Computes the overlapping image intensities from probe shifts, used for normalization.

* **Parameters:**
  **shifts** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Tensor of probe shifts.
* **Returns:**
  Tensor representing the overlap image.

#### shift(x, x_shift, y_shift, pad_zeros=True)

Applies a shift to the tensor `x` by `x_shift` and `y_shift`.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input tensor.
  * **x_shift** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Shift in x-direction.
  * **y_shift** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Shift in y-direction.
  * **pad_zeros** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If True, pads shifted regions with zeros.
* **Returns:**
  Shifted tensor.
