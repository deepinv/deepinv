# StructuredRandomPhaseRetrieval

### *class* deepinv.physics.StructuredRandomPhaseRetrieval(img_size, output_size, n_layers, transform='fft', diagonal_mode='uniform_phase', shared_weights=False, dtype=torch.cfloat, device='cpu', \*\*kwargs)

Bases: [`PhaseRetrieval`](https://deepinv.org/api/stubs/deepinv.physics.PhaseRetrieval.html.md#deepinv.physics.PhaseRetrieval)

Structured random phase retrieval model corresponding to the operator

$$
A(x) = |\prod_{i=1}^N (F D_i) x|^2,
$$

where $F$ is the Discrete Fourier Transform (DFT) matrix, and $D_i$ are diagonal matrices with elements of unit norm and random phases, and $N$ refers to the number of layers. It is also possible to replace $x$ with $Fx$ as an additional 0.5 layer.

For oversampling, we first pad the input signal with zeros to match the output shape and pass it to $A(x)$. For undersampling, we first pass the signal in its original shape to $A(x)$ and trim the output signal to match the output shape.

The phase of the diagonal elements of the matrices $D_i$ are drawn from a uniform distribution in the interval $[0, 2\pi]$.

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – shape (C, H, W) of inputs.
  * **output_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – shape (C, H, W) of outputs.
  * **n_layers** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – number of layers $N$. If `layers=N + 0.5`, a first $F$ transform is included, i.e., $A(x)=|\prod_{i=1}^N (F D_i) F x|^2$.
  * **transform** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – structured transform to use. Default is ‘fft’.
  * **diagonal_mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – sampling distribution for the diagonal elements. Default is ‘uniform_phase’.
  * **shared_weights** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if True, the same diagonal matrix is used for all layers. Default is False.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – Signals are processed in dtype. Default is torch.cfloat.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device for computation. Default is `cpu`.

#### *static* get_structure(n_layers)

Returns the structure of the operator as a string.

* **Parameters:**
  **n_layers** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – number of layers.
* **Returns:**
  (str) the structure of the operator, e.g., “FDFD”.
* **Return type:**
  [str](https://docs.python.org/3.9/library/stdtypes.html#str)
