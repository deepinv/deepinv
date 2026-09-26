# sinc_filter

### deepinv.physics.functional.sinc_filter(factor=2, length=11, windowed=True, device='cpu')

Anti-aliasing sinc filter, optionally multiplied by a Kaiser window.

The kaiser window parameter is computed as follows:

$$
A = 2.285 \cdot (L - 1) \cdot 3.14 \cdot \Delta f + 7.95
$$

where $\Delta f = 2 (2 - \sqrt{2}) / \text{factor}$. Then, the beta parameter is computed as:

$$
\beta = \begin{cases}
    0 & \text{if } A \leq 21 \\
    0.5842 \cdot (A - 21)^{0.4} + 0.07886 \cdot (A - 21) & \text{if } 21 < A \leq 50 \\
    0.1102 \cdot (A - 8.7) & \text{otherwise}
\end{cases}
$$

* **Parameters:**
  * **factor** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Downsampling factor. If Tensor, can only have one element.
  * **length** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Length of the filter.
  * **windowed** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Whether to multiply by Kaiser window.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – device to put the filter on (cpu or cuda)
