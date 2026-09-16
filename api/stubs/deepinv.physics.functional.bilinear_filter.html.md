# bilinear_filter

### deepinv.physics.functional.bilinear_filter(factor=2, device='cpu')

Bilinear filter.

It has size (2\*factor, 2\*factor) and is defined as

$$
w(x, y) = \begin{cases}
    (1 - |x|) \cdot (1 - |y|) & \text{if } |x| \leq 1 \text{ and } |y| \leq 1 \\
    0 & \text{otherwise}
\end{cases}
$$

for $x, y \in {-\text{factor} + 0.5, -\text{factor} + 0.5 + 1/\text{factor}, \ldots, \text{factor} - 0.5}$.

* **Parameters:**
  * **factor** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – downsampling factor
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – device to put the filter on (cpu or cuda)
