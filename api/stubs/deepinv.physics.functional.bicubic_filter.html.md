# bicubic_filter

### deepinv.physics.functional.bicubic_filter(factor=2, device='cpu')

Bicubic filter.

It has size (4\*factor, 4\*factor) and is defined as

$$
w(x, y) = \begin{cases}
    (a + 2)|x|^3 - (a + 3)|x|^2 + 1 & \text{if } |x| \leq 1 \\
    a|x|^3 - 5a|x|^2 + 8a|x| - 4a & \text{if } 1 < |x| < 2 \\
    0 & \text{otherwise}
\end{cases}
$$

for $x, y \in {-2\text{factor} + 0.5, -2\text{factor} + 0.5 + 1/\text{factor}, \ldots, 2\text{factor} - 0.5}$.

* **Parameters:**
  * **factor** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – downsampling factor
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – device to put the filter on (cpu or cuda)
