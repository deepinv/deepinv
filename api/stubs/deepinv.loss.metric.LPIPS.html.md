# LPIPS

### *class* deepinv.loss.metric.LPIPS(net_type='alex', device=None, check_input_range=True, \*\*kwargs)

Bases: [`Metric`](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric)

Learned Perceptual Image Patch Similarity (LPIPS) metric.

Calculates the LPIPS $\text{LPIPS}(\hat{x},x)$ where $\hat{x}=\inverse{y}$.

Computes the perceptual similarity between two images, based on a pre-trained deep neural network.
Uses implementation from [torchmetrics](https://lightning.ai/docs/torchmetrics/stable/image/learned_perceptual_image_patch_similarity.html).

The inputs `x_net`, `x` must both have 3 channels and be in `[0, 1]`. Optionally use `norm_inputs` argument to clip to `[0, 1]`.

#### NOTE
By default, no reduction is performed in the batch dimension.

* **Example:**

```default
from deepinv.utils import load_example
from deepinv.loss.metric import LPIPS
m = LPIPS()
x = torch.ones(2, 3, 32, 32)
x_net = x - 0.01
m(x_net, x)
```

* **Parameters:**
  * **net_type** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – network architecture to use. Options: ‘alex’, ‘vgg’, ‘squeeze’. Default: ‘alex’.
  * **complex_abs** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – perform complex magnitude before passing data to metric function. If `True`,
    the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
  * **reduction** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – a method to reduce metric score over individual batch scores. `mean`: takes the mean, `sum` takes the sum, `none` or None no reduction will be applied (default).
  * **norm_inputs** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – normalize images before passing to metric. `l2` normalizes by $\ell_2$ spatial norm, `min_max` normalizes by min and max of each input.
  * **check_input_range** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if True, raise error if inputs aren’t in the appropriate range `[0, 1]`.
  * **center_crop** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]* *,* *None*) – If not `None` (default), center crop the tensor(s) before computing the metrics.
    If an `int` is provided, the cropping is applied equally on all spatial dimensions (by default, all dimensions except the first two).
    If `tuple` of `int`, cropping is performed over the last `len(center_crop)` dimensions. If positive values are provided, a standard center crop is applied.
    If negative (or zero) values are passed, cropping will be done by removing `center_crop` pixels from the borders (useful when tensors vary in size across the dataset).
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – LPIPS net device.
