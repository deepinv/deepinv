# GMSD

### *class* deepinv.loss.metric.GMSD(c=0.0026, \*\*kwargs)

Bases: [`Metric`](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric)

Gradient Magnitude Similarity Deviation (GMSD) metric.

Calculates $\text{GMSD}(\hat{x},x)$, as proposed in Xue *et al.*<sup>[1](#footcite-xue2013gradient)</sup>.
GMSD measures perceptual image quality from the standard deviation of the pixel-wise gradient magnitude
similarity (GMS) map. A lower value indicates better quality.

The reference and reconstructed gradient magnitudes $m_x$ and $m_{\hat{x}}$ are computed using the
Prewitt operators $h_x, h_y$,

$$
m_x = \sqrt{(x \ast h_x)^2 + (x \ast h_y)^2},\qquad
m_{\hat{x}} = \sqrt{(\hat{x} \ast h_x)^2 + (\hat{x} \ast h_y)^2},

$$

where $\ast$ denotes 2D convolution. The pixel-wise GMS map is

$$
\text{GMS}(\hat{x},x) = \frac{2 m_x m_{\hat{x}} + c}{m_x^2 + m_{\hat{x}}^2 + c},

$$

where c is a small stability constant.
GMSD is the spatial standard deviation of this map computed independently for each image in the batch.
For multi-channel images, GMSD is calculated independently for each channel, then averaged across channels to produce a single scalar per image.

* **Example:**

```pycon
>>> import torch
>>> from deepinv.loss.metric import GMSD
>>> m = GMSD()
>>> x_net = x = torch.ones(3, 1, 8, 8) # B,1,H,W
>>> m(x_net, x)
tensor([0., 0., 0.])
```

* **Parameters:**
  * **c** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – positive stability constant $c$ of the GMS map. The original paper uses $c=170$
    for images in $[0, 255]$. Rescale accordingly for images in $[0, 1]$. Default: 0.0026.
  * **complex_abs** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – perform complex magnitude before passing data to metric function. If `True`,
    the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
  * **reduction** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – a method to reduce metric score over individual batch scores. `mean`: takes the mean, `sum` takes the sum, `none` or None no reduction will be applied (default).
  * **norm_inputs** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – normalize images before passing to metric. `l2` normalizes by $\ell_2$ spatial norm, `min_max` normalizes by min and max of each input.
  * **center_crop** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]* *,* *None*) – If not `None` (default), center crop the tensor(s) before computing the metrics.
    If an `int` is provided, the cropping is applied equally on all spatial dimensions (by default, all dimensions except the first two).
    If `tuple` of `int`, cropping is performed over the last `len(center_crop)` dimensions. If positive values are provided, a standard center crop is applied.
    If negative (or zero) values are passed, cropping will be done by removing `center_crop` pixels from the borders (useful when tensors vary in size across the dataset).

<hr />

* **References:**

* <a id='footcite-xue2013gradient'>**[1]**</a> Wufeng Xue, Lei Zhang, Xuanqin Mou, and Alan C Bovik. Gradient magnitude similarity deviation: a highly efficient perceptual image quality index. *IEEE transactions on image processing*, 23(2):684–695, 2013.
