# BRISQUE

### *class* deepinv.loss.metric.BRISQUE(weights_path='download', max_pixel=1.0, device='cpu', dtype=torch.float32, \*\*kwargs)

Bases: [`Metric`](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric)

Blind/Referenceless Image Spatial QUality Evaluator (BRISQUE) metric.

Calculates the BRISQUE score $\text{BRISQUE}(\hat{x})$ where $\hat{x}=\inverse{y}$.
It is a no-reference image quality metric introduced by Mittal *et al.*<sup>[1](#footcite-mittal2012no)</sup>,
which quantifies how far an image departs from the natural scene statistics of
pristine natural images. Lower is better, with scores roughly in $[0, 100]$.

BRISQUE works with images of 1 or 3 channels. If the image has 3 channels,
it is assumed to be RGB and converted to relative luminance, then, at two scales, the mean
subtracted contrast normalized (MSCN) coefficients

$$
\hat{x}_{ij} = \frac{x_{ij} - \mu_{ij}}{\sigma_{ij} + 1}
$$

are computed with a $7\times 7$ Gaussian window. A generalized Gaussian is fitted
to the MSCN coefficients and asymmetric generalized Gaussians are fitted to their four
neighbouring products, yielding 36 features which are mapped to a quality score by a
support vector regressor pre-trained on the [LIVE IQA dataset](https://live.ece.utexas.edu/research/quality/subjective.htm).

This is a PyTorch translation of the implementation
([https://github.com/dsoellinger/blind_image_quality_toolbox](https://github.com/dsoellinger/blind_image_quality_toolbox)). The pre-trained support vector
regressor is the one released with the original MATLAB implementation, and weights
were downloaded from [https://github.com/dsoellinger/blind_image_quality_toolbox/blob/master/%2Bbrisque/allmodel](https://github.com/dsoellinger/blind_image_quality_toolbox/blob/master/%2Bbrisque/allmodel).

#### NOTE
The features and the regressor were fitted on images in the $[0, 255]$ range.
Inputs are internally rescaled from `[0, max_pixel]` to $[0, 255]$,
so make sure `max_pixel` matches the intensity scale of your data.

#### NOTE
By default, no reduction is performed in the batch dimension.

* **Parameters:**
  * **weights_path** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path) *,* *None*) – path to the support vector regressor weights.
    If `'download'` (default), the weights released with the original implementation are downloaded.
  * **max_pixel** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – maximum pixel value of the input images, used to rescale them to
    the $[0, 255]$ range expected by the regressor. Default: 1.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – device on which the regressor weights are stored. Default: `'cpu'`.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – dtype used for the feature computation (the regressor is always evaluated
    in `float64`). Default: `torch.float32`.
  * **complex_abs** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – perform complex magnitude before passing data to metric function. If `True`,
    the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
  * **reduction** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – a method to reduce metric score over individual batch scores. `mean`: takes the mean, `sum` takes the sum, `none` or None no reduction will be applied (default).
  * **norm_inputs** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – normalize images before passing to metric. `l2` normalizes by $\ell_2$ spatial norm, `min_max` normalizes by min and max of each input.
  * **center_crop** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]* *,* *None*) – If not `None` (default), center crop the tensor(s) before computing the metrics.
    If an `int` is provided, the cropping is applied equally on all spatial dimensions (by default, all dimensions except the first two).
    If `tuple` of `int`, cropping is performed over the last `len(center_crop)` dimensions. If positive values are provided, a standard center crop is applied.
    If negative (or zero) values are passed, cropping will be done by removing `center_crop` pixels from the borders (useful when tensors vary in size across the dataset).

<hr />

* **Example:**

```pycon
>>> from deepinv.loss.metric import BRISQUE
>>> m = BRISQUE()
>>> x_net = torch.rand(2, 3, 32, 32)  # batch of 2 RGB images in [0, 1]
>>> m(x_net).shape
torch.Size([2])
```

<hr />

* **References:**

* <a id='footcite-mittal2012no'>**[1]**</a> Anish Mittal, Anush Krishna Moorthy, and Alan Conrad Bovik. No-reference image quality assessment in the spatial domain. *IEEE Transactions on Image Processing*, 21(12):4695–4708, 2012.

#### estimate_aggd_param(vecs, eps=1e-12)

Fit an asymmetric generalized Gaussian distribution to each row by moment matching.

* **Parameters:**
  * **vecs** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – `(B, N)` tensor of samples.
  * **eps** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – stabilizer used in the denominators.
* **Returns:**
  tuple of `(B,)` tensors with the shape parameter, the left and the right standard deviations.
* **Return type:**
  [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]

#### estimate_ggd_param(vecs, eps=1e-12)

Fit a generalized Gaussian distribution to each row by moment matching.

* **Parameters:**
  * **vecs** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – `(B, N)` tensor of samples.
  * **eps** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – stabilizer used in the denominators.
* **Returns:**
  tuple of `(B,)` tensors with the shape parameter and the standard deviation.
* **Return type:**
  [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]

#### features(x_net)

Compute the 36 natural scene statistics features of BRISQUE.

* **Parameters:**
  **x_net** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – `(B, 1, H, W)` single-channel images in the `[0, 255]` range.
* **Returns:**
  `(B, 36)` tensor of features.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### metric(x_net, \*args, \*\*kwargs)

Compute the BRISQUE score for a batch of images.

* **Parameters:**
  **x_net** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – `(B, C, H, W)` input tensors with C=1 or 3 channels.
* **Returns:**
  `(B,)` tensor of BRISQUE scores.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
