# NIMA

### *class* deepinv.loss.metric.NIMA(variant='aesthetic', weights_path='download', max_pixel=1.0, device='cpu', \*\*kwargs)

Bases: [`Metric`](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric)

Neural Image Assessment (NIMA) metric.

Calculates the NIMA score $\text{NIMA}(\hat{x})$ where $\hat{x}=\inverse{y}$.
It is a no-reference image quality metric introduced by Talebi and Milanfar<sup>[1](#footcite-talebi2018nima)</sup>,
which predicts the distribution of human opinion scores an image would receive.

A convolutional network outputs a probability $p_i$ for each of the 10 score bins,
and the metric returns the mean opinion score

$$
\text{NIMA}(\hat{x}) = \sum_{i=1}^{10} i \, p_i \in [1, 10],
$$

where higher is better. Use [`distribution`](#deepinv.loss.metric.NIMA.distribution)
to obtain the full predicted distribution, whose spread indicates how much raters would disagree.

Two pre-trained heads are available, selected with `variant`:

- `'aesthetic'` (default), trained on the AVA dataset [[103](https://deepinv.org/user_guide/other/biblio.html.md#id150)], which rates the aesthetic appeal of an image;
- `'technical'`, trained on the TID2013 dataset [[113](https://deepinv.org/user_guide/other/biblio.html.md#id149)], which rates the amount of distortion in an image

This is adapted from the `image-quality-assessment` implementation in
([https://github.com/idealo/image-quality-assessment](https://github.com/idealo/image-quality-assessment)), which we gratefully acknowledge.
The MobileNet backbone and both heads use their released weights, converted to PyTorch.

#### WARNING
The network expects $224\times 224$ inputs, so images are bilinearly resized before
being scored, without preserving the aspect ratio, as in the original implementation.

#### NOTE
Single-channel images are replicated over three channels, as the network expects RGB input.

#### NOTE
By default, no reduction is performed in the batch dimension.

* **Parameters:**
  * **variant** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – which pre-trained head to use, either `'aesthetic'` or `'technical'`. Default: `'aesthetic'`.
  * **weights_path** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path) *,* *None*) – path to the network weights. If `'download'` (default),
    the weights of the chosen `variant` are downloaded.
  * **max_pixel** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – maximum pixel value of the input images, used to rescale them to the
    $[-1, 1]$ range expected by the network. Default: 1.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – device on which the network is stored. Default: `'cpu'`.
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
>>> from deepinv.loss.metric import NIMA
>>> m = NIMA()
>>> x_net = torch.rand(2, 3, 64, 64)  # batch of 2 RGB images in [0, 1]
>>> m(x_net).shape
torch.Size([2])
```

<hr />

* **References:**

* <a id='footcite-talebi2018nima'>**[1]**</a> Hossein Talebi and Peyman Milanfar. Nima: neural image assessment. *IEEE Transactions on Image Processing*, 27(8):3998–4011, 2018.

#### distribution(x_net)

Predict the distribution of human opinion scores of a batch of images.

Resizes to the network input size and rescale to $[-1, 1]$.

* **Parameters:**
  **x_net** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – `(B, C, H, W)` input tensors with C=1 or 3 channels.
* **Returns:**
  `(B, 10)` tensor of probabilities, where entry $i$ is the predicted
  probability that a rater would give the image a score of $i+1$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### metric(x_net, \*args, \*\*kwargs)

Compute the mean opinion score of a batch of images.

* **Parameters:**
  **x_net** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – `(B, C, H, W)` input tensors with C=1 or 3 channels.
* **Returns:**
  `(B,)` tensor of NIMA scores, between 1 and 10, higher is better.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
