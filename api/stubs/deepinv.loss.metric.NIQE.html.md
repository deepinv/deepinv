# NIQE

### *class* deepinv.loss.metric.NIQE(weights_path='download', denominator=1, round_tensor=False, patch_size=96, patch_overlap=0, device='cpu', dtype=torch.float32, \*\*kwargs)

Bases: [`Metric`](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric)

Natural Image Quality Evaluator (NIQE) metric.

Calculates the NIQE $\text{NIQE}(\hat{x})$ where $\hat{x}=\inverse{y}$.
It is a no-reference image quality metric that estimates the quality of images.

This implementation is based on the original Matlab code (available at [http://live.ece.utexas.edu/research/quality/niqe_release.zip](http://live.ece.utexas.edu/research/quality/niqe_release.zip)).
One exception is that the original code always converted the image to float64. This implementation converts
to dtype specified at init, but always use float64 when calculating the pseudoinverse.

#### NOTE
The input image must be sufficiently large compared to `patch_size` to ensure an adequate number of
patches can be extracted. NIQE fits a Multivariate Gaussian (MVG) model to
the Natural Scene Statistics (NSS) features (MSCN coefficients) of these
patches, then measures the distance between this model and a reference MVG
pre-fitted on pristine natural images. Too few patches yield an unreliable
covariance matrix estimate, degrading the accuracy of the quality score.

#### NOTE
`denominator` defaults to 1. This was used in the original work, with fitting and testing data in [0,255]. When working with
another intensity scale, change `denominator` appropriately to ensure it doesn’t dominate over σ. For example, `denominator=1/255`
is a good starting point for intensity scale [0,1].

#### NOTE
By default, no reduction is performed in the batch dimension.

* **Parameters:**
  * **weights_path** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Path to weights created with `.create_weights`. If ‘download’ (default), downloads the weights provided by Mittal *et al.*<sup>[1](#footcite-mittal2012making)</sup>. If None, mu and cov are not initialized (useful when fitting custom weights).
  * **denominator** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – stabilizer to add to the std in the image normalization step (eq.1). Defaults to 1
  * **round_tensor** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to round the input. The original NIQE implementation used rounding and requires input to be range [0, 255]. Do not set round_tensor if incoming tensors will be in [0,1] style ranges. Defaults to False.
  * **patch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – spatial size of the square patches used to compute NSS features. Larger values yield more
    robust per-patch statistics but require larger inputs and produce fewer patches. Defaults to 96.
  * **patch_overlap** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of pixels overlapped between adjacent patches (stride is `patch_size - patch_overlap`).
    Increase to extract more patches from a given image. Defaults to 0.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – device to use for the metric computation. Default: ‘cpu’.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – dtype used for the metric computation (the pseudoinverse is always computed in float64). Default: `torch.float32`.
  * **complex_abs** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – perform complex magnitude before passing data to metric function. If `True`,
    the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
  * **reduction** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – a method to reduce metric score over individual batch scores. `mean`: takes the mean, `sum` takes the sum, `none` or None no reduction will be applied (default).
  * **norm_inputs** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – normalize images before passing to metric. `l2``normalizes by L2 spatial norm, ``min_max` normalizes by min and max of each input.
  * **center_crop** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]* *,* *None*) – If not `None` (default), center crop the tensor(s) before computing the metrics.
    If an `int` is provided, the cropping is applied equally on all spatial dimensions (by default, all dimensions except the first two).
    If `tuple` of `int`, cropping is performed over the last `len(center_crop)` dimensions. If positive values are provided, a standard center crop is applied.
    If negative (or zero) values are passed, cropping will be done by removing `center_crop` pixels from the borders (useful when tensors vary in size across the dataset).

<hr />

* **References:**

* <a id='footcite-mittal2012making'>**[1]**</a> Anish Mittal, Rajiv Soundararajan, and Alan C Bovik. Making a “completely blind” image quality analyzer. *IEEE Signal processing letters*, 20(3):209–212, 2012.

#### create_weights(dataset, sharpness_threshold=0.75, save_path=None)

Fit NIQE model parameters (mu_prisparam, cov_prisparam) from a dataset of ‘pristine’ images,
following the original MATLAB pipeline with two scales and sharpness-based patch selection.
`patch_size`, `patch_overlap`, and `denominator` used are those passed at init (unless modified post-init by user).

`dataset` should yield a (C, H, W) `Tensor` or a dictionary with key `"x"` containing such image, where C=1 and C=3 are allowed. If C=3, RGB is assumed and will be converted
to greyscale using 0.299\*R + 0.587\*G + 0.114\*B.

* **Parameters:**
  * **dataset** ([*torch.utils.data.Dataset*](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)) – for each item, should yield a Tensor representing a
    distortion-free (pristine) image or a dictionary with key `"x"` containing such image. Should have finite \_\_len_\_ and indexable \_\_getitem_\_.
  * **sharpness_threshold** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – only patches whose sharpness is at least
    `sharpness_threshold` of the per-image peak sharpness (measured from σ at scale 1) are kept.
  * **save_path** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Path to which weights are to be saved. Must have `.pt` extension. If not passed, weights are returned without saving.
* **Returns:**
  (mu_prisparam, cov_prisparam) as self.dtype on self.device. Also updates self.mu_p, self.cov_p.
