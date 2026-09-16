# WaveletNoiseEstimator

### *class* deepinv.models.WaveletNoiseEstimator

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Wavelet Gaussian noise level estimator.

This estimator was proposed in Donoho and Johnstone<sup>[1](#footcite-donoho1994ideal)</sup>. It estimates the standard
deviation of a Gaussian noise corrupted image. More precisely, given a noisy image
$y = x + n$ where $n \sim \mathcal{N}(0, \sigma^2)$, the noise level estimator computes an
estimate of $\sigma$ as

$$
\hat{\sigma} = \frac{\text{median}(|w|)}{0.6745}

$$

where $w$ are the wavelet coefficients of the noisy image $y$ at the first level of decomposition.

#### NOTE
As noted by the authors, this estimator is an upper bound on the noise level, and may overestimate the true
noise level in some cases, in particular if the SNR is high (i.e., the noise level is low compared to the
signal level). In such cases, the estimator may be less accurate than the [`PatchCovarianceNoiseEstimator`](https://deepinv.org/api/stubs/deepinv.models.PatchCovarianceNoiseEstimator.html.md#deepinv.models.PatchCovarianceNoiseEstimator)
which is based on the eigenvalues of the covariance matrix of image patches.

#### WARNING
This estimator assumes that the noise in the corrupted image follows a Gaussian distribution.
It may not perform well if the noise distribution deviates significantly from Gaussian, or if the image contains
strong edges or textures that can affect the wavelet coefficients.

#### WARNING
This model requires Pytorch Wavelets (`ptwt`) to be installed. It can be installed with
`pip install ptwt`.

<hr />

* **Examples:**
  ```pycon
  >>> import deepinv as dinv
  >>> from deepinv.models import WaveletNoiseEstimator
  >>> rng = torch.Generator('cpu').manual_seed(0)
  >>> noise = dinv.physics.GaussianNoise(0.1, rng=rng)(torch.zeros(1, 1, 256, 256))
  >>> noise_estimator = WaveletNoiseEstimator()
  >>> sigma_est = noise_estimator(noise)
  >>> print(sigma_est)
  tensor([0.1003])
  ```

<hr />

* **References:**

* <a id='footcite-donoho1994ideal'>**[1]**</a> David L Donoho and Iain M Johnstone. Ideal spatial adaptation by wavelet shrinkage. *Biometrika*, 81(3):425–455, 1994.

#### *static* estimate_noise(x)

Estimates noise level in image im.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) estimated noise level
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### forward(x)

Forward pass.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image
* **Returns:**
  (:class: `torch.Tensor`) estimated noise level
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-models-waveletnoiseestimator"></a>

## Examples using `WaveletNoiseEstimator`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example focuses on blind image Gaussian denoising, i.e. the problem">  <div class="sphx-glr-thumbnail-title">Blind denoising with noise level estimation</div>
</div>
<!-- thumbnail-parent-div-close --></div>
