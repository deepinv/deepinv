# PatchCovarianceNoiseEstimator

### *class* deepinv.models.PatchCovarianceNoiseEstimator

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Patch Covariance Gaussian noise level estimator.

This method was initially proposed in Chen *et al.*<sup>[1](#footcite-chen2015efficient)</sup>. Given a noisy image $y = x + n$ where
$n \sim \mathcal{N}(0, \sigma^2)$, this estimator computes an estimate of $\sigma$ based on the
eigenvalues of the covariance matrix of image patches.

#### WARNING
This estimator assumes that the noise in the corrupted image follows a Gaussian distribution.
It may not perform well if the noise distribution deviates significantly from Gaussian, or if the image lacks
sufficient homogeneous regions for reliable patch statistics.

<hr />

* **Examples:**

```pycon
>>> import deepinv as dinv
>>> from deepinv.models import PatchCovarianceNoiseEstimator
>>> rng = torch.Generator('cpu').manual_seed(0)
>>> noise = dinv.physics.GaussianNoise(0.1, rng=rng)(torch.zeros(1, 1, 256, 256))
>>> noise_estimator = PatchCovarianceNoiseEstimator()
>>> sigma_est = noise_estimator(noise)
>>> print(sigma_est)
tensor([0.0995])
```

<hr />

* **References:**

* <a id='footcite-chen2015efficient'>**[1]**</a> Guangyong Chen, Fengyuan Zhu, and Pheng Ann Heng. An efficient statistical method for image noise level estimation. In *Proceedings of the IEEE international conference on computer vision*, 477–485. 2015.

#### *static* estimate_noise(x, patch_size=8, stride=3)

Estimates noise level from the image by computing the covariance of image patches.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image
  * **patch_size** ( *(*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *)*) – patch size
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) estimated noise level
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### forward(x)

Forward pass.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) estimated noise level
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-models-patchcovariancenoiseestimator"></a>

## Examples using `PatchCovarianceNoiseEstimator`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example focuses on blind image Gaussian denoising, i.e. the problem">  <div class="sphx-glr-thumbnail-title">Blind denoising with noise level estimation</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to denoise images corrupted by Poisson-Gaussian noise &lt;deepinv.physics.PoissonGaussianNoise&gt; using the Generalized Anscombe Transform (GAT) &lt;deepinv.models.AnscombeDenoiser&gt;, which converts any Gaussian denoiser into a Poisson-Gaussian denoiser :footcitemakitalo2012optimal.">  <div class="sphx-glr-thumbnail-title">Poisson-Gaussian Denoising with the Generalized Anscombe Transform</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised (blind) denoising of a low-field MRI scan without ground truth data.">  <div class="sphx-glr-thumbnail-title">Low-field MRI denoising without ground truth</div>
</div>
<!-- thumbnail-parent-div-close --></div>
