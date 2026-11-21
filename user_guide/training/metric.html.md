<a id="metric"></a>

# Metrics

This module contains popular metrics for inverse problems.

Metrics are generally used to evaluate the performance of a model, or as the distance function inside a loss function.

## Introduction

All metrics inherit from the base class [`deepinv.loss.metric.Metric`](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric), which is a [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).
All metrics take either `x_net, x` for a full-reference metric or `x_net` for a no-reference metric.

All metrics can perform a standard set of pre and post processing, including
operating on complex numbers, normalisation and reduction. See [`deepinv.loss.metric.Metric`](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric) for more details.

#### NOTE
By default, metrics do not reduce over the batch dimension, as the usual usage is to average the metrics over a dataset yourself.
This discourages averaging over metrics which might in turn have averaged over uneven batch sizes.
Note we provide [`deepinv.utils.AverageMeter`](https://deepinv.org/api/stubs/deepinv.utils.AverageMeter.html.md#deepinv.utils.AverageMeter) to easily keep track of the average of metrics.
For example, we use this in our trainer [`deepinv.Trainer`](https://deepinv.org/api/stubs/deepinv.Trainer.html.md#deepinv.Trainer).

However, you can use the `reduction` argument to perform reduction, e.g. if you want a single metric calculation rather than over a dataset.

All metrics can either be used directly as metrics, or as the backbone for training losses.
To do this, wrap the metric in a suitable loss such as [`deepinv.loss.SupLoss`](https://deepinv.org/api/stubs/deepinv.loss.SupLoss.html.md#deepinv.loss.SupLoss) or [`deepinv.loss.MCLoss`](https://deepinv.org/api/stubs/deepinv.loss.MCLoss.html.md#deepinv.loss.MCLoss).
In this way, [`deepinv.loss.metric.MSE`](https://deepinv.org/api/stubs/deepinv.loss.metric.MSE.html.md#deepinv.loss.metric.MSE) replaces [`torch.nn.MSELoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss) and [`deepinv.loss.metric.MAE`](https://deepinv.org/api/stubs/deepinv.loss.metric.MAE.html.md#deepinv.loss.metric.MAE) replaces [`torch.nn.L1Loss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss),
and you can use these in a loss like `SupLoss(metric=MSE())`.

Metrics can be classified as distortion or perceptual,
see [the Perception-Distortion Tradeoff](https://openaccess.thecvf.com/content_cvpr_2018/papers/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.pdf)
for an explanation of distortion vs perceptual metrics.

Finally, you can also wrap existing metric functions using `Metric(metric=f)`, see [`deepinv.loss.metric.Metric`](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric) for an example.

#### NOTE
For some metrics, higher is better; for these, you must also set `train_loss=True`.

#### TIP
For convenience, you can also import metrics directly from `deepinv.metric` or `deepinv.loss`.

Example:

```pycon
>>> import torch
>>> import deepinv as dinv
>>> m = dinv.metric.SSIM()
>>> x = torch.ones(2, 3, 16, 16) # B,C,H,W
>>> x_hat = x + 0.01
>>> m(x_hat, x) # Calculate metric for each image in batch
tensor([1.0000, 1.0000])
>>> m = dinv.metric.SSIM(reduction="sum")
>>> m(x_hat, x) # Sum over batch
tensor(1.9999)
>>> l = dinv.loss.MCLoss(metric=dinv.metric.SSIM(train_loss=True, reduction="mean")) # Use SSIM for training
```

<a id="full-reference-metrics"></a>

## Full Reference Metrics

Full reference metrics are used to measure the difference between the original `x` and the reconstructed image `x_net`.

#### Full Reference Metrics

| **Metric**                                                                                                                       | **Definition**                                                                                                                                                                                                       |
|----------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`deepinv.loss.metric.MSE`](https://deepinv.org/api/stubs/deepinv.loss.metric.MSE.html.md#deepinv.loss.metric.MSE)                                 | $\text{MSE}(\hat{x},x) = \frac{1}{n} \sum_{i=1}^n (x_i - \hat{x}_i)^2$                                                                                                                                               |
| [`deepinv.loss.metric.NMSE`](https://deepinv.org/api/stubs/deepinv.loss.metric.NMSE.html.md#deepinv.loss.metric.NMSE)                               | $\text{NMSE}(\hat{x},x) = \frac{\| x - \hat{x} \|_2^2}{\| x \|_2^2}$                                                                                                                                                 |
| [`deepinv.loss.metric.NRMSE`](https://deepinv.org/api/stubs/deepinv.loss.metric.NRMSE.html.md#deepinv.loss.metric.NRMSE)                             | $\text{NRMSE}(\hat{x},x) = \frac{\| x - \hat{x} \|_2}{\| x \|_2} = \sqrt{\text{NMSE}(\hat{x},x)}$                                                                                                                    |
| [`deepinv.loss.metric.MAE`](https://deepinv.org/api/stubs/deepinv.loss.metric.MAE.html.md#deepinv.loss.metric.MAE)                                 | $\text{MAE}(\hat{x},x) = \frac{1}{n} \sum_{i=1}^n |x_i - \hat{x}_i|$                                                                                                                                                 |
| [`deepinv.loss.metric.PSNR`](https://deepinv.org/api/stubs/deepinv.loss.metric.PSNR.html.md#deepinv.loss.metric.PSNR)                               | $\text{PSNR}(\hat{x},x) = 10 \cdot \log_{10} \left( \frac{\text{MAX}^2}{\text{MSE}(\hat{x},x)} \right)$, where $\text{MAX}$ is the maximum possible pixel value of the image                                         |
| [`deepinv.loss.metric.SNR`](https://deepinv.org/api/stubs/deepinv.loss.metric.SNR.html.md#deepinv.loss.metric.SNR)                                 | $\mathrm{SNR} = 10 \log_{10} \left( \frac{\| x_i \|_2^2}{\|x_i - \hat{x}_i\|_2^2} \right)$                                                                                                                           |
| [`deepinv.loss.metric.SSIM`](https://deepinv.org/api/stubs/deepinv.loss.metric.SSIM.html.md#deepinv.loss.metric.SSIM)                               | $\text{SSIM}(\hat{x},x) = \frac{(2 \mu_x \mu_{\hat{x}} + C_1)(2 \sigma_{x\hat{x}} + C_2)}{(\mu_x^2 + \mu_{\hat{x}}^2 + C_1)(\sigma_x^2 + \sigma_{\hat{x}}^2 + C_2)}$, where $\mu$ and $\sigma$ are mean and variance |
| [`deepinv.loss.metric.L1L2`](https://deepinv.org/api/stubs/deepinv.loss.metric.L1L2.html.md#deepinv.loss.metric.L1L2)                               | $\text{L1L2}(\hat{x},x) = \alpha \|x - \hat{x}\|_1 + (1 - \alpha) \|x - \hat{x}\|_2$, where $\alpha$ is a balancing parameter                                                                                        |
| [`deepinv.loss.metric.LpNorm`](https://deepinv.org/api/stubs/deepinv.loss.metric.LpNorm.html.md#deepinv.loss.metric.LpNorm)                           | $\text{LpNorm}(\hat{x},x) = \|x - \hat{x}\|_p^p$                                                                                                                                                                     |
| [`deepinv.loss.metric.LPIPS`](https://deepinv.org/api/stubs/deepinv.loss.metric.LPIPS.html.md#deepinv.loss.metric.LPIPS)                             | Uses a pretrained network to calculate the perceptual similarity between two images.                                                                                                                                 |
| [`deepinv.loss.metric.SpectralAngleMapper`](https://deepinv.org/api/stubs/deepinv.loss.metric.SpectralAngleMapper.html.md#deepinv.loss.metric.SpectralAngleMapper) | Multispectral image metric that calculates spectral similarity between bands.                                                                                                                                        |
| [`deepinv.loss.metric.ERGAS`](https://deepinv.org/api/stubs/deepinv.loss.metric.ERGAS.html.md#deepinv.loss.metric.ERGAS)                             | “Error relative global dimensionless synthesis” multispectral image metric for pan-sharpening problems.                                                                                                              |
| [`deepinv.loss.metric.HaarPSI`](https://deepinv.org/api/stubs/deepinv.loss.metric.HaarPSI.html.md#deepinv.loss.metric.HaarPSI)                         | HaarPSI metric tuned for natural and medical images.                                                                                                                                                                 |
| [`deepinv.loss.metric.CosineSimilarity`](https://deepinv.org/api/stubs/deepinv.loss.metric.CosineSimilarity.html.md#deepinv.loss.metric.CosineSimilarity)       | $\text{CosineSim}(\hat{x}, x) =\dfrac{\langle \hat{x}, x \rangle}{\|\hat{x}\|_2 \, \|x\|_2}$,where $\langle \hat{x}, x \rangle$ is the Euclidean inner product.                                                      |
| [`deepinv.loss.metric.GMSD`](https://deepinv.org/api/stubs/deepinv.loss.metric.GMSD.html.md#deepinv.loss.metric.GMSD)                               | Gradient Magnitude Similarity Deviation                                                                                                                                                                              |
| [`deepinv.loss.metric.RecoveryCoefficient`](https://deepinv.org/api/stubs/deepinv.loss.metric.RecoveryCoefficient.html.md#deepinv.loss.metric.RecoveryCoefficient) | $\mathrm{RC}(\hat{x}, x) = \frac{\sum_{i \in \Omega} \hat{x}_i}{\sum_{i \in \Omega} x_i}$                                                                                                                            |

<a id="no-reference-metrics"></a>

## No Reference Metrics

We implement no-reference perceptual metrics, they only require the reconstructed image `x_net`.

#### No Reference Metrics

| **Metric**                                                                                                             | **Definition**                                                                                                                                                            |
|------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`deepinv.loss.metric.NIQE`](https://deepinv.org/api/stubs/deepinv.loss.metric.NIQE.html.md#deepinv.loss.metric.NIQE)                     | Calculates deviation of image from statistical regularities of natural images.                                                                                            |
| [`deepinv.loss.metric.BRISQUE`](https://deepinv.org/api/stubs/deepinv.loss.metric.BRISQUE.html.md#deepinv.loss.metric.BRISQUE)               | Scores the deviation of an image from the natural scene statistics of pristine natural images, using a support vector regressor trained on human quality ratings.         |
| [`deepinv.loss.metric.NIMA`](https://deepinv.org/api/stubs/deepinv.loss.metric.NIMA.html.md#deepinv.loss.metric.NIMA)                     | Predicts the distribution of human opinion scores of an image with a convolutional network, either of its aesthetic appeal or of its technical quality. Higher is better. |
| [`deepinv.loss.metric.QNR`](https://deepinv.org/api/stubs/deepinv.loss.metric.QNR.html.md#deepinv.loss.metric.QNR)                       | Multispectral image metric $\text{QNR}(\hat{x}) = (1-D_\lambda)^\alpha(1 - D_s)^\beta$, where $D_\lambda$ and $D_s$ are spectral and spatial distortions.                 |
| [`deepinv.loss.metric.BlurStrength`](https://deepinv.org/api/stubs/deepinv.loss.metric.BlurStrength.html.md#deepinv.loss.metric.BlurStrength)     | Calculates the blurriness of an image based on the spread of edges. Can be used to measure motion blur or out-of-focus blur.                                              |
| [`deepinv.loss.metric.SharpnessIndex`](https://deepinv.org/api/stubs/deepinv.loss.metric.SharpnessIndex.html.md#deepinv.loss.metric.SharpnessIndex) | Calculates the sharpness of an image, can be used to asses image quality.                                                                                                 |
