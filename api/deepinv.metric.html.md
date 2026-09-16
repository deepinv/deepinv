# deepinv.metric

Metrics are generally used to evaluate the performance of a model, or as the distance function inside a loss function.
Please refer to the [user guide](https://deepinv.org/user_guide/training/metric.html.md#metric) for more information.

## Base class

**User Guide:** refer to [Metrics](https://deepinv.org/user_guide/training/metric.html.md#metric) for more information.

| [`deepinv.loss.metric.Metric`](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric)   | Base class for metrics.   |
|----------------------------------------------------------------------------------------------------------|---------------------------|

## Full Reference Metrics

**User Guide:** refer to [Full Reference Metrics](https://deepinv.org/user_guide/training/metric.html.md#full-reference-metrics) for more information.

| [`deepinv.loss.metric.MSE`](https://deepinv.org/api/stubs/deepinv.loss.metric.MSE.html.md#deepinv.loss.metric.MSE)                                 | Mean Squared Error metric.                                    |
|----------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| [`deepinv.loss.metric.NMSE`](https://deepinv.org/api/stubs/deepinv.loss.metric.NMSE.html.md#deepinv.loss.metric.NMSE)                               | Normalized Mean Squared Error metric.                         |
| [`deepinv.loss.metric.MAE`](https://deepinv.org/api/stubs/deepinv.loss.metric.MAE.html.md#deepinv.loss.metric.MAE)                                 | Mean Absolute Error metric.                                   |
| [`deepinv.loss.metric.PSNR`](https://deepinv.org/api/stubs/deepinv.loss.metric.PSNR.html.md#deepinv.loss.metric.PSNR)                               | Peak Signal-to-Noise Ratio (PSNR) metric.                     |
| [`deepinv.loss.metric.SNR`](https://deepinv.org/api/stubs/deepinv.loss.metric.SNR.html.md#deepinv.loss.metric.SNR)                                 | Compute the signal-to-noise ratio (SNR)                       |
| [`deepinv.loss.metric.SSIM`](https://deepinv.org/api/stubs/deepinv.loss.metric.SSIM.html.md#deepinv.loss.metric.SSIM)                               | Structural Similarity Index (SSIM) metric using torchmetrics. |
| [`deepinv.loss.metric.L1L2`](https://deepinv.org/api/stubs/deepinv.loss.metric.L1L2.html.md#deepinv.loss.metric.L1L2)                               | Combined L2 and L1 metric.                                    |
| [`deepinv.loss.metric.LpNorm`](https://deepinv.org/api/stubs/deepinv.loss.metric.LpNorm.html.md#deepinv.loss.metric.LpNorm)                           | $\ell_p$ metric for $p>0$.                                    |
| [`deepinv.loss.metric.LPIPS`](https://deepinv.org/api/stubs/deepinv.loss.metric.LPIPS.html.md#deepinv.loss.metric.LPIPS)                             | Learned Perceptual Image Patch Similarity (LPIPS) metric.     |
| [`deepinv.loss.metric.SpectralAngleMapper`](https://deepinv.org/api/stubs/deepinv.loss.metric.SpectralAngleMapper.html.md#deepinv.loss.metric.SpectralAngleMapper) | Spectral Angle Mapper (SAM).                                  |
| [`deepinv.loss.metric.ERGAS`](https://deepinv.org/api/stubs/deepinv.loss.metric.ERGAS.html.md#deepinv.loss.metric.ERGAS)                             | Error relative global dimensionless synthesis metric.         |
| [`deepinv.loss.metric.HaarPSI`](https://deepinv.org/api/stubs/deepinv.loss.metric.HaarPSI.html.md#deepinv.loss.metric.HaarPSI)                         | HaarPSI metric with tuned parameters.                         |
| [`deepinv.loss.metric.CosineSimilarity`](https://deepinv.org/api/stubs/deepinv.loss.metric.CosineSimilarity.html.md#deepinv.loss.metric.CosineSimilarity)       | Cosine similarity metric.                                     |
| [`deepinv.loss.metric.GMSD`](https://deepinv.org/api/stubs/deepinv.loss.metric.GMSD.html.md#deepinv.loss.metric.GMSD)                               | Gradient Magnitude Similarity Deviation (GMSD) metric.        |
| [`deepinv.loss.metric.RecoveryCoefficient`](https://deepinv.org/api/stubs/deepinv.loss.metric.RecoveryCoefficient.html.md#deepinv.loss.metric.RecoveryCoefficient) | Recovery Coefficient metric used in emission tomography.      |
| [`deepinv.loss.metric.NRMSE`](https://deepinv.org/api/stubs/deepinv.loss.metric.NRMSE.html.md#deepinv.loss.metric.NRMSE)                             | Normalized Root Mean Squared Error metric.                    |

## No Reference Metrics

**User Guide:** refer to [No Reference Metrics](https://deepinv.org/user_guide/training/metric.html.md#no-reference-metrics) for more information.

| [`deepinv.loss.metric.NIQE`](https://deepinv.org/api/stubs/deepinv.loss.metric.NIQE.html.md#deepinv.loss.metric.NIQE)                     | Natural Image Quality Evaluator (NIQE) metric.                        |
|------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| [`deepinv.loss.metric.BRISQUE`](https://deepinv.org/api/stubs/deepinv.loss.metric.BRISQUE.html.md#deepinv.loss.metric.BRISQUE)               | Blind/Referenceless Image Spatial QUality Evaluator (BRISQUE) metric. |
| [`deepinv.loss.metric.NIMA`](https://deepinv.org/api/stubs/deepinv.loss.metric.NIMA.html.md#deepinv.loss.metric.NIMA)                     | Neural Image Assessment (NIMA) metric.                                |
| [`deepinv.loss.metric.QNR`](https://deepinv.org/api/stubs/deepinv.loss.metric.QNR.html.md#deepinv.loss.metric.QNR)                       | Quality with No Reference (QNR) metric for pansharpening.             |
| [`deepinv.loss.metric.BlurStrength`](https://deepinv.org/api/stubs/deepinv.loss.metric.BlurStrength.html.md#deepinv.loss.metric.BlurStrength)     | No-reference blur strength metric for batched images.                 |
| [`deepinv.loss.metric.SharpnessIndex`](https://deepinv.org/api/stubs/deepinv.loss.metric.SharpnessIndex.html.md#deepinv.loss.metric.SharpnessIndex) | No-reference sharpness index metric for 2D images.                    |
