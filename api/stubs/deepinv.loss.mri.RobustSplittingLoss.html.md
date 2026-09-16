# RobustSplittingLoss

### *class* deepinv.loss.mri.RobustSplittingLoss(mask_generator, physics_generator, noise_model=None, alpha=0.75, metric=None)

Bases: [`WeightedSplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.mri.WeightedSplittingLoss.html.md#deepinv.loss.mri.WeightedSplittingLoss)

Robust Weighted Splitting Loss

Implements the Robust-SSDU loss from Millard and Chiew<sup>[1](#footcite-millard2024clean)</sup>.
The loss is designed for problems where measurements are observed as $y_i=M_iAx+\epsilon$,
where $M_i$ is a random mask, such as in [`MRI`](https://deepinv.org/api/stubs/deepinv.physics.MRI.html.md#deepinv.physics.MRI) where `A` is the Fourier transform,
and $\epsilon$ is Gaussian noise.
The loss is related to the [`deepinv.loss.mri.WeightedSplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.mri.WeightedSplittingLoss.html.md#deepinv.loss.mri.WeightedSplittingLoss) as follows:

$$
\mathcal{L}_\text{Robust-SSDU}=\mathcal{L}_\text{Weighted-SSDU}(\tilde{y};y) + \lVert(1+\frac{1}{\alpha^2}) M_1 M (\forw{\inverse{\tilde{y},A} - y}\rVert_2^2
$$

where $\tilde{y}\sim\mathcal{N}(y,\alpha^2\sigma^2\mathbf{I})$ is further noised (i.e. “noisier”) measurement, and $\alpha$ is a hyperparameter.
This is derived from Eqs. 34 & 35 of the paper <sup>[1](#footcite-millard2024clean)</sup>.
At inference, the original measurement $y$ is used as input.

#### NOTE
See [`deepinv.loss.mri.WeightedSplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.mri.WeightedSplittingLoss.html.md#deepinv.loss.mri.WeightedSplittingLoss) on what is expected of the input measurements, and the `mask_generator`.

* **Parameters:**
  * **mask_generator** ([*deepinv.physics.generator.BernoulliSplittingMaskGenerator*](https://deepinv.org/api/stubs/deepinv.physics.generator.BernoulliSplittingMaskGenerator.html.md#deepinv.physics.generator.BernoulliSplittingMaskGenerator)) – splitting mask generator for further subsampling.
  * **physics_generator** ([*deepinv.physics.generator.BaseMaskGenerator*](https://deepinv.org/api/stubs/deepinv.physics.generator.BaseMaskGenerator.html.md#deepinv.physics.generator.BaseMaskGenerator)) – original mask generator used to generate the measurements.
  * **noise_model** ([*deepinv.physics.NoiseModel*](https://deepinv.org/api/stubs/deepinv.physics.NoiseModel.html.md#deepinv.physics.NoiseModel)) – noise model for adding further noise, must be of same type as original measurement noise.
    Note this loss only supports [`deepinv.physics.GaussianNoise`](https://deepinv.org/api/stubs/deepinv.physics.GaussianNoise.html.md#deepinv.physics.GaussianNoise).
  * **alpha** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – hyperparameter controlling further noise std.
  * **metric** ([*Metric*](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric) *,* [*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – metric used for computing data consistency, which is set as the mean squared error by default.

<hr />

* **References:**

* <a id='footcite-millard2024clean'>**[1]**</a> Charles Millard and Mark Chiew. Clean self-supervised mri reconstruction from noisy, sub-sampled training data with robust ssdu. *Bioengineering*, 11(12):1305, 2024.

#### *class* Noisier2NoiseMetric(weight, pixel_metric)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Helper metric for computing weighted Noisier2Noise

#### *static* expand_mask(mask, y)

Expand mask intermediate dimensions to match those of y, where intermediate
dimensions are those (e.g. depth, coils, time) that are not the first two dims (batch, channel),
nor the final two dims (H, W).
