# R2RLoss

### *class* deepinv.loss.R2RLoss(metric=None, noise_model=None, alpha=0.15, eval_n_samples=5)

Bases: [`Loss`](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss)

Generalized Recorrupted-to-Recorrupted (GR2R) Loss

This self-supervised loss can be used when the noise model is
Gaussian, Poisson, Gamma or Binomial. The GR2R loss is defined as:

$$
y_1 \sim p(y_1 \vert y, \alpha),
$$

where

| Noise Model                                        | $p(y_1 \vert y,\alpha)$                                                                                                                       |
|----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| $y \sim \mathcal{N}(x, I\sigma^2)$                 | $y_1 = y + \sqrt{\frac{\alpha}{1-\alpha}} \boldsymbol{\omega}, \quad \boldsymbol{\omega} \sim \mathcal{N}(0, I\sigma^2)$                      |
| $z \sim \mathcal{P}(x/\gamma), \quad y = \gamma z$ | $y_1 = \frac{y - \gamma \boldsymbol{\omega}}{1 - \alpha}, \quad \boldsymbol{\omega} \sim \mathrm{Bin}(z, \alpha)$                             |
| $y \sim \mathcal{G}(\ell, \ell / x)$               | $y_1 = y \circ (\mathbf{1} - \boldsymbol{\omega}) / (1 - \alpha), \quad \boldsymbol{\omega} \sim \mathrm{Beta}(\ell\alpha, \ell(1 - \alpha))$ |
| $z \sim \mathrm{Bin}(\ell, x), \quad y = z / \ell$ | $y_1 = \frac{y - \boldsymbol{\omega} / \ell}{1 - \alpha}, \quad \boldsymbol{\omega} \sim \mathrm{HypGeo}(\ell, \ell\alpha, z)$                |

and

$$
y_2 = \frac{1}{\alpha} \left( y - y_1(1-\alpha) \right),
$$

then, the loss is computed as:

$$
\| AR(y_1) - y_2 \|_2^2,
$$

where, $R$ is the trainable network, $A$ is the forward operator,
$y$ is the noisy measurement, and $\alpha$ is a scaling factor.

The loss was first introduced by Pang *et al.*<sup>[1](#footcite-pang2021recorrupted)</sup>
for the specific case of Gaussian noise, formalizing the Noise2Noisier loss from Moran *et al.*<sup>[2](#footcite-moran2020noisier2noise)</sup>,
such that it is statistically equivalent to the supervised loss function defined on noisy/clean image pairs.
The loss was later extended to other exponential family noise distributions by Monroy *et al.*<sup>[3](#footcite-monroy2025generalized)</sup>, including Poisson,
Gamma and Binomial noise distributions.

#### WARNING
The model should be adapted before training using the method [`adapt_model()`](#deepinv.loss.R2RLoss.adapt_model) to include the additional noise at the input.

#### NOTE
To obtain the best test performance, the trained model should be averaged at test time
over multiple realizations of the added noise, i.e. $\hat{x} = \frac{1}{N}\sum_{i=1}^N R(y_1^{(i)})$,
where $N>1$. This can be achieved using [`adapt_model()`](#deepinv.loss.R2RLoss.adapt_model).

#### NOTE
If the `noise_model` parameter is not provided, the noise model from the physics module will be used.

* **Parameters:**
  * **metric** ([*Metric*](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric) *,* [*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Metric for calculating loss, defaults to MSE.
  * **noise_model** ([*NoiseModel*](https://deepinv.org/api/stubs/deepinv.physics.NoiseModel.html.md#deepinv.physics.NoiseModel)) – Noise model of the natural exponential family, defaults to None. Implemented options are [`deepinv.physics.GaussianNoise`](https://deepinv.org/api/stubs/deepinv.physics.GaussianNoise.html.md#deepinv.physics.GaussianNoise), [`deepinv.physics.PoissonNoise`](https://deepinv.org/api/stubs/deepinv.physics.PoissonNoise.html.md#deepinv.physics.PoissonNoise) and [`deepinv.physics.GammaNoise`](https://deepinv.org/api/stubs/deepinv.physics.GammaNoise.html.md#deepinv.physics.GammaNoise)
  * **alpha** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Scaling factor of the corruption.
  * **eval_n_samples** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of samples used for the Monte Carlo approximation.

<hr />

* **Example:**

```pycon
>>> import torch
>>> import deepinv as dinv
>>> sigma = 0.1
>>> noise_model = dinv.physics.GaussianNoise(sigma)
>>> physics = dinv.physics.Denoising(noise_model)
>>> model = dinv.models.MedianFilter()
>>> loss = dinv.loss.R2RLoss(noise_model=noise_model, eval_n_samples=2)
>>> model = loss.adapt_model(model) # important step!
>>> x = torch.ones((1, 1, 8, 8))
>>> y = physics(x)
>>> x_net = model(y, physics, update_parameters=True) # save extra noise in forward pass
>>> l = loss(x_net, y, physics, model)
>>> print(l.item() > 0)
True
```

<hr />

* **References:**

* <a id='footcite-pang2021recorrupted'>**[1]**</a> Tongyao Pang, Huan Zheng, Yuhui Quan, and Hui Ji. Recorrupted-to-recorrupted: unsupervised deep learning for image denoising. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2043–2052. 2021.
* <a id='footcite-moran2020noisier2noise'>**[2]**</a> Nick Moran, Dan Schmidt, Yu Zhong, and Patrick Coady. Noisier2noise: learning to denoise from unpaired noisy data. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 12064–12072. 2020.
* <a id='footcite-monroy2025generalized'>**[3]**</a> Brayan Monroy, Jorge Bacca, and Julián Tachella. Generalized recorrupted-to-recorrupted: self-supervised learning beyond gaussian noise. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, 28155–28164. 2025.

#### adapt_model(model, \*\*kwargs)

Adds noise to model input.

This method modifies a reconstruction
model $R$ to include the re-corruption mechanism at the input:

$$
\hat{R}(y) = \frac{1}{N}\sum_{i=1}^N R(y_1^{(i)}),
$$

where $y_1^{(i)} \sim p(y_1 \vert y, \alpha)$ are i.i.d samples, and $N\geq 1$ are the number of samples used for the Monte Carlo approximation.
During training (i.e. when `model.train()`), we use only one sample, i.e. $N=1$
for computational efficiency, whereas at test time, we use multiple samples for better performance.

* **Parameters:**
  * **model** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Reconstruction model.
  * **noise_model** ([*NoiseModel*](https://deepinv.org/api/stubs/deepinv.physics.NoiseModel.html.md#deepinv.physics.NoiseModel)) – Noise model of the natural exponential family.
    Implemented options are [`deepinv.physics.GaussianNoise`](https://deepinv.org/api/stubs/deepinv.physics.GaussianNoise.html.md#deepinv.physics.GaussianNoise), [`deepinv.physics.PoissonNoise`](https://deepinv.org/api/stubs/deepinv.physics.PoissonNoise.html.md#deepinv.physics.PoissonNoise) and [`deepinv.physics.GammaNoise`](https://deepinv.org/api/stubs/deepinv.physics.GammaNoise.html.md#deepinv.physics.GammaNoise)
  * **alpha** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Scaling factor of the corruption.
* **Returns:**
  ([`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) Modified model.

#### forward(x_net, y, physics, model, \*\*kwargs)

Computes the GR2R Loss.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurements.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Forward operator associated with the measurements.
  * **model** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Reconstruction model.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) R2R loss.

#### *property* name *: [bool](https://docs.python.org/3.9/library/functions.html#bool)*

The name of the loss function. This attribute is deprecated in favor of the class name and it will be removed in a future version.

<a id="sphx-glr-backref-deepinv-loss-r2rloss"></a>

## Examples using `R2RLoss`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model :footciteterris2025reconstruct to solve inverse problems.">  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Equivariant splitting consists in minimizing a self-supervised loss to train a reconstruction model using measurement data only :footcitesechaud26Equivariant.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised (blind) denoising of a low-field MRI scan without ground truth data.">  <div class="sphx-glr-thumbnail-title">Low-field MRI denoising without ground truth</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, using noisy images only via the Generalized Recorrupted2Recorrupted (GR2R) loss :footcitemonroy2025generalized, which exploits knowledge about the noise distribution. You can change the noise distribution by selecting from predefined noise models such as Gaussian, Poisson, and Gamma noise.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the Generalized R2R loss.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
