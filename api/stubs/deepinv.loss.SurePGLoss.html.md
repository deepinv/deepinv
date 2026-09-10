# SurePGLoss

### *class* deepinv.loss.SurePGLoss(sigma, gain, tau1=1e-3, tau2=1e-2, second_derivative=False, unsure=False, step_size=(1e-4, 1e-4), momentum=(0.9, 0.9), rng=None)

Bases: [`Loss`](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss)

SURE loss for Poisson-Gaussian noise

The loss is designed for the following noise model:

$$
y = \gamma z + \epsilon
$$

where $u = A(x)$, $z \sim \mathcal{P}\left(\frac{u}{\gamma}\right)$,
and $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$.

The loss is computed as

$$
& \frac{1}{m}\|y-A\inverse{y}\|_2^2-\frac{\gamma}{m} 1^{\top}y-\sigma^2
+\frac{2}{m\tau_1}(b\odot (\gamma y + \sigma^2 I))^{\top} \left(A\inverse{y+\tau b}-A\inverse{y} \right) \\\\
& +\frac{2\gamma \sigma^2}{m\tau_2^2}c^{\top} \left( A\inverse{y+\tau c} + A\inverse{y-\tau c} - 2A\inverse{y} \right)
$$

where $R$ is the trainable network, $y$ is the noisy measurement vector,
$b$ is a Bernoulli random variable taking values of -1 and 1 each with a probability of 0.5,
$\tau$ is a small positive number, and $\odot$ is an elementwise multiplication.

If the measurement data is truly Poisson-Gaussian
this loss is an unbiased estimator of the mean squared loss $\frac{1}{m}\|u-A\inverse{y}\|_2^2$
where $z$ is the noiseless measurement.

See Le *et al.*<sup>[1](#footcite-le2014unbiased)</sup> for details.

#### WARNING
The loss can be sensitive to the choice of $\tau$, which should be proportional to the size of $y$.
The default value of 0.01 is adapted to $y$ vectors with entries in $[0,1]$.

#### NOTE
If the noise levels are unknown, the loss can be adapted to the UNSURE loss introduced by Tachella *et al.*<sup>[2](#footcite-tachella2024unsure)</sup>,
which also learns the noise levels.

* **Parameters:**
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Standard deviation of the Gaussian noise.
  * **gamma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Gain of the Poisson Noise.
  * **tau** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Approximation constant for the Monte Carlo approximation of the divergence.
  * **tau2** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Approximation constant for the second derivative.
  * **second_derivative** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `False`, the last term in the loss (approximating the second derivative) is removed
    to speed up computations, at the cost of a possibly inexact loss. Default `True`.
  * **unsure** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, the loss is adapted to the UNSURE loss introduced by Tachella *et al.*<sup>[2](#footcite-tachella2024unsure)</sup>
    where $\gamma$ and $\sigma^2$ are also learned (their input value is used as initialization).
  * **step_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *]*) – Step size for the gradient ascent of the noise levels if unsure is `True`.
  * **momentum** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *]*) – Momentum for the gradient ascent of the noise levels if unsure is `True`.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – Optional random number generator. Default is None.

<hr />

* **References:**

* <a id='footcite-le2014unbiased'>**[1]**</a> MY Le, ED Angelini, and JC Olivo-Marin. An unbiased risk estimator for image denoising in the presence of mixed poisson-gaussian noise [j]. *IEEE Transactions on Image Processing*, 23(6):2750–2755, 2014.
* <a id='footcite-tachella2024unsure'>**[2]**</a> Julián Tachella, Mike Davies, and Laurent Jacques. Unsure: self-supervised learning with unknown noise level and stein’s unbiased risk estimate. *arXiv preprint arXiv:2409.01985*, 2024.

#### forward(y, x_net, physics, model, \*\*kwargs)

Computes the SURE loss.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements.
  * **x_net** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – reconstructed image $\inverse{y}$.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Forward operator associated with the measurements
  * **f** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Reconstruction network
* **Returns:**
  torch.Tensor loss of size (batch_size,)
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-loss-surepgloss"></a>

## Examples using `SurePGLoss`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images only via the SURE loss, which exploits knowledge about the noise distribution.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the SURE loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images with unknown noise level only via the UNSURE loss, which is introduced by :footcitetachella2024unsure.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the UNSURE loss.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
