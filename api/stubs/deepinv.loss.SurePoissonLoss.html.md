# SurePoissonLoss

### *class* deepinv.loss.SurePoissonLoss(gain, tau=1e-3, rng=None)

Bases: [`Loss`](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss)

SURE loss for Poisson noise

The loss is designed for the following noise model:

$$
y = \gamma z \quad \text{with}\quad z\sim \mathcal{P}(\frac{u}{\gamma}), \quad u=A(x).
$$

The loss is computed as

$$
\frac{1}{m}\|y-A\inverse{y}\|_2^2-\frac{\gamma}{m} 1^{\top}y
+\frac{2\gamma}{m\tau}(b\odot y)^{\top} \left(A\inverse{y+\tau b}-A\inverse{y}\right)
$$

where $R$ is the trainable network, $y$ is the noisy measurement vector of size $m$,
$b$ is a Bernoulli random variable taking values of -1 and 1 each with a probability of 0.5,
$\tau$ is a small positive number, and $\odot$ is an elementwise multiplication.

See Le *et al.*<sup>[1](#footcite-le2014unbiased)</sup> for details.
If the measurement data is truly Poisson
this loss is an unbiased estimator of the mean squared loss $\frac{1}{m}\|u-A\inverse{y}\|_2^2$
where $z$ is the noiseless measurement.

#### WARNING
The loss can be sensitive to the choice of $\tau$, which should be proportional to the size of $y$.
The default value of 0.01 is adapted to $y$ vectors with entries in $[0,1]$.

* **Parameters:**
  * **gain** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Gain of the Poisson Noise.
  * **tau** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Approximation constant for the Monte Carlo approximation of the divergence.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – Optional random number generator. Default is None.

<hr />

* **References:**

* <a id='footcite-le2014unbiased'>**[1]**</a> MY Le, ED Angelini, and JC Olivo-Marin. An unbiased risk estimator for image denoising in the presence of mixed poisson-gaussian noise [j]. *IEEE Transactions on Image Processing*, 23(6):2750–2755, 2014.

#### forward(y, x_net, physics, model, \*\*kwargs)

Computes the SURE loss.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements.
  * **x_net** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – reconstructed image $\inverse{y}$.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Forward operator associated with the measurements
  * **model** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Reconstruction network
* **Returns:**
  torch.Tensor loss of size (batch_size,)
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### *property* name *: [bool](https://docs.python.org/3.9/library/functions.html#bool)*

The name of the loss function. This attribute is deprecated in favor of the class name and it will be removed in a future version.

<a id="sphx-glr-backref-deepinv-loss-surepoissonloss"></a>

## Examples using `SurePoissonLoss`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images only via the SURE loss, which exploits knowledge about the noise distribution.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the SURE loss.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
