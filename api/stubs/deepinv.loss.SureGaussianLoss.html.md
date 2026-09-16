# SureGaussianLoss

### *class* deepinv.loss.SureGaussianLoss(sigma, tau=1e-2, B=lambda x: ..., unsure=False, step_size=1e-4, momentum=0.9, rng=None)

Bases: [`Loss`](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss)

SURE loss for Gaussian noise

The loss is designed for the following noise model:

$$
y \sim\mathcal{N}(u,\sigma^2 I) \quad \text{with}\quad u= A(x).
$$

The loss is computed as

$$
\frac{1}{m}\|B(y - A\inverse{y})\|_2^2 -\sigma^2 +\frac{2\sigma^2}{m\tau}b^{\top} B^{\top} \left(A\inverse{y+\tau b_i} -
A\inverse{y}\right)
$$

where $R$ is the trainable network, $A$ is the forward operator,
$y$ is the noisy measurement vector of size $m$, $A$ is the forward operator,
$B$ is an optional linear mapping which should be approximately $A^{\dagger}$ (or any stable approximation),
$b\sim\mathcal{N}(0,I)$ and $\tau\geq 0$ is a hyperparameter controlling the
Monte Carlo approximation of the divergence.

This loss approximates the divergence of $A\inverse{y}$ (in the original SURE loss)
using the Monte Carlo approximation in Luisier *et al.*<sup>[1](#footcite-luisier2007new)</sup>.

If the measurement data is truly Gaussian with standard deviation $\sigma$,
this loss is an unbiased estimator of the mean squared loss $\frac{1}{m}\|u-A\inverse{y}\|_2^2$
where $z$ is the noiseless measurement.

#### WARNING
The loss can be sensitive to the choice of $\tau$, which should be proportional to the size of $y$.
The default value of 0.01 is adapted to $y$ vectors with entries in $[0,1]$.

#### NOTE
If the noise level is unknown, the loss can be adapted to the UNSURE loss introduced by Tachella *et al.*<sup>[2](#footcite-tachella2024unsure)</sup>,
which also learns the noise level.

* **Parameters:**
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Standard deviation of the Gaussian noise.
  * **tau** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Approximation constant for the Monte Carlo approximation of the divergence.
  * **B** (*Callable* *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Optional linear metric $B$, which can be used to improve
    the performance of the loss. If ‘A_dagger’, the pseudo-inverse of the forward operator is used.
    Otherwise the metric should be a linear operator that approximates the pseudo-inverse of the forward operator
    such as [`deepinv.physics.LinearPhysics.prox_l2()`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics.prox_l2) with large $\gamma$. By default, the identity is used.
  * **unsure** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, the loss is adapted to the UNSURE loss introduced by Tachella *et al.*<sup>[2](#footcite-tachella2024unsure)</sup>
    where the noise level $\sigma$ is also learned (the input value is used as initialization).
  * **step_size** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Step size for the gradient ascent of the noise level if unsure is `True`.
  * **momentum** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Momentum for the gradient ascent of the noise level if unsure is `True`.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – Optional random number generator. Default is None.

<hr />

* **References:**

* <a id='footcite-luisier2007new'>**[1]**</a> Florian Luisier, Thierry Blu, and Michael Unser. A new sure approach to image denoising: interscale orthonormal wavelet thresholding. *IEEE Transactions on image processing*, 16(3):593–606, 2007.
* <a id='footcite-tachella2024unsure'>**[2]**</a> Julián Tachella, Mike Davies, and Laurent Jacques. Unsure: self-supervised learning with unknown noise level and stein’s unbiased risk estimate. *arXiv preprint arXiv:2409.01985*, 2024.

#### forward(y, x_net, physics, model, \*\*kwargs)

Computes the SURE Loss.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurements.
  * **x_net** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – reconstructed image $\inverse{y}$.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Forward operator associated with the measurements.
  * **model** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Reconstruction network.
* **Returns:**
  torch.Tensor loss of size (batch_size,)
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### *property* name *: [bool](https://docs.python.org/3.9/library/functions.html#bool)*

The name of the loss function. This attribute is deprecated in favor of the class name and it will be removed in a future version.

<a id="sphx-glr-backref-deepinv-loss-suregaussianloss"></a>

## Examples using `SureGaussianLoss`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In this example we demonstrate remote sensing inverse problems for multispectral satellite imaging.">  <div class="sphx-glr-thumbnail-title">Remote sensing with satellite images</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Equivariant splitting consists in minimizing a self-supervised loss to train a reconstruction model using measurement data only :footcitesechaud26Equivariant.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images only via the SURE loss, which exploits knowledge about the noise distribution.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the SURE loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images with unknown noise level only via the UNSURE loss, which is introduced by :footcitetachella2024unsure.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the UNSURE loss.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
