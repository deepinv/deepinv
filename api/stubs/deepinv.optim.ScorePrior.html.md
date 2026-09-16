# ScorePrior

### *class* deepinv.optim.ScorePrior(denoiser, \*args, \*\*kwargs)

Bases: [`Prior`](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)

Score via MMSE denoiser $\nabla \reg{x}=\left(x-\operatorname{D}_{\sigma}(x)\right)/\sigma^2$.

This approximates the score of a distribution using Tweedie’s formula, i.e.,

$$
- \nabla \log p_{\sigma}(x) \propto \left(x-D(x,\sigma)\right)/\sigma^2
$$

where $p_{\sigma} = p*\mathcal{N}(0,I\sigma^2)$ is the prior convolved with a Gaussian kernel,
$D(\cdot,\sigma)$ is a (trained or model-based) denoiser with noise level $\sigma$,
which is typically set to a low value.

#### NOTE
If $\sigma=1$, this prior is equal to [`deepinv.optim.RED`](https://deepinv.org/api/stubs/deepinv.optim.RED.html.md#deepinv.optim.RED), which is defined in
Regularization by Denoising (RED) Romano *et al.*<sup>[1](#footcite-romano2017little)</sup> and doesn’t require the normalization.

#### NOTE
This class can also be used with maximum-a-posteriori (MAP) denoisers,
but $p_{\sigma}(x)$ is not given by the convolution with a Gaussian kernel, but rather
given by the Moreau-Yosida envelope of $p(x)$, i.e.,

$$
p_{\sigma}(x)=e^{- \inf_z \left(-\log p(z) + \frac{1}{2\sigma}\|x-z\|^2 \right)}.
$$

<hr />

* **References:**

* <a id='footcite-romano2017little'>**[1]**</a> Yaniv Romano, Michael Elad, and Peyman Milanfar. The little engine that could: regularization by denoising (red). *SIAM Journal on Imaging Sciences*, 10(4):1804–1844, 2017.

#### grad(x, sigma_denoiser, \*args, \*\*kwargs)

Applies the denoiser to the input signal.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the input tensor.
  * **sigma_denoiser** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – the noise level.
* **Returns:**
  (torch.Tensor) gradient at $x$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### score(x, sigma_denoiser, \*args, \*\*kwargs)

Computes the score function $\nabla \log p_\sigma$, using Tweedie’s formula.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the input tensor.
  * **sigma_denoiser** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – the noise level.

#### *static* stable_division(a, b, epsilon=1e-7)

Performs a safe-guarded division by adding a small constant $\epsilon$ to the denominator when it is close to zero.

* **Parameters:**
  * **a** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – numerator.
  * **b** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – denominator.
  * **epsilon** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – small constant added to the denominator when it is close to zero.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) result of the division.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-optim-scoreprior"></a>

## Examples using `ScorePrior`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to build your custom sampling kernel. Here we build a preconditioned Unadjusted Langevin Algorithm (PreconULA) that takes advantage of the singular value decomposition of the forward operator to accelerate the sampling.">  <div class="sphx-glr-thumbnail-title">Building your custom MCMC sampling algorithm.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows you how to use sampling algorithms to quantify uncertainty of a reconstruction from incomplete and noisy measurements.">  <div class="sphx-glr-thumbnail-title">Uncertainty quantification with PnP-ULA.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
