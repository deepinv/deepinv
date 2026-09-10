# DEAL

### *class* deepinv.models.DEAL(sigma_denoiser=0.1, lambda_reg=10.0, max_iter=50, auto_scale=False, target_y_std=25.0, color=False, device=None, clamp_output=True, pretrained='pretrained', inner_iter=200, outer_iter=60)

Bases: [`Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor)

Deep Equilibrium Attention Least Squares (DEAL) reconstruction model.

This model solves linear inverse problems using a learned equilibrium-based
regularizer combined with iterative conjugate gradient least-squares updates.
It can be used for image restoration and reconstruction tasks such as
denoising, deblurring, and computed tomography reconstruction.

This implementation is adapted from the official
[DEAL repository](https://github.com/mehrsapo/DEAL).

For the original method, see Pourya *et al.*<sup>[1](#footcite-pourya2025dealing)</sup>.

A pretrained network can be loaded by setting `pretrained='download'`.

The reconstruction is obtained by solving a regularized least-squares problem

$$
\hat{x} = \arg\min_x \frac{1}{2}\|Ax - y\|^2 + \lambda g_\theta(x)
$$

where $A$ is the forward operator, $y$ the measurements, and
$g_\theta$ is the learned adaptive regularizer.

In the implementation, the learned regularizer is induced by a masked linear
operator of the form

$$
L_{\theta,c}(u, x) = m_{\theta,c}(u) \odot K_{\theta,c} x,
$$

so that the regularization term can be written as

$$
g_{\theta}(u, x) = \sum_{c=1}^{C} \frac{1}{2}\|L_{\theta,c}(u, x)\|_2^2
= \sum_{c=1}^{C} \frac{1}{2}\|m_{\theta,c}(u) \odot K_{\theta,c} x\|_2^2,
$$

where $K_{\theta,c}$ are learned linear filters,
$m_{\theta,c}(u)$ are spatially varying masks predicted by the network,
and $\odot$ denotes element-wise multiplication.

The optimization is performed iteratively using a fixed-point scheme.
At each outer iteration, the algorithm updates the reconstruction by solving
a linearized least-squares subproblem using conjugate gradient:

$$
x^{(k+1)} \approx \arg\min_x
\frac{1}{2}\|Ax - y\|^2
+
\frac{\lambda}{2}
\sum_{c=1}^{C}
\|m_{\theta,c}(x^{(k)}) \odot K_{\theta,c}x\|_2^2.
$$

* **Parameters:**
  * **sigma_denoiser** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – denoiser noise level parameter
  * **lambda_reg** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – regularization strength $\lambda$ used by the DEAL solver
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of outer fixed-point iterations
  * **auto_scale** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, rescales measurements in reconstruction
    mode when their empirical standard deviation is between `0` and `5`.
    This option is useful when measurements are given in a normalized range but
    the pretrained DEAL inverse-problem solver expects a larger intensity scale.
    It is disabled by default.
  * **target_y_std** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – target measurement standard deviation used by
    `auto_scale` when enabled
  * **color** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, use the color DEAL variant; otherwise grayscale
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *|* *None*) – compute device. If `None`, use CUDA if available
  * **clamp_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, clamp output to `[0, 1]`
  * **pretrained** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *|* *None*) – checkpoint path, `'download'`,
    `'pretrained'`, or `None`. If `None`, no pretrained weights are
    loaded. If `'download'` or `'pretrained'`, the official DEAL
    pretrained weights are downloaded and loaded.
  * **inner_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of iterations of the conjugate gradient
    algorithm.
  * **outer_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of inner fixed-point iterations and conjugate gradient

<hr />

* **References:**

* <a id='footcite-pourya2025dealing'>**[1]**</a> Mehrsa Pourya, Erich Kobler, Michael Unser, and Sebastian Neumayer. DEALing with image reconstruction: deep attentive least squares. In *Forty-second International Conference on Machine Learning*. 2025. URL: [https://openreview.net/forum?id=mMasOShOVt](https://openreview.net/forum?id=mMasOShOVt).

#### *property* device *: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*

Return the current device of the internal DEAL module.

#### forward(y, physics=None, sigma=None)

Run DEAL as either a denoiser or a reconstructor.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input measurements
  * **physics** ([*LinearPhysics*](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics) *|* *None*) – forward operator for reconstruction. If
    `None`, DEAL is applied as a denoiser.
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *|* *None*) – denoising noise level used when `physics` is
    `None`
* **Returns:**
  reconstructed or denoised image
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### *property* mask *: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

Return the current DEAL mask.

<a id="sphx-glr-backref-deepinv-models-deal"></a>

## Examples using `DEAL`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the Deep Equilibrium Attention Least Squares (DEAL) model in DeepInverse for both denoising and a simple reconstruction settings.">  <div class="sphx-glr-thumbnail-title">DEAL denoising and reconstruction</div>
</div>
<!-- thumbnail-parent-div-close --></div>
