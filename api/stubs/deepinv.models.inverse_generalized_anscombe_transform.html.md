# inverse_generalized_anscombe_transform

### deepinv.models.inverse_generalized_anscombe_transform(x, gain, sigma)

Inverse Generalized Anscombe Transform (IGAT)

The transform converts an approximately Gaussian signal $z$ (output of the
[`generalized_anscombe_transform()`](https://deepinv.org/api/stubs/deepinv.models.generalized_anscombe_transform.html.md#deepinv.models.generalized_anscombe_transform)) back to the original
[`Poisson-Gaussian`](https://deepinv.org/api/stubs/deepinv.physics.PoissonGaussianNoise.html.md#deepinv.physics.PoissonGaussianNoise) domain with
gain $\gamma$ and [`Gaussian noise`](https://deepinv.org/api/stubs/deepinv.physics.GaussianNoise.html.md#deepinv.physics.GaussianNoise)
standard deviation $\sigma$, see Makitalo and Foi<sup>[1](#footcite-makitalo2012optimal)</sup>.

The transform is defined as the algebraic inverse of the
[`generalized_anscombe_transform()`](https://deepinv.org/api/stubs/deepinv.models.generalized_anscombe_transform.html.md#deepinv.models.generalized_anscombe_transform):

$$
h^{-1}(x) = \frac{1}{4}x^2 + \frac{1}{4}\sqrt{\frac{3}{2}}\, x^{-1} - \frac{11}{8} x^{-2} + \frac{5}{8}\sqrt{\frac{3}{2}}\, x^{-3} - \frac{1}{8} - \frac{\sigma^2}{\gamma^2}
$$

#### NOTE
The formula varies slightly from the one proposed in <sup>[1](#footcite-makitalo2012optimal)</sup>,
as the library considers a normalized Poisson-Gaussian noise model, $y = \gamma \mathcal{P}(x/\gamma) + \epsilon$,
whereas the authors consider $y = \gamma \mathcal{P}(x) + \epsilon$.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Anscombe-transformed tensor.
  * **gain** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *|* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Gain of the Poisson distribution $\gamma$
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *|* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Standard deviation of the Gaussian noise $\sigma$
* **Return torch.Tensor:**
  Reconstructed measurements in the original domain

<hr />

* **References:**

* <a id='footcite-makitalo2012optimal'>**[1]**</a> Markku Makitalo and Alessandro Foi. Optimal inversion of the generalized anscombe transformation for poisson-gaussian noise. *IEEE transactions on image processing*, 22(1):91–103, 2012.
