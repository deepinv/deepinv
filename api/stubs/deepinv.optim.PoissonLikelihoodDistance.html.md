# PoissonLikelihoodDistance

### *class* deepinv.optim.PoissonLikelihoodDistance(gain=1.0, bkg=0, denormalize=False)

Bases: [`Distance`](https://deepinv.org/api/stubs/deepinv.optim.Distance.html.md#deepinv.optim.Distance)

(Negative) Log-likelihood of the Poisson distribution.

$$
\distance{y}{x} =  \sum_i y_i \log(y_i / x_i) + x_i - y_i
$$

#### NOTE
The function is not Lipschitz smooth w.r.t. $x$ in the absence of background ($\beta=0$).

* **Parameters:**
  * **gain** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – gain of the measurement $y$. Default: 1.0.
  * **bkg** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – background level $\beta$. Default: 0.
  * **denormalize** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if True, the measurement is divided by the gain. By default, in the
    [`deepinv.physics.PoissonNoise`](https://deepinv.org/api/stubs/deepinv.physics.PoissonNoise.html.md#deepinv.physics.PoissonNoise), the measurements are multiplied by the gain after being sampled by
    the Poisson distribution. Default: True.

#### fn(x, y, \*args, \*\*kwargs)

Computes the Kullback-Leibler divergence

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the distance is computed.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Observation $y$.

#### grad(x, y, \*args, \*\*kwargs)

Gradient of the Kullback-Leibler divergence

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – signal $x$ at which the function is computed.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurement $y$.

#### prox(x, y, \*args, gamma=1.0, \*\*kwargs)

Proximal operator of the Kullback-Leibler divergence

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – signal $x$ at which the function is computed.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurement $y$.
  * **gamma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – proximity operator step size.
