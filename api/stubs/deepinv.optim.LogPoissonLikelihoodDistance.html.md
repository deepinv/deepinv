# LogPoissonLikelihoodDistance

### *class* deepinv.optim.LogPoissonLikelihoodDistance(N0=1024.0, mu=1 / 50.0)

Bases: [`Distance`](https://deepinv.org/api/stubs/deepinv.optim.Distance.html.md#deepinv.optim.Distance)

Log-Poisson negative log-likelihood.

$$
\distance{z}{y} =  N_0 (1^{\top} \exp(-\mu z)+ \mu \exp(-\mu y)^{\top}x)
$$

Corresponds to LogPoissonNoise with the same arguments N0 and mu.
There is no closed-form of the prox known.

* **Parameters:**
  * **N0** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – average number of photons
  * **mu** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – normalization constant
