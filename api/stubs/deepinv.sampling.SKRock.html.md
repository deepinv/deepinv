# SKRock

### *class* deepinv.sampling.SKRock(prior, data_fidelity, step_size=1.0, inner_iter=10, eta=0.05, alpha=1.0, max_iter=1e3, burnin_ratio=0.2, thinning=10, clip=(-1.0, 2.0), thresh_conv=1e-3, save_chain=False, verbose=False, sigma=0.05)

Bases: [`BaseSampling`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSampling.html.md#deepinv.sampling.BaseSampling)

Plug-and-Play SKROCK algorithm.

Obtains samples of the posterior distribution using an orthogonal Runge-Kutta-Chebyshev stochastic
approximation to accelerate the standard Unadjusted Langevin Algorithm.

The algorithm was introduced by Pereyra *et al.*<sup>[1](#footcite-pereyra2020accelerating)</sup>.

- SKROCK assumes that the denoiser is $L$-Lipschitz differentiable
- For convergence, SKROCK required step_size smaller than $\frac{1}{L+\|A\|_2^2}$

#### WARNING
This a legacy class provided for convenience. See the example in [Markov Chain Monte Carlo](https://deepinv.org/user_guide/reconstruction/sampling.html.md#mcmc) for details on how to build a SKRock sampler.

* **Parameters:**
  * **prior** ([*deepinv.optim.ScorePrior*](https://deepinv.org/api/stubs/deepinv.optim.ScorePrior.html.md#deepinv.optim.ScorePrior) *,* [*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – negative log-prior based on a trained or model-based denoiser.
  * **data_fidelity** ([*deepinv.optim.DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity) *,* [*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – negative log-likelihood function linked with the
    noise distribution in the acquisition physics.
  * **step_size** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Step size of the algorithm. Tip: use physics.lipschitz to compute the Lipschitz
  * **eta** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – $\eta$ SKROCK damping parameter.
  * **alpha** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – regularization parameter $\alpha$.
  * **inner_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of inner SKROCK iterations.
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of outer iterations.
  * **thinning** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Thins the Markov Chain by an integer $\geq 1$ (i.e., keeping one out of `thinning`
    samples to compute posterior statistics).
  * **burnin_ratio** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – percentage of iterations used for burn-in period. The burn-in samples are discarded
    constant with a numerical algorithm.
  * **clip** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – Tuple containing the box-constraints $[a,b]$.
    If `None`, the algorithm will not project the samples.
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – prints progress of the algorithm.
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – noise level used in the plug-and-play prior denoiser. A larger value of sigma will result in
    a more regularized reconstruction.

<hr />

* **References:**

* <a id='footcite-pereyra2020accelerating'>**[1]**</a> Marcelo Pereyra, Luis Vargas Mieles, and Konstantinos C Zygalakis. Accelerating proximal markov chain monte carlo by using an explicit stabilized method. *SIAM Journal on Imaging Sciences*, 13(2):905–935, 2020.

#### forward(y, physics, seed=None, x_init=None, g_statistics=lambda d: ...)

Runs the chain to obtain the posterior mean and variance of the reconstruction of the measurements y.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurements
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Forward operator associated with the measurements
  * **seed** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Random seed for generating the Monte Carlo samples
  * **g_statistics** (*List* *[**Callable* *]*  *|* *Callable*) – List of functions for which to compute posterior statistics, or a single function.
    The sampler will compute the posterior mean and variance of each function in the list. Note the sampler outputs a dictionary so they must act on `d["x"]`.
    Default: `lambda d: d["x"]` (identity function)
* **Returns:**
  (tuple of torch.Tensor) containing the posterior mean and variance.
* **Return type:**
  [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]
