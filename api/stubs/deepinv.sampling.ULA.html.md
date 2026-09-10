# ULA

### *class* deepinv.sampling.ULA(prior, data_fidelity, step_size=1.0, sigma=0.05, alpha=1.0, max_iter=1e3, thinning=5, burnin_ratio=0.2, clip=(-1.0, 2.0), thresh_conv=1e-3, save_chain=False, verbose=False)

Bases: [`BaseSampling`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSampling.html.md#deepinv.sampling.BaseSampling)

Projected Plug-and-Play Unadjusted Langevin Algorithm.

The algorithm runs the following markov chain iteration (Algorithm 2 from Laumont *et al.*<sup>[1](#footcite-laumont2022bayesian)</sup>):

$$
x_{k+1} = \Pi_{[a,b]} \left(x_{k} + \eta \nabla \log p(y|A,x_k) +
\eta \alpha \nabla \log p(x_{k}) + \sqrt{2\eta}z_{k+1} \right).
$$

where $x_{k}$ is the $k$ th sample of the Markov chain,
$\log p(y|x)$ is the log-likelihood function, $\log p(x)$ is the log-prior,
$\eta>0$ is the step size, $\alpha>0$ controls the amount of regularization,
$\Pi_{[a,b]}(x)$ projects the entries of $x$ to the interval $[a,b]$ and
$z\sim \mathcal{N}(0,I)$ is a standard Gaussian vector.

- Projected PnP-ULA assumes that the denoiser is $L$-Lipschitz differentiable
- For convergence, ULA required step_size smaller than $\frac{1}{L+\|A\|_2^2}$

#### WARNING
This a legacy class provided for convenience. See the example in [Markov Chain Monte Carlo](https://deepinv.org/user_guide/reconstruction/sampling.html.md#mcmc) for details on how to build a ULA sampler.

* **Parameters:**
  * **prior** ([*deepinv.optim.ScorePrior*](https://deepinv.org/api/stubs/deepinv.optim.ScorePrior.html.md#deepinv.optim.ScorePrior) *,* [*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – negative log-prior based on a trained or model-based denoiser.
  * **data_fidelity** ([*deepinv.optim.DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity) *,* [*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – negative log-likelihood function linked with the
    noise distribution in the acquisition physics.
  * **step_size** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – step size $\eta>0$ of the algorithm.
    Tip: use [`deepinv.physics.LinearPhysics.compute_norm()`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics.compute_norm) to compute the Lipschitz constant of a linear forward operator.
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – noise level used in the plug-and-play prior denoiser. A larger value of sigma will result in
    a more regularized reconstruction.
  * **alpha** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – regularization parameter $\alpha$
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of Monte Carlo iterations.
  * **thinning** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Thins the Markov Chain by an integer $\geq 1$ (i.e., keeping one out of `thinning`
    samples to compute posterior statistics).
  * **burnin_ratio** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – percentage of iterations used for burn-in period, should be set between 0 and 1.
    The burn-in samples are discarded constant with a numerical algorithm.
  * **clip** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – Tuple containing the box-constraints $[a,b]$.
    If `None`, the algorithm will not project the samples.
  * **crit_conv** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Threshold for verifying the convergence of the mean and variance estimates.
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – prints progress of the algorithm.

<hr />

* **References:**

* <a id='footcite-laumont2022bayesian'>**[1]**</a> Rémi Laumont, Valentin De Bortoli, Andrés Almansa, Julie Delon, Alain Durmus, and Marcelo Pereyra. Bayesian imaging using plug & play priors: when langevin meets tweedie. *SIAM Journal on Imaging Sciences*, 15(2):701–737, 2022.

#### forward(y, physics, seed=None, x_init=None, g_statistics=lambda d: ...)

Runs the chain to obtain the posterior mean and variance of the reconstruction of the measurements y.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurements
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Forward operator associated with the measurements
  * **seed** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Random seed for generating the Monte Carlo samples
  * **g_statistics** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[**Callable* *]*  *|* *Callable*) – List of functions for which to compute posterior statistics, or a single function.
    The sampler will compute the posterior mean and variance of each function in the list. Note the sampler outputs a dictionary so they must act on `d["x"]`.
    Default: `lambda d: d["x"]` (identity function)
* **Returns:**
  (tuple of torch.Tensor) containing the posterior mean and variance.
