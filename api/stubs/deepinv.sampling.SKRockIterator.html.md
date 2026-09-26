# SKRockIterator

### *class* deepinv.sampling.SKRockIterator(algo_params, clip=None)

Bases: [`SamplingIterator`](https://deepinv.org/api/stubs/deepinv.sampling.SamplingIterator.html.md#deepinv.sampling.SamplingIterator)

Single iteration of the SK-ROCK (Stabilized Runge-Kutta-Chebyshev) Algorithm.

Obtains samples of the posterior distribution using an orthogonal Runge-Kutta-Chebyshev stochastic
approximation to accelerate the standard Unadjusted Langevin Algorithm.

The algorithm was introduced in Pereyra *et al.*<sup>[1](#footcite-pereyra2020accelerating)</sup>.

- SKROCK assumes that the denoiser is $L$-Lipschitz differentiable
- For convergence, SKROCK requires that `step_size` smaller than $\frac{1}{L+\|A\|_2^2}$

* **Parameters:**
  * **clip** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *(*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,*[*int*](https://docs.python.org/3.9/library/functions.html#int) *)*) – Tuple of (min, max) values to clip/project the samples into a bounded range during sampling.
    Useful for images where pixel values should stay within a specific range (e.g., (0,1) or (0,255)). Default: `None`
  * **algo_params** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – Dictionary containing the algorithm parameters (see table below)

| Parameter   | Type   | Description                                                                                                                        |
|-------------|--------|------------------------------------------------------------------------------------------------------------------------------------|
| step_size   | float  | Step size of the algorithm (default: 1.0). Tip: use physics.lipschitz to compute the Lipschitz constant                            |
| alpha       | float  | Regularization parameter $\alpha$ (default: 1.0)                                                                                   |
| inner_iter  | int    | Number of internal iterations (default: 10)                                                                                        |
| eta         | float  | Damping parameter $\eta$ (default: 0.05)                                                                                           |
| sigma       | float  | Noise level for the score prior denoiser (default: 0.05). A larger value of sigma will result in a more regularized reconstruction |

<hr />

* **References:**

* <a id='footcite-pereyra2020accelerating'>**[1]**</a> Marcelo Pereyra, Luis Vargas Mieles, and Konstantinos C Zygalakis. Accelerating proximal markov chain monte carlo by using an explicit stabilized method. *SIAM Journal on Imaging Sciences*, 13(2):905–935, 2020.

#### forward(X, y, physics, cur_data_fidelity, cur_prior, iteration, \*args, \*\*kwargs)

Performs a single SK-ROCK sampling step.

* **Parameters:**
  * **X** (*Dict*) – Dictionary containing the current state $x_t$.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Observed measurements/data tensor
  * **physics** ([*Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Forward operator
  * **cur_data_fidelity** ([*DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)) – Negative log-likelihood function
  * **cur_prior** ([*ScorePrior*](https://deepinv.org/api/stubs/deepinv.optim.ScorePrior.html.md#deepinv.optim.ScorePrior)) – Prior
* **Returns:**
  Dictionary `{"x": x}` containing the next state $x_{t+1}$ in the Markov chain.
* **Return type:**
  Dict
