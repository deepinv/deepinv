# ULAIterator

### *class* deepinv.sampling.ULAIterator(algo_params, clip=None)

Bases: [`SamplingIterator`](https://deepinv.org/api/stubs/deepinv.sampling.SamplingIterator.html.md#deepinv.sampling.SamplingIterator)

Projected Plug-and-Play Unadjusted Langevin Algorithm.

The algorithm, introduced by Laumont *et al.* [[73](https://deepinv.org/user_guide/other/biblio.html.md#id63)] runs the following markov chain iteration
(Algorithm 2 from [[73](https://deepinv.org/user_guide/other/biblio.html.md#id63)]):

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
- For convergence, ULA requires that `step_size` is smaller than $\frac{1}{L+\|A\|_2^2}$

* **Parameters:**
  * **clip** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *(*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,*[*int*](https://docs.python.org/3.9/library/functions.html#int) *)*) – Tuple of (min, max) values to clip/project the samples into a bounded range during sampling.
    Useful for images where pixel values should stay within a specific range (e.g., (0,1) or (0,255)). Default: `None`
  * **algo_params** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – Dictionary containing the algorithm parameters (see table below)

| Parameter   | Type   | Description                                      |
|-------------|--------|--------------------------------------------------|
| step_size   | float  | Step size $\eta$ (default: 1.0)                  |
| alpha       | float  | Regularization parameter $\alpha$ (default: 1.0) |
| sigma       | float  | Noise level for the score model (default: 0.05)  |
* **Returns:**
  Next state $X_{t+1}$ in the Markov chain
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### forward(X, y, physics, cur_data_fidelity, cur_prior, iteration, \*args, \*\*kwargs)

Performs a single ULA sampling step using the Unadjusted Langevin Algorithm.

Computes the next state in the Markov chain using the formula:

$$
x_{t+1} = x_t + \eta \nabla \log p(y|A,x_t) + \eta \alpha \nabla \log p(x_t) + \sqrt{2\eta}z_{t+1}
$$

where $z_{t+1} \sim \mathcal{N}(0,I)$ is a standard Gaussian noise vector.

* **Parameters:**
  * **X** (*Dict*) – Dictionary containing the current state $x_t$.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Observed measurements/data tensor
  * **physics** ([*Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Forward operator $A$ that models the measurement process
  * **cur_data_fidelity** ([*DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)) – Negative log-likelihood function
  * **cur_prior** ([*ScorePrior*](https://deepinv.org/api/stubs/deepinv.optim.ScorePrior.html.md#deepinv.optim.ScorePrior)) – Score-based prior model for $\nabla \log p(x)$
  * **iteration** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Current iteration number in the sampling process (zero-indexed)
* **Returns:**
  Dictionary `{"x": x}` containing the next state $x_{t+1}$ in the Markov chain.
* **Return type:**
  Dict

<a id="sphx-glr-backref-deepinv-sampling-ulaiterator"></a>

## Examples using `ULAIterator`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to build your custom sampling kernel. Here we build a preconditioned Unadjusted Langevin Algorithm (PreconULA) that takes advantage of the singular value decomposition of the forward operator to accelerate the sampling.">  <div class="sphx-glr-thumbnail-title">Building your custom MCMC sampling algorithm.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows you how to use sampling algorithms to quantify uncertainty of a reconstruction from incomplete and noisy measurements.">  <div class="sphx-glr-thumbnail-title">Uncertainty quantification with PnP-ULA.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
