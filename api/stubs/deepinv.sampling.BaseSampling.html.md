# BaseSampling

### *class* deepinv.sampling.BaseSampling(iterator, data_fidelity, prior, max_iter=100, callback=lambda X, \*\*kwargs: ..., burnin_ratio=0.2, thresh_conv=1e-3, crit_conv='residual', thinning=10, history_size=5, verbose=False)

Bases: [`Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor)

Base class for Monte Carlo sampling.

This class aims to sample from the posterior distribution $p(x|y)$, where $y$ represents the observed
measurements and $x$ is the (unknown) image to be reconstructed. The sampling process generates a
sequence of states (samples) $X_0, X_1, \ldots, X_N$ from a Markov chain. Each state $X_k$ contains the
current estimate of the unknown image, denoted $x_k$, and may include other latent variables.
The class then computes statistics (e.g., image posterior mean, image posterior variance) from the samples $X_k$.

This class can be used to create new Monte Carlo samplers by implementing the sampling kernel through [`deepinv.sampling.SamplingIterator`](https://deepinv.org/api/stubs/deepinv.sampling.SamplingIterator.html.md#deepinv.sampling.SamplingIterator):

```default
# define your sampler (possibly a Markov kernel which depends on the previous sample)
class MyIterator(SamplingIterator):
    def __init__(self):
        super().__init__()

    def initialize_latent_variables(x, y, physics, data_fidelity, prior):
        # initialize a latent variable
        latent_z = g(x, y, physics, data_fidelity, prior)
        return {"x": x, "z": latent_z}

    def forward(self, X, y, physics, data_fidelity, prior, params_algo):
        # run one sampling kernel iteration
        new_X = f(X, y, physics, data_fidelity, prior, params_algo)
        return new_X

# create the sampler
sampler = BaseSampling(MyIterator(), prior, data_fidelity, iterator_params)

# compute posterior mean and variance of reconstruction of x
mean, var = sampler.sample(y, physics)
```

This class computes the mean and variance of the chain using Welford’s algorithm, which avoids storing the whole
Monte Carlo samples. It can also maintain a history of the `history_size` most recent samples.

Note on retained sample calculation:
: With the default parameters (max_iter=100, burnin_ratio=0.2, thinning=10), the number
  of samples actually used for statistics is calculated as follows:
  <br/>
  - Total iterations: 100
  - Burn-in period: 100 \* 0.2 = 20 iterations (discarded)
  - Remaining iterations: 80
  - With thinning of 10, we keep iterations 20, 30, 40, 50, 60, 70, 80, 90
  - This results in 8 retained samples used for computing the posterior statistics

* **Parameters:**
  * **iterator** ([*deepinv.sampling.SamplingIterator*](https://deepinv.org/api/stubs/deepinv.sampling.SamplingIterator.html.md#deepinv.sampling.SamplingIterator)) – The sampling iterator that defines the MCMC kernel
  * **data_fidelity** ([*deepinv.optim.DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)) – Negative log-likelihood function linked with the noise distribution in the acquisition physics
  * **prior** ([*deepinv.optim.Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)) – Negative log-prior
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – The number of Monte Carlo iterations to perform. Default: 100
  * **burnin_ratio** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Percentage of iterations used for burn-in period (between 0 and 1). Default: 0.2
  * **thinning** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Integer to thin the Monte Carlo samples (keeping one out of `thinning` samples). Default: 10
  * **thresh_conv** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – The convergence threshold for the mean and variance. Default: `1e-3`
  * **callback** (*Callable*) – A function that is called on every (thinned) sample state dictionary for diagnostics. It is called with the current sample `X`, the current `statistics` (a list of Welford objects), and the current iteration number `iter` as keyword arguments.
  * **history_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* [*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Number of most recent samples to store in memory. If `True`, all samples are stored. If `False`, no samples are stored. If an integer, it specifies the number of most recent samples to store. Default: 5
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Whether to print progress of the algorithm. Default: `False`

#### forward(y, physics, x_init=None, seed=None, \*\*kwargs)

Run the MCMC sampling chain and return the posterior sample mean.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – The observed measurements
  * **physics** ([*Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Forward operator of your inverse problem
  * **x_init** (*Union* *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict) *,* *None* *]*) – Optional initial state of the Markov chain. This can be a `torch.Tensor` to initialize the image `X["x"]`, or a `dict` to initialize the entire state `X` including any latent variables. In most cases, providing a tensor to initialize `X["x"]` will be sufficient.
    Default: `None`
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Optional random seed for reproducible sampling.
    Default: `None`
* **Returns:**
  Posterior sample mean
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### get_chain()

Retrieve the stored history of samples.

Returns a list of dictionaries, where each dictionary contains the state of the sampler.

Only includes samples after the burn-in period and thinning.

* **Returns:**
  List of stored sample states (dictionaries) from oldest to newest. Each dictionary contains the sample `"x": x` along with any latent variables.
* **Return type:**
  [list](https://docs.python.org/3.9/library/stdtypes.html#list)[[dict](https://docs.python.org/3.9/library/stdtypes.html#dict)]
* **Raises:**
  [**RuntimeError**](https://docs.python.org/3.9/library/exceptions.html#RuntimeError) – If history storage was disabled (history_size=False)

Example:

```default
from deepinv.sampling import BaseSampling, SamplingIterator

sampler = BaseSampling(SamplingIterator(...), data_fidelity, prior, history_size=5)
_ = sampler(measurements, forward_operator)
history = sampler.get_chain()
latest_state = history[-1]  # Get most recent state dictionary
latest_sample = latest_state["x"] # Get sample from state
```

#### *property* mean_has_converged *: [bool](https://docs.python.org/3.9/library/functions.html#bool)*

Returns a boolean indicating if the posterior mean verifies the convergence criteria.

#### sample(y, physics, x_init=None, seed=None, g_statistics=(lambda d: ...,), \*\*kwargs)

Execute the MCMC sampling chain and compute posterior statistics.

This method runs the main MCMC sampling loop to generate samples from the posterior
distribution and compute their statistics using Welford’s online algorithm.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – The observed measurements/data tensor
  * **physics** ([*Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Forward operator of your inverse problem.
  * **x_init** (*Union* *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict) *,* *None* *]*) – Optional initial state of the Markov chain. This can be a `torch.Tensor` to initialize the image `X["x"]`, or a `dict` to initialize the entire state `X` including any latent variables. In most cases, providing a tensor to initialize `X["x"]` will be sufficient.
    Default: `None`
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Optional random seed for reproducible sampling.
    Default: `None`
  * **g_statistics** (*Union* *[**List* *[**Callable* *]* *,* *Callable* *]*) – List of functions for which to compute posterior statistics.
    The sampler will compute the posterior mean and variance of each function in the list.
    The input to these functions is a dictionary `d` which contains the current state of the sampler alongside any latent variables. `d["x"]` will always be the current image. See specific iterators for details on what (if any) latent variables they provide.
    Default: `lambda d: d["x"]` (identity function on the image).
  * **g_statistics** – List of functions for which to compute posterior statistics, or a single function.
  * **kwargs** – Additional arguments passed to the sampling iterator (e.g., proposal distributions)
* **Returns:**
  If a single g_statistic was specified: Returns tuple (mean, var) of torch.Tensors
  <br/>
  If multiple g_statistics were specified: Returns tuple (means, vars) of lists of torch.Tensors
  <br/>
* **Return type:**
  [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]

Example:

```default
from deepinv.sampling import BaseSampling, ULAIterator

iterator = ULAIterator(...) # define iterator

# Basic usage with default settings
sampler = BaseSampling(iterator, data_fidelity, prior)
mean, var = sampler.sample(measurements, forward_operator)

# Using multiple statistics
sampler = BaseSampling(
    iterator, data_fidelity, prior,
    g_statistics=[lambda X: X["x"], lambda X: X["x"]**2]
)
means, vars = sampler.sample(measurements, forward_operator)
```

#### *property* var_has_converged *: [bool](https://docs.python.org/3.9/library/functions.html#bool)*

Returns a boolean indicating if the posterior variance verifies the convergence criteria.

<a id="sphx-glr-backref-deepinv-sampling-basesampling"></a>

## Examples using `BaseSampling`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to build your custom sampling kernel. Here we build a preconditioned Unadjusted Langevin Algorithm (PreconULA) that takes advantage of the singular value decomposition of the forward operator to accelerate the sampling.">  <div class="sphx-glr-thumbnail-title">Building your custom MCMC sampling algorithm.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows you how to use the DDRM diffusion algorithm :footcitekawar2022denoising to reconstruct images and also compute the uncertainty of a reconstruction from incomplete and noisy measurements.">  <div class="sphx-glr-thumbnail-title">Image reconstruction with a diffusion model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows you how to use sampling algorithms to quantify uncertainty of a reconstruction from incomplete and noisy measurements.">  <div class="sphx-glr-thumbnail-title">Uncertainty quantification with PnP-ULA.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
