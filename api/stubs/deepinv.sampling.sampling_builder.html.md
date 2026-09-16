# sampling_builder

### deepinv.sampling.sampling_builder(iterator, data_fidelity, prior, params_algo=MappingProxyType({}), max_iter=100, thresh_conv=1e-3, burnin_ratio=0.2, thinning=10, history_size=5, verbose=False, callback=lambda X, \*\*kwargs: ..., \*\*kwargs)

Helper function for building an instance of the [`deepinv.sampling.BaseSampling`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSampling.html.md#deepinv.sampling.BaseSampling) class.

See [Uncertainty quantification with PnP-ULA.](https://deepinv.org/auto_examples/sampling/demo_sampling.html.md#sphx-glr-auto-examples-sampling-demo-sampling-py) and [Markov Chain Monte Carlo](https://deepinv.org/user_guide/reconstruction/sampling.html.md#mcmc) for example usage.

See the docs for [`deepinv.sampling.BaseSampling`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSampling.html.md#deepinv.sampling.BaseSampling) for further examples and information.

* **Parameters:**
  * **iterator** ([*SamplingIterator*](https://deepinv.org/api/stubs/deepinv.sampling.SamplingIterator.html.md#deepinv.sampling.SamplingIterator) *|* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Either a SamplingIterator instance or a string naming the iterator class
  * **data_fidelity** ([*DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)) – Negative log-likelihood function
  * **prior** ([*Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)) – Negative log-prior
  * **params_algo** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – Dictionary containing the parameters for the algorithm
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of Monte Carlo iterations
  * **burnin_ratio** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Percentage of iterations for burn-in
  * **thinning** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Integer to thin the Monte Carlo samples
  * **history_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* [*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Number of most recent samples to store in memory. If `True`, all samples are stored. If `False`, no samples are stored. If an integer, it specifies the number of most recent samples to store. Default: 5
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Whether to print progress
  * **callback** (*Callable*) – A function that is called on every (thinned) sample state dictionary for diagnostics. It is called with the current sample `X`, the current `statistics` (a list of Welford objects), and the current iteration number `iter` as keyword arguments.
  * **kwargs** – Additional keyword arguments passed to the iterator constructor when a string is provided as the iterator parameter
* **Returns:**
  Configured BaseSampling instance in eval mode
* **Return type:**
  [*BaseSampling*](https://deepinv.org/api/stubs/deepinv.sampling.BaseSampling.html.md#deepinv.sampling.BaseSampling)

## Examples using `sampling_builder`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to build your custom sampling kernel. Here we build a preconditioned Unadjusted Langevin Algorithm (PreconULA) that takes advantage of the singular value decomposition of the forward operator to accelerate the sampling.">  <div class="sphx-glr-thumbnail-title">Building your custom MCMC sampling algorithm.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows you how to use sampling algorithms to quantify uncertainty of a reconstruction from incomplete and noisy measurements.">  <div class="sphx-glr-thumbnail-title">Uncertainty quantification with PnP-ULA.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
