# SamplingIterator

### *class* deepinv.sampling.SamplingIterator(algo_params, \*\*kwargs)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Base class for sampling iterators.

All samplers should implement the `forward` method which performs one step of the Markov chain Monte Carlo sampling process,
generating the next state $X_{t+1}$ given the current state $X_t$.
Where $X_t$ is a `dict` containing the image $x_t$ as well as any latent variables.
See the docs for [`deepinv.sampling.BaseSampling`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSampling.html.md#deepinv.sampling.BaseSampling) for an example along with more information.

* **Parameters:**
  **algo_params** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – Dictionary containing the parameters for the sampling algorithm

#### forward(X, y, physics, cur_data_fidelity, cur_prior, iteration, \*args, \*\*kwargs)

Performs a single sampling step: $X_t \rightarrow X_{t+1}, where :math:`X_t$ is a `dict` containing the image $x_t$ as well as any latents\`

* **Parameters:**
  * **X** (*Dict*) – Dictionary containing the current image $X_t$ of the Markov chain along with any latent variables.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Observed measurements/data tensor
  * **physics** ([*Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Forward operator
  * **cur_data_fidelity** ([*DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)) – Negative log-likelihood
  * **cur_prior** ([*Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)) – Negative log-prior term
  * **iteration** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Current iteration number in the sampling process (zero-indexed)
  * **args** – Additional positional arguments
  * **kwargs** – Additional keyword arguments
* **Returns:**
  Dictionary `{"x": x, ...}` containing the next state along with any latent variables.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)[[str](https://docs.python.org/3.9/library/stdtypes.html#str), [*Any*](https://docs.python.org/3.9/library/typing.html#typing.Any)]

#### initialize_latent_variables(x_init, y, physics, cur_data_fidelity, cur_prior)

Initializes latent variables for the sampling iterator.

This method is intended to be overridden by subclasses to initialize any latent variables
required by the specific sampling algorithm. The default implementation simply returns the
initial state `x` in a dictionary.

* **Parameters:**
  * **x_init** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Initial state tensor.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Observed measurements/data tensor.
  * **physics** ([*Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Forward operator.
  * **cur_data_fidelity** ([*DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)) – Negative log-likelihood.
  * **cur_prior** ([*Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)) – Negative log-prior term.
* **Returns:**
  Dictionary containing the initial state `x` and any latent variables.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)[[str](https://docs.python.org/3.9/library/stdtypes.html#str), [*Any*](https://docs.python.org/3.9/library/typing.html#typing.Any)]

<a id="sphx-glr-backref-deepinv-sampling-samplingiterator"></a>

## Examples using `SamplingIterator`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to build your custom sampling kernel. Here we build a preconditioned Unadjusted Langevin Algorithm (PreconULA) that takes advantage of the singular value decomposition of the forward operator to accelerate the sampling.">  <div class="sphx-glr-thumbnail-title">Building your custom MCMC sampling algorithm.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
