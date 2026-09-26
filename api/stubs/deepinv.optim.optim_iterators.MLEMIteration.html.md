# MLEMIteration

### *class* deepinv.optim.optim_iterators.MLEMIteration(eps=1e-6, cost_fn=None, \*\*kwargs)

Bases: [`OptimIterator`](https://deepinv.org/api/stubs/deepinv.optim.OptimIterator.html.md#deepinv.optim.OptimIterator)

Iterator for the Maximum-Likelihood Expectation-Maximization (MLEM) algorithm for Poisson inverse problems.

Class for a single iteration of the MLEM algorithm <sup>[1](#footcite-sheppmaximumlikelihoodreconstruction1982)</sup>,
which is a classic baseline reconstruction method for inverse problems with Poisson noise statistics.
More details on the algorithm can be found in the documentation of the [`deepinv.optim.optimizers.MLEM`](https://deepinv.org/api/stubs/deepinv.optim.MLEM.html.md#deepinv.optim.MLEM) optimizer.

<hr />

* **References:**

* <a id='footcite-sheppmaximumlikelihoodreconstruction1982'>**[1]**</a> Lawrence A Shepp and Yehuda Vardi. Maximum likelihood reconstruction for emission tomography. *IEEE Transactions on Medical Imaging*, 1(2):113–122, 1982.

#### forward(X, cur_data_fidelity, cur_prior, cur_params, y, physics, sensitivity, \*args, \*\*kwargs)

Single Maximum-Likelihood Expectation-Maximization (MLEM) iteration.

This corresponds to an update on both the Poisson negative log-likelihood and prior terms if a prior is provided, and only on Poisson negative log-likelihood otherwise.

* **Parameters:**
  * **X** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – Dictionary containing the current iterate and the estimated cost.
  * **cur_data_fidelity** ([*deepinv.optim.DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)) – Instance of the DataFidelity class defining the current data_fidelity.
  * **cur_prior** ([*deepinv.optim.Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)) – Instance of the Prior class defining the current prior.
  * **cur_params** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – Dictionary containing the current parameters of the algorithm.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input data.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Instance of the physics modeling the data-fidelity term.
  * **sensitivity** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Precomputed sensitivity map $A^T \mathbf{1}$.
