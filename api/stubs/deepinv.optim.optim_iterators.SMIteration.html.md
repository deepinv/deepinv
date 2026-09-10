# SMIteration

### *class* deepinv.optim.optim_iterators.SMIteration(lamb=10, n_iter=50, preprocessing=lambda x: ..., \*\*kwargs)

Bases: [`OptimIterator`](https://deepinv.org/api/stubs/deepinv.optim.OptimIterator.html.md#deepinv.optim.OptimIterator)

Iterator for Spectral Methods for [`deepinv.physics.PhaseRetrieval`](https://deepinv.org/api/stubs/deepinv.physics.PhaseRetrieval.html.md#deepinv.physics.PhaseRetrieval).

Class for a single iteration of the Spectral Methods algorithm to find the principal eigenvector of the regularized weighted covariance matrix:

$$
M = \conj{B} \text{diag}(T(y)) B + \lambda I,

$$

where $B$ is the linear operator of the phase retrieval class, $T(\cdot)$ is a preprocessing function for the measurements, and $I$ is the identity matrix of corresponding dimensions. Parameter $\lambda$ tunes the strength of regularization.

The iteration is given by

$$
x_{k+1} &= M x_k \\
x_{k+1} &= \operatorname{prox}_{\gamma g}(x_{k+1})

$$

where $\gamma$ is a stepsize that should satisfy $\lambda \gamma \leq 2/\operatorname{Lip}(\|\nabla f\|)$.

#### forward(x, cur_prior, cur_params, y, physics, \*args)

Single iteration of the spectral method.

* **Parameters:**
  * **x** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – the current iterate $x_k$.
  * **cur_prior** ([*deepinv.optim.Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)) – Instance of the Prior class defining the current prior.
  * **cur_params** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – Dictionary containing the current parameters of the algorithm.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input data.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Instance of the physics containing the forward operator.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) The new iterate $x_{k+1}$.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
