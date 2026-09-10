# SIRTIteration

### *class* deepinv.optim.optim_iterators.SIRTIteration(\*\*kwargs)

Bases: [`OptimIterator`](https://deepinv.org/api/stubs/deepinv.optim.OptimIterator.html.md#deepinv.optim.OptimIterator)

Iterator for the Simultaneous Iterative Reconstruction Technique (SIRT) algorithm.

Class for a single iteration of the SIRT algorithm.

The iteration is given by:

$$
\begin{equation*}
x_{k+1} = x_k + \tau V A^{\top} W (y - A x_k)
\end{equation*}

$$

where

- $\tau$ is a stepsize parameter which should satisfy $0 < \tau < 2$
- $W = \mathrm{diag}\left(\frac{1}{\sum_{i}a_{ij}}\right)$, a diagonal matrix where each element is the inverse of the row sums of $A$,
- $V = \mathrm{diag}\left(\frac{1}{\sum_{j}a_{ij}}\right)$, a diagonal matrix where each element is the inverse of the column sums of $A$.

#### forward(X, cur_data_fidelity, cur_prior, cur_params, y, physics, \*\*kwargs)

Computes a single iteration of the SIRT algorithm.

:param : param dict X: Dictionary containing the current iterate and the estimated cost.
:param : param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
:param : param deepinv.optim.Prior cur_prior: Instance of the Prior class defining the current prior.
:param : param dict cur_params: Dictionary containing the current parameters of the algorithm.
:param : param torch.Tensor y: Input data.
:param : param deepinv.physics.Physics physics: Instance of the physics modeling the observation.
:param : return: Dictionary `{"est": (x,), "cost": F}` containing the updated current iterate and the estimated current cost.
