# SIRT

### *class* deepinv.optim.SIRT(data_fidelity=None, stepsize=1.0, max_iter=100, crit_conv='residual', thres_conv=1e-5, early_stop=False, custom_metrics=None, custom_init=None, \*\*kwargs)

Bases: [`BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim)

Simultaneous Iterative Reconstruction Technique (SIRT) optimization module.

Implementation of the Simultaneous Iterative Reconstruction Technique (SIRT)
<sup>[1](#footcite-gilbert-iterative-1972)</sup>
algorithm for tomographic reconstruction. This algorithm is especially used in transmission tomography, i.e for X-ray computed tomography.
The algorithm minimizes a weighted least-squares problem of the form $\|Ax-y\|_{W}^2$ and does not support any regularization.

Iterations are given by

$$
\begin{equation*}
x_{k+1} = x_k + \tau V A^{\top} W (y - A x_k)
\end{equation*}

$$

where

- $\tau$ is a stepsize parameter,
- $W = \mathrm{diag}\left(\frac{1}{\sum_{i}a_{ij}}\right)$, a diagonal matrix where each element is the inverse of the row sums of $A$,
- $V = \mathrm{diag}\left(\frac{1}{\sum_{j}a_{ij}}\right)$, a diagonal matrix where each element is the inverse of the column sums of $A$.

The stepsize parameter $\tau$, sometimes called relaxation parameter, should satisfy $0 < \tau < 2$ for convergence.

The entries of $W$ are inversely proportional to the length traveled by each ray. The algorithm tolerates larger errors for measurements induced by rays that intersect a larger portion of the object. The entries of $V$ are inversely proportional to the number of rays intersecting each voxel. It balances the update based on the voxels’ sensitivity to the measurements.

For using early stopping or stepsize backtracking, see the documentation of the [`deepinv.optim.BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim) class.
The SIRT iterations are defined in the iterator class [`deepinv.optim.optim_iterators.SIRTIteration`](https://deepinv.org/api/stubs/deepinv.optim.optim_iterators.SIRTIteration.html.md#deepinv.optim.optim_iterators.SIRTIteration).

* **Parameters:**
  * **data_fidelity** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *,* [*deepinv.optim.DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)) – data-fidelity term $\datafid{x}{y}$. Note that SIRT only decreases the least-squares data-fidelity term $\|Ax-y\|_2^2$, but other data-fidelities can still be measured along the iterations.
  * **stepsize** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – stepsize parameter $\tau$. Default: `1.0`.
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of iterations of the optimization algorithm. Default: `100`.
  * **crit_conv** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – convergence criterion to be used for claiming convergence, either `"residual"` (residual
    of the iterate norm) or `"cost"` (on the cost function). Default: `"residual"`.
  * **thres_conv** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – convergence threshold for the chosen convergence criterion. Default: `1e-5`.
  * **early_stop** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to stop the algorithm as soon as the convergence criterion is met. Default: `False`.
  * **custom_metrics** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – dictionary of custom metric functions to be computed along the iterations. The keys of the dictionary are the names of the metrics, and the values are functions that take as input the current and previous iterates, and return a scalar value. Default: `None`.
  * **custom_init** (*Callable*) – Custom initialization of the algorithm.

<hr />

* **References:**

* <a id='footcite-gilbert-iterative-1972'>**[1]**</a> Peter Gilbert. Iterative methods for the three-dimensional reconstruction of an object from projections. *Journal of Theoretical Biology*, 36(1):105–117, 1972.
