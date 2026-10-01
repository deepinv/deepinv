# PMDIteration

### *class* deepinv.optim.optim_iterators.PMDIteration(bregman_potential=None, \*\*kwargs)

Bases: [`OptimIterator`](https://deepinv.org/api/stubs/deepinv.optim.OptimIterator.html.md#deepinv.optim.OptimIterator)

Iterator for Proximal Mirror Descent (PMD).

Class for a single iteration of the Proximal Mirror Descent (PMD) algorithm for minimizing $f(x) + \lambda \regname(x)$.

For a given Bregman convex potential $h$, the iteration is given by

$$
u_{k} &= \nabla h^*(\nabla h(x_k) - \gamma \nabla f(x_k)) \\
x_{k+1} &= \operatorname{prox^h}_{\gamma \lambda \regname}(u_k)

$$

where $\gamma$ is a stepsize that should satisfy $\gamma \leq 2/L$ with $L$ verifying $Lh-f$ is convex.
The potential $h$ should be specified in the cur_params dictionary.
