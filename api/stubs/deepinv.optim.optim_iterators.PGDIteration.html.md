# PGDIteration

### *class* deepinv.optim.optim_iterators.PGDIteration(\*\*kwargs)

Bases: [`OptimIterator`](https://deepinv.org/api/stubs/deepinv.optim.OptimIterator.html.md#deepinv.optim.OptimIterator)

Iterator for proximal gradient descent.

Class for a single iteration of the Proximal Gradient Descent (PGD) algorithm for minimizing $f(x) + \lambda \regname(x)$.

The iteration is given by

$$
u_{k} &= x_k -  \gamma \nabla f(x_k) \\
x_{k+1} &= \operatorname{prox}_{\gamma \lambda \regname}(u_k)

$$

where $\gamma$ is a stepsize that should satisfy $\gamma \leq 2/\operatorname{Lip}(\|\nabla f\|)$.
