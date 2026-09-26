# HQSIteration

### *class* deepinv.optim.optim_iterators.HQSIteration(\*\*kwargs)

Bases: [`OptimIterator`](https://deepinv.org/api/stubs/deepinv.optim.OptimIterator.html.md#deepinv.optim.OptimIterator)

Single iteration of half-quadratic splitting.

Class for a single iteration of the Half-Quadratic Splitting (HQS) algorithm for minimising $f(x) + \lambda \regname(x)$.
The iteration is given by

$$
u_{k} &= \operatorname{prox}_{\gamma f}(x_k) \\
x_{k+1} &= \operatorname{prox}_{\sigma \lambda \regname}(u_k).

$$

where $\gamma$ and $\sigma$ are step-sizes. Note that this algorithm does not converge to
a minimizer of $f(x) + \lambda  \regname(x)$, but instead to a minimizer of
$\gamma\, ^1f+\sigma \lambda \regname$, where $^1f$ denotes
the Moreau envelope of $f$
