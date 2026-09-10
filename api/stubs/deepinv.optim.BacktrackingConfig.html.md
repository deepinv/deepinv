# BacktrackingConfig

### *class* deepinv.optim.BacktrackingConfig(gamma=0.1, eta=0.9, max_iter=20)

Bases: [`object`](https://docs.python.org/3.9/library/functions.html#object)

Configuration parameters for backtracking line search on the stepsize.

* **Parameters:**
  * **gamma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Armijo-like parameter (controls sufficient decrease).
  * **eta** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Step reduction factor (e.g. multiply step by eta on failure).
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Maximum number of backtracking steps.
