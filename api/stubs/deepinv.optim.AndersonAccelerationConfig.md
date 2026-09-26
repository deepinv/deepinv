# AndersonAccelerationConfig

### *class* deepinv.optim.AndersonAccelerationConfig(history_size=10, beta=0.9, eps=0.1, full_backprop=False)

Bases: [`object`](https://docs.python.org/3.9/library/functions.html#object)

Configuration parameters for Anderson acceleration of a fixed-point algorithm.

* **Parameters:**
  * **history_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of past iterates used in Anderson acceleration.
  * **beta** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Momentum coefficient in Anderson acceleration.
  * **eps** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Regularization parameter for Anderson acceleration.
  * **full_backprop** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Compute backpropagation through all iterates of Anderson acceleration instead of the last iterate only. Default: `False`.
