# DEQConfig

### *class* deepinv.optim.DEQConfig(jacobian_free=False, anderson_acceleration_backward=False, history_size_backward=5, beta_backward=1.0, eps_backward=0.0001, max_iter_backward=50)

Bases: [`object`](https://docs.python.org/3.9/library/functions.html#object)

Configuration parameters for Deep Equilibrium models.

* **Parameters:**
  * **jacobian_free** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Whether to use a Jacobian-free backward pass (see Fung *et al.*<sup>[1](#footcite-fung2022jfb)</sup>).
  * **anderson_acceleration_backward** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Whether to use Anderson acceleration for solving the backward equilibrium.
  * **history_size_backward** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of past iterates used in Anderson acceleration for the backward pass.
  * **beta_backward** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Momentum coefficient in Anderson acceleration for the backward pass.
  * **eps_backward** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Regularization parameter for Anderson acceleration in the backward pass.
  * **max_iter_backward** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Maximum number of iterations in the backward equilibrium solver.

<hr />

* **References:**

* <a id='footcite-fung2022jfb'>**[1]**</a> Samy Wu Fung, Howard Heaton, Qiuwei Li, Daniel McKenzie, Stanley Osher, and Wotao Yin. Jfb: jacobian-free backpropagation for implicit networks. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 36, 6648–6656. 2022.
