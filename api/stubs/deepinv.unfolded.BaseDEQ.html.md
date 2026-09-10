# BaseDEQ

### *class* deepinv.unfolded.BaseDEQ(\*args, max_iter_backward=50, anderson_acceleration_backward=False, history_size_backward=5, beta_anderson_acc_backward=1.0, eps_anderson_acc_backward=1e-4, jacobian_free=False, \*\*kwargs)

Bases: [`BaseUnfold`](https://deepinv.org/api/stubs/deepinv.unfolded.BaseUnfold.html.md#deepinv.unfolded.BaseUnfold)

Base class for deep equilibrium (DEQ) algorithms. Child of [`deepinv.unfolded.BaseUnfold`](https://deepinv.org/api/stubs/deepinv.unfolded.BaseUnfold.html.md#deepinv.unfolded.BaseUnfold).

#### Deprecated
Deprecated since version 0.3.6: The `BaseDEQ` class is deprecated and will be removed in future versions.
Instead of using this function, define a DEQ algorithm using the [`deepinv.optim.BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim) class with argument `DEQ=True`,
e.g. `model = PGD(data_fidelity, prior, ..., DEQ = True, ...)`.

Enables to turn any fixed-point algorithm into a DEQ algorithm, i.e. an algorithm
that can be virtually unrolled infinitely, leveraging the implicit function theorem.
The backward pass is performed using fixed point iterations to find solutions of the fixed-point equation

$$
\begin{equation}
v = \left(\frac{\partial \operatorname{FixedPoint}(x^\star)}{\partial x^\star} \right )^{\top} v + u.
\end{equation}
$$

where $u$ is the incoming gradient from the backward pass,
and $x^\star$ is the equilibrium point of the forward pass.

See [this tutorial](http://implicit-layers-tutorial.org/deep_equilibrium_models/) for more details.

#### NOTE
For now DEQ is only possible with PGD, HQS and GD optimization algorithms.

* **Parameters:**
  * **max_iter_backward** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Maximum number of backward iterations. Default: `50`.
  * **anderson_acceleration_backward** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if True, the Anderson acceleration is used at iteration of fixed-point algorithm for computing the backward pass. Default: `False`.
  * **history_size_backward** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – size of the history used for the Anderson acceleration for the backward pass. Default: `5`.
  * **beta_anderson_acc_backward** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – momentum of the Anderson acceleration step for the backward pass. Default: `1.0`.
  * **eps_anderson_acc_backward** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – regularization parameter of the Anderson acceleration step for the backward pass. Default: `1e-4`.
  * **jacobian_free** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Does not inverse the Jacobian but simply uses `v=u`.

#### forward(y, physics, x_gt=None, compute_metrics=False, \*\*kwargs)

The forward pass of the DEQ algorithm. Compared to [`deepinv.unfolded.BaseUnfold`](https://deepinv.org/api/stubs/deepinv.unfolded.BaseUnfold.html.md#deepinv.unfolded.BaseUnfold), the backward algorithm is performed using fixed point iterations.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input tensor.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Physics object.
  * **x_gt** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – (optional) ground truth image, for plotting the PSNR across optim iterations.
  * **compute_metrics** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to compute the metrics or not. Default: `False`.
* **Returns:**
  If `compute_metrics` is `False`,  returns ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) the output of the algorithm.
  Else, returns ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), dict) the output of the algorithm and the metrics.
