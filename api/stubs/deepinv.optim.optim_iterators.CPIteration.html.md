# CPIteration

### *class* deepinv.optim.optim_iterators.CPIteration(\*\*kwargs)

Bases: [`OptimIterator`](https://deepinv.org/api/stubs/deepinv.optim.OptimIterator.html.md#deepinv.optim.OptimIterator)

Iterator for Chambolle-Pock.

Class for a single iteration of the Chambolle-Pock Primal-Dual (PD) algorithm Chambolle and Pock<sup>[1](#footcite-chambolle2011first)</sup> for minimising $F(Kx) + \lambda G(x)$ or $\lambda F(x) + G(Kx)$ for generic functions $F$ and $G$.
Our implementation corresponds to Algorithm 1 of Chambolle and Pock<sup>[1](#footcite-chambolle2011first)</sup>.

If the attribute `g_first` is set to `False` (by default), the iteration is given by

$$
u_{k+1} &= \operatorname{prox}_{\sigma F^*}(u_k + \sigma K z_k) \\
x_{k+1} &= \operatorname{prox}_{\tau \lambda G}(x_k-\tau K^\top u_{k+1}) \\
z_{k+1} &= x_{k+1} + \beta(x_{k+1}-x_k)

$$

where $F^*$ is the Fenchel-Legendre conjugate of $F$, $\beta>0$ is a relaxation parameter, and $\sigma$ and $\tau$ are step-sizes that should
satisfy $\sigma \tau \|K\|^2 \leq 1$.

If the attribute `g_first` is set to `True`, the functions $F$ and $G$ are inverted in the previous iteration.

In particular, setting $F = \distancename$, $K = A$ and $G = \regname$, the above algorithms solves

$$
\underset{x}{\operatorname{min}} \,\,  \distancename(Ax, y) + \lambda \regname(x)
$$

with a splitting on $\distancename$, with not differentiability assumption needed on $\distancename$
or $\regname$, not any invertibility assumption on $A$.

Note that the algorithm requires an intiliazation of the three variables $x_0$, $z_0$ and $u_0$.

<hr />

* **References:**

* <a id='footcite-chambolle2011first'>**[1]**</a> Antonin Chambolle and Thomas Pock. A first-order primal-dual algorithm for convex problems with applications to imaging. *Journal of mathematical imaging and vision*, 40:120–145, 2011.

#### forward(X, cur_data_fidelity, cur_prior, cur_params, y, physics, \*args, \*\*kwargs)

Single iteration of the Chambolle-Pock algorithm.

* **Parameters:**
  * **X** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – Dictionary containing the current iterate and the estimated cost.
  * **cur_data_fidelity** ([*deepinv.optim.DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)) – Instance of the DataFidelity class defining the current data_fidelity.
  * **cur_prior** ([*deepinv.optim.Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)) – Instance of the Prior class defining the current prior.
  * **cur_params** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – dictionary containing the current parameters of the algorithm.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input data.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Instance of the physics modeling the data-fidelity term.
* **Returns:**
  Dictionary `{"est": (x, ), "cost": F}` containing the updated current iterate and the estimated current cost.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)[[str](https://docs.python.org/3.9/library/stdtypes.html#str), [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)] | [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]
