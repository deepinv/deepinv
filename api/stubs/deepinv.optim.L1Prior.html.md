# L1Prior

### *class* deepinv.optim.L1Prior(\*args, \*\*kwargs)

Bases: [`Prior`](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)

$\ell_1$ prior $\reg{x} = \| x \|_1$.

#### fn(x, \*args, \*\*kwargs)

Computes the regularizer $\reg{x} = \| x \|_1$.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the prior is computed.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) prior $\reg{x}$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### prox(x, \*args, ths=1.0, gamma=1.0, \*\*kwargs)

Calculates the proximity operator of the l1 regularization term $\regname$ at $x$.

More precisely, it computes

$$
\operatorname{prox}_{\gamma g}(x) = \operatorname{sign}(x) \max(|x| - \gamma, 0)

$$

where $\gamma$ is a stepsize.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the proximity operator is computed.
  * **gamma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – stepsize of the proximity operator.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) proximity operator at $x$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
