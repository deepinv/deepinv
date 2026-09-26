# L12Prior

### *class* deepinv.optim.L12Prior(\*args, l2_axis=-1, \*\*kwargs)

Bases: [`Prior`](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)

$\ell_{1,2}$ prior $\reg{x} = \sum_i\| x_i \|_2$.

The $\ell_2$ norm is computed over a tensor axis that can be defined by the user. By default, `l2_axis=-1`.

* **Parameters:**
  **l2_axis** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – dimension in which the $\ell_2$ norm is computed.

<hr />

* **Examples:**

```pycon
>>> import torch
>>> from deepinv.optim import L12Prior
>>> seed = torch.manual_seed(0) # Random seed for reproducibility
>>> x = torch.randn(2, 1, 3, 3) # Define random 3x3 image
>>> prior = L12Prior()
>>> prior.fn(x)
tensor([5.4949, 4.3881])
>>> prior.prox(x)
tensor([[[[-0.4666, -0.4776,  0.2348],
          [ 0.3636,  0.2744, -0.7125],
          [-0.1655,  0.8986,  0.2270]]],


        [[[-0.0000, -0.0000,  0.0000],
          [ 0.7883,  0.9000,  0.5369],
          [-0.3695,  0.4081,  0.5513]]]])
```

#### fn(x, \*args, \*\*kwargs)

Computes the regularizer $\reg{x} = \sum_i\| x_i \|_2$.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the prior is computed.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) prior $\reg{x}$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### prox(x, \*args, gamma=1.0, \*\*kwargs)

Calculates the proximity operator of the $\ell_{1,2}$ function at $x$.

More precisely, it computes

$$
\operatorname{prox}_{\gamma g}(x) = (1 - \frac{\gamma}{\mathrm{max}(\Vert x \Vert_2,\gamma)}) x

$$

where $\gamma$ is a stepsize.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the proximity operator is computed.
  * **gamma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – stepsize of the proximity operator.
  * **l2_axis** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – axis in which the l2 norm is computed.
* **Return torch.Tensor:**
  proximity operator at $x$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
