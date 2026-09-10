# IndicatorL2Distance

### *class* deepinv.optim.IndicatorL2Distance(radius=None)

Bases: [`Distance`](https://deepinv.org/api/stubs/deepinv.optim.Distance.html.md#deepinv.optim.Distance)

Indicator of $\ell_2$ ball with radius $r$.

The indicator function of the $\ell_2$ ball with radius $r$, denoted as $\iota_{\mathcal{B}_2(y,r)(x)}$,
is defined as

$$
\iota_{\mathcal{B}_2(y,r)}(x)= \left.
    \begin{cases}
      0, & \text{if } \|x-y\|_2\leq r \\
      +\infty & \text{else.}
    \end{cases}
    \right.
$$

* **Parameters:**
  **radius** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – radius of the ball. Default: None.

#### fn(x, y, \*args, radius=None, \*\*kwargs)

Computes the batched indicator of $\ell_2$ ball with radius `radius`, i.e. $\iota_{\mathcal{B}(y,r)}(x)$.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the indicator is computed. $u$ is assumed to be of shape (B, …) where B is the batch size.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Observation $y$ of the same dimension as $u$.
  * **radius** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – radius of the $\ell_2$ ball. If `radius` is None, the radius of the ball is set to `self.radius`. Default: None.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) indicator of $\ell_2$ ball with radius `radius`. If the point is inside the ball, the output is 0, else it is 1e16.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### prox(x, y, \*args, radius=None, gamma=None, \*\*kwargs)

Proximal operator of the indicator of $\ell_2$ ball with radius `radius`, i.e.

$$
\operatorname{prox}_{\iota_{\mathcal{B}_2(y,r)}}(x) = \operatorname{proj}_{\mathcal{B}_2(y, r)}(x)
$$

where $\operatorname{proj}_{C}(x)$ denotes the projection on the closed convex set $C$.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the proximity operator is computed.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Observation $y$ of the same dimension as $x$.
  * **gamma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – step-size. Note that this parameter is not used in this function.
  * **radius** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – radius of the $\ell_2$ ball.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) projection on the $\ell_2$ ball of radius `radius` and centered in `y`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
