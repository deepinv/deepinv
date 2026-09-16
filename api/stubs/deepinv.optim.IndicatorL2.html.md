# IndicatorL2

### *class* deepinv.optim.IndicatorL2(radius=None)

Bases: [`DataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)

Data-fidelity as the indicator of $\ell_2$ ball with radius $r$.

$$
\iota_{\mathcal{B}_2(y,r)}(u)= \left.
    \begin{cases}
      0, & \text{if } \|u-y\|_2\leq r \\
      +\infty & \text{else.}
    \end{cases}
    \right.
$$

* **Parameters:**
  **radius** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – radius of the ball. Default: None.

#### prox(x, y, physics, \*args, radius=None, stepsize=None, crit_conv=1e-5, max_iter=100, \*\*kwargs)

Proximal operator of the indicator of $\ell_2$ ball with radius `radius`, i.e.

$$
\operatorname{prox}_{\gamma \iota_{\mathcal{B}_2(y, r)}(A\cdot)}(x) = \underset{u}{\text{argmin}} \,\, \iota_{\mathcal{B}_2(y, r)}(Au)+\frac{1}{2}\|u-x\|_2^2
$$

Since no closed form is available for general measurement operators, we use a dual forward-backward algorithm,
as suggested in Combettes and Pesquet<sup>[1](#footcite-combettes2011proximal)</sup>.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the proximity operator is computed.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Data $y$ of the same dimension as $\forw{x}$.
  * **radius** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – radius of the $\ell_2$ ball.
  * **stepsize** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – step-size of the dual-forward-backward algorithm.
  * **crit_conv** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – convergence criterion of the dual-forward-backward algorithm.
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of iterations of the dual-forward-backward algorithm.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) projection on the $\ell_2$ ball of radius `radius` and centered in `y`.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<hr />

* **References:**

* <a id='footcite-combettes2011proximal'>**[1]**</a> Patrick L Combettes and Jean-Christophe Pesquet. Proximal splitting methods in signal processing. *Fixed-point algorithms for inverse problems in science and engineering*, pages 185–212, 2011.

<a id="sphx-glr-backref-deepinv-optim-indicatorl2"></a>

## Examples using `IndicatorL2`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Image inpainting consists in solving y = Ax where A is a mask operator. This problem can be reformulated as the following minimization problem:">  <div class="sphx-glr-thumbnail-title">Unfolded Chambolle-Pock for constrained image inpainting</div>
</div>
<!-- thumbnail-parent-div-close --></div>
