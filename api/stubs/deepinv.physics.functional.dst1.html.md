# dst1

### deepinv.physics.functional.dst1(x, , dim=(-1,), inverse=False, orthosf=True)

Compute the one-dimensional [discrete sine transform](https://en.wikipedia.org/wiki/Discrete_sine_transform) of type I (DST-I) or its inverse (IDST-I).

The DST-I of a vector $x$ of length $N$ is defined as

$$
\mathrm{DST-I}(x)_k = - \frac{1}{2} \Im(\mathrm{DFT}(y_(k+1))),
$$

where $y$ is the odd extension of $x$. The IDST-I is defined as

$$
\mathrm{IDST-I}(x)_k = - \frac{1}{N + 1} \Im(\mathrm{DFT}(y_(k+1))).
$$

If `orthosf=True`, it computes the orthogonal sign-flipped DST-I instead:

$$
\mathrm{OSFDST-I}(x)_k = \frac{1}{\sqrt{2N + 2}} \Im(\mathrm{DFT}(y_(k+1))).
$$

#### NOTE
The orthogonal sign-flipped DST-I is its own inverse, hence when `orthosf=True`, we have `dst1(dst1(x)) = x` and the parameter `inverse` has no effect.

#### NOTE
When multiple dimensions are specified, the DST-I is applied to each dimension separably.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input tensor.
  * **dim** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – Dimension along which to compute the transform. Default is `(-1,)` (the last dimension).
  * **inverse** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If True, compute the inverse DST-I (IDST-I). If False (default), compute the DST-I. It has not effect when `ortho=True`.
  * **orthosf** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If True (default), compute the orthogonal sign-flipped DST-I, otherwise compute the standard DST-I.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) The transformed tensor.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
