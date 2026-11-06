# multiplier_adjoint

### deepinv.physics.functional.multiplier_adjoint(x, mult)

Implements the adjoint of diagonal matrices or multipliers $x$ and `mult`.

The adjoint of this operation is [`deepinv.physics.functional.multiplier()`](https://deepinv.org/api/stubs/deepinv.physics.functional.multiplier.html.md#deepinv.physics.functional.multiplier)

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Image of size `(B, C, ...)`.
  * **filter** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Filter of size `(b, c, ...)` where `b` can be either `1` or `B` and `c` can be either `1` or `C`.

If `b = 1` or `c = 1`, then this function supports broadcasting as the same as [numpy](https://numpy.org/doc/stable/user/basics.broadcasting.html).

:return torch.Tensor : the output of the multiplier, same shape as $x$
