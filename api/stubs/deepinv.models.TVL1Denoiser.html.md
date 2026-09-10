# TVL1Denoiser

### *class* deepinv.models.TVL1Denoiser(verbose=False, tau=0.01, rho=1.99, n_it_max=1000, crit=1e-5, x2=None, u2=None, ths=None)

Bases: [`TVDenoiser`](https://deepinv.org/api/stubs/deepinv.models.TVDenoiser.html.md#deepinv.models.TVDenoiser)

Compute the proximal operator of the conjugate TV-L1 regularization term.

This operator projects the input tensor onto the interval
[-lambda2, lambda2] by clamping each element independently.

* **Parameters:**
  * **u** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input dual variable tensor.
  * **lambda2** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *or* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Clamping threshold defining
    the projection bounds.
* **Returns:**
  Tensor with all values constrained to the
  interval [-lambda2, lambda2].
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
