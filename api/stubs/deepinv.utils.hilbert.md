# hilbert

### deepinv.utils.hilbert(x, dim=-1)

Compute the analytical signal via Hilbert transform.

#### NOTE
This function uses `scipy` and is therefore not efficient nor differentiable. If a
pure `torch` implementation is required, please raise a feature request issue on
[GitHub](https://github.com/deepinv/deepinv/issues).

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – real-valued input signal of arbitrary shape.
  * **dim** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – dimension along which the transform is computed, e.g. the time axis of
    raw data or the depth axis of an image. (default: `-1`)
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) the complex-valued analytical signal, of the same shape
  as `x`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
