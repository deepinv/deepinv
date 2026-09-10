# histogram

### deepinv.physics.functional.histogram(x, bins=10, low=None, upp=None, \*\*kwargs)

Computes the histogram of a tensor.

This is a `torch` implementation of `numpy.histogram`.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – A tensor, `(*,)`.
  * **bins** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – The number of bins.
  * **low** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – The lower bound. If `low` is `None` the min of `x` is used instead.
  * **upp** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – The upper bound. If `upp` is `None` the max of `x` is used instead.
  * **kwargs** – Keyword arguments passed to `histogramdd`.
* **Return torch.Tensor:**
  The histogram
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
