# histogramdd

### deepinv.physics.functional.histogramdd(x, bins=10, low=None, upp=None, bounded=False, weights=None, sparse=False, edges=None)

Computes the multidimensional histogram of a tensor.

This is a `torch` implementation of `numpy.histogramdd`.
This function is borrowed from [torchist](https://github.com/francois-rozet/torchist/).

#### NOTE
Similar to `numpy.histogram`, all bins are half-open except the last bin which
also includes the upper bound.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – A tensor, (\*, D).
  * **bins** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – The number of bins in each dimension, scalar or (D,).
  * **low** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *]*) – The lower bound in each dimension, scalar or (D,). If `low` is `None`,
    the min of `x` is used instead.
  * **upp** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *]*) – The upper bound in each dimension, scalar or (D,). If `upp` is `None`,
    the max of `x` is used instead.
  * **bounded** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Whether `x` is bounded by `low` and `upp`, included.
    If `False`, out-of-bounds values are filtered out.
  * **weights** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – A tensor of weights, `(\*,)`. Each sample of `x` contributes
    its associated weight towards the bin count (instead of 1).
  * **sparse** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Whether the histogram is returned as a sparse tensor or not.
  * **edges** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – The edges of the histogram. Either a vector or a list of vectors.
    If provided, `bins`, `low` and `upp` are inferred from `edges`.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) : the histogram
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
