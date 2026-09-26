# random_choice

### deepinv.physics.functional.random_choice(a, size=None, replace=True, p=None, rng=None)

PyTorch equivalent of [`numpy.random.choice()`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html#numpy.random.choice)

* **Parameters:**
  * **a** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the 1-D input tensor
  * **size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – output shape.
  * **replace** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to sample with replacement. Default is True, meaning that a value of `a` can be selected multiple times.
  * **p** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the probabilities for each entry in `a`. If not given, the sample assumes a uniform distribution over all entries in `a`.
* **Returns:**
  the generated random samples in the same device as `a`.

<hr />

* **Examples:**
  ```pycon
  >>> import torch
  >>> from deepinv.physics.functional import random_choice
  >>> a = torch.tensor([1.,2.,3.,4.,5.])
  >>> p = torch.tensor([0,0,1.,0,0])
  >>> print(random_choice(a, 2, replace=True, p=p))
  tensor([3., 3.])
  ```
