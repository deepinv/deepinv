# RandomPhantomDataset

### *class* deepinv.utils.phantoms.RandomPhantomDataset(length, size=128, n_data=1, transform=None, use_dict_output=False)

Bases: [`Dataset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)

Dataset of random ellipsoid phantoms generated on the fly.

* **Parameters:**
  * **length** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Length of the dataset.
  * **size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Size of the phantom (square) image.
  * **n_data** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of phantoms to generate per sample.
  * **transform** (*Callable*) – Transformation to apply to the output image.
  * **use_dict_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Whether to return output as dict with key “x” or a bare Tensor (default: `False`).
