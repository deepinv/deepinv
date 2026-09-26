# load_blosc2

### deepinv.utils.load_blosc2(fname, as_memmap=False, dtype=np.float32, \*\*kwargs)

Load volume from blosc2 file as torch tensor.

* **Parameters:**
  * **fname** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – file to load.
  * **as_memmap** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool) *,*) – open this file as a memory-mapped array (which does not load the entire array into memory). This is useful when extracting patches from large arrays or to quickly infer dtype and shape.
  * **dtype** ([*numpy.dtype*](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html#numpy.dtype) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – data type to use when loading the blosc2 file. This is ignored if `as_memmap` is `True`.
* **Returns:**
  [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) containing loaded numpy array. If `as_memmap` is `True`, returns a blosc2 array object instead.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | blosc2.ndarray.NDArray
