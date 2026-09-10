# load_mat

### deepinv.utils.load_mat(fname, mat73=False, \*\*kwargs)

Load MATLAB array from file.

This function depends on the `scipy` package. You can install it with `pip install scipy`.

* **Parameters:**
  * **fname** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – filename to load
  * **mat73** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if file is MATLAB 7.3 or above, load with `mat73`. Requires
    `mat73`, install with `pip install mat73`.
* **Returns:**
  dict with str keys and numpy array values.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)[[str](https://docs.python.org/3.9/library/stdtypes.html#str), [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)]
