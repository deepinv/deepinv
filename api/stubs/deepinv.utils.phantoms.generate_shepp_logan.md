# generate_shepp_logan

### deepinv.utils.phantoms.generate_shepp_logan(size, channels=0)

Generate a Shepp-Logan phantom approximation in PyTorch.

* **Parameters:**
  * **size** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – square size of phantom.
  * **channels** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – if 0, return tensor of shape (size, size), otherwise (channels, size, size).
