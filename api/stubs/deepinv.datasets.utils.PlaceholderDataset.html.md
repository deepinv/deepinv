# PlaceholderDataset

### *class* deepinv.datasets.utils.PlaceholderDataset(n=1, shape=(1, 64, 64))

Bases: [`ImageDataset`](https://deepinv.org/api/stubs/deepinv.datasets.ImageDataset.html.md#deepinv.datasets.ImageDataset)

A placeholder dataset for test purposes.

Produces image pairs x,y that are random tensor of shape specified.

* **Parameters:**
  * **n** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of samples in dataset, defaults to 1
  * **shape** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – image shape, (channel, height, width), defaults to (1, 64, 64)
