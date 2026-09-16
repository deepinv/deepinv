# create_strategy

### deepinv.distributed.strategies.create_strategy(img_size, tiling_dims=None, \*\*kwargs)

Create a distributed signal strategy.

* **Parameters:**
  * **img_size** (*Sequence* *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – full shape of the signal tensor, including batch and channel dimensions (e.g., `(B, C, H, W)`).
  * **tiling_dims** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,*  *...* *]*  *|* *None*) – dimensions to tile. If `None`, defaults to last N dimensions.
* **Returns:**
  the created strategy instance.
* **Return type:**
  [*DistributedSignalStrategy*](https://deepinv.org/api/stubs/deepinv.distributed.strategies.DistributedSignalStrategy.html.md#deepinv.distributed.strategies.DistributedSignalStrategy)
