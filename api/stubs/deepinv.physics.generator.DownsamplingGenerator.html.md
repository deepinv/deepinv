# DownsamplingGenerator

### *class* deepinv.physics.generator.DownsamplingGenerator(filters=('gaussian', 'bilinear', 'bicubic'), factors=(2, 4), psf_size=None, rng=None, device='cpu', dtype=torch.float32)

Bases: [`PhysicsGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.PhysicsGenerator.html.md#deepinv.physics.generator.PhysicsGenerator)

Random downsampling generator.

Generates random downsampling factors and filters.
This can be used for generating parameters to be passed to the
[`Downsampling`](https://deepinv.org/api/stubs/deepinv.physics.Downsampling.html.md#deepinv.physics.Downsampling) class.

```pycon
>>> from deepinv.physics.generator import DownsamplingGenerator
>>> list_filters = ["bilinear", "bicubic", "gaussian"]
>>> list_factors = [2, 4]
>>> generator = DownsamplingGenerator(filters=list_filters, factors=list_factors)
>>> ds = generator.step(batch_size=1)  # dict_keys(['filter', 'factor'])
>>> filter = ds['filter']
>>> factor = ds['factor']
```

#### NOTE
If batch size = 1, a random filter and factor is sampled in (filters, factors) at each step.
If batch size > 1 and multiple factors are provided, a unique factor is sampled for the whole batch, but filters can vary. In this case,
it is recommended to set the `psf_size` argument to ensure that all filters in the batch have the same shape.

* **Parameters:**
  * **filters** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *]*) – list of filters to use for downsampling. Default is [“gaussian”, “bilinear”, “bicubic”].
  * **factors** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – list of factors to use for downsampling. Default is [2, 4].
  * **psf_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – size of the point spread function (PSF) to use for the filters, necessary to stack different filters. If None, the default size of the filter from the filter functions will be used. Default is None.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – random number generator. Default is None.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – device to use. Default is “cpu”.
  * **dtype** ([*type*](https://docs.python.org/3.9/library/functions.html#type)) – data type to use. Default is torch.float32.

#### get_kernel(filter_str=None, factor=None)

Returns a batched tensor of filters associated to a given filter name and factor.

* **Parameters:**
  * **filter_str** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – filter name. Default is None.
  * **factor** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – downsampling factor. Default is None.

#### step(batch_size=1, seed=None)

Generates a random downsampling factor and filter.

* **Parameters:**
  * **batch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – batch size. Default is 1.
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – seed for random number generator. Default is None.

#### str2filter(filter_name, factor)

Returns the filter associated to a given filter name and factor.
