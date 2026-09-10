# RandomNoise

### *class* deepinv.transform.RandomNoise(\*args, noise_type='gaussian', sigma=0.1, \*\*kwargs)

Bases: [`Transform`](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform)

Random noise transform.

For now, only Gaussian noise is supported. Override this class and replace the `sigma` parameter for other noise models.

This transform is reproducible: for given param dict `noise_model`, the transform is deterministic.

Note the inverse transform is not well-defined for this transform.

* **Parameters:**
  * **noise_type** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – noise distribution, currently only supports Gaussian noise.
  * **sigma** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – noise parameter or range to pick randomly.
