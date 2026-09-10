# ShiftTime

### *class* deepinv.transform.ShiftTime(\*args, padding='reflect', \*\*kwargs)

Bases: [`Transform`](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform)

Shift a video in time with reflective padding.

Generates `n_trans` randomly transformed versions.

See [`deepinv.transform.Transform`](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform) for further details and examples.

* **Parameters:**
  * **n_trans** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of transformed versions generated per input image.
  * **padding** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – `"reflect"` performs reflective padding, `"wrap"` performs wrap padding (i.e. roll)
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – random number generator, if None, use torch.Generator(), defaults to None

#### roll_reflect_1d(x, by=0, dim=0)

Roll in one dimension with reflect padding.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image
  * **by** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – amount to roll by, defaults to 0
  * **dim** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – dimension to roll, defaults to 0
