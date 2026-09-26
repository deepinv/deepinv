# AverageMeter

### *class* deepinv.utils.AverageMeter(name, fmt=':f')

Bases: [`object`](https://docs.python.org/3.9/library/functions.html#object)

Compute and store aggregates online from a stream of scalar values

The supported aggregates are:
- vals: the list of all processed values
- val: the last value processed
- avg: the average of all processed values
- sum: the sum of all processed values
- count: the number of processed values
- std: the standard deviation of all processed values
- sum2: the sum of squares of all processed values

* **Parameters:**
  * **name** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – meter name for printing
  * **fmt** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – meter format for printing

#### reset()

Reset the stored aggregates.

#### update(val, n=1)

Process new scalar value(s) and update the stored aggregates.

* **Parameters:**
  * **val** ([*numpy.ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int)) – either array (i.e. batch) of values or single value
  * **n** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – weight, defaults to 1
