# RotateViaShear

### *class* deepinv.transform.RotateViaShear(\*args, limits=360.0, multiples=1.0, positive=False, center=None, \*\*kwargs)

Bases: [`Transform`](https://deepinv.org/api/stubs/deepinv.transform.Transform.md#deepinv.transform.Transform)

2D rotations implemented via shear composition through FFT.

Generates `n_trans` randomly rotated versions of 2D images using
FFT-based shear operations.

Picks integer angles between `-limits` and `limits`, by default
-360 to 360 degrees. Set `positive=True` to use only positive angles.

* **Parameters:**
  * **limits** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – rotation angles are sampled in the range
    `(-limits, limits)`.
  * **multiples** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – angles are sampled in multiples of this value.
    Defaults to 1.
  * **positive** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – if `True`, only use positive angles.
  * **center** ([*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.12/library/functions.html#int) *,* [*int*](https://docs.python.org/3.12/library/functions.html#int) *]* *,* *None*) – center of rotation. If `None`,
    use the center of the image.
  * **n_trans** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – number of transformed versions generated per input image.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – random number generator.
