# test_pad

### deepinv.models.utils.test_pad(model, L, modulo=16, padding_mode='replicate')

Pads the image to fit the model’s expected image size.

* **Parameters:**
  * **model** – model to run on the padded image.
  * **L** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image.
  * **modulo** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – spatial dimensions are padded up to a multiple of `modulo`.
  * **padding_mode** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str)) – padding mode passed to [`torch.nn.functional.pad()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pad.html#torch.nn.functional.pad), e.g. `"replicate"` (default, as in DPIR),
    `"reflect"` or `"circular"`. `"reflect"` and `"circular"` require the padding to be smaller than the corresponding
    input dimension; if this is not the case, the padding falls back to `"replicate"`.
