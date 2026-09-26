# load_image

### deepinv.utils.load_image(path, img_size=None, grayscale=False, resize_mode='crop', device='cpu', dtype=torch.float32)

Load an image from a file and return a torch.Tensor with a batch dimension.

* **Parameters:**
  * **path** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Path to the image file.
  * **img_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – Size of the image to return.
  * **grayscale** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Whether to convert the image to grayscale.
  * **resize_mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – If `img_size` is not None, options are `"crop"` or `"resize"`.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device on which to load the image (gpu or cpu).
* **Returns:**
  [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) containing the image with an added batch dimension.
