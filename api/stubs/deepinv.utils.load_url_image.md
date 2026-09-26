# load_url_image

### deepinv.utils.load_url_image(url=None, img_size=None, grayscale=False, resize_mode='crop', device='cpu', dtype=torch.float32)

Load an image from a URL and return a torch.Tensor with a batch dimension.

* **Parameters:**
  * **url** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – URL of the image file.
  * **img_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – Size of the image to return.
  * **grayscale** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Whether to convert the image to grayscale.
  * **resize_mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – If `img_size` is not None, options are `"crop"` or `"resize"`.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device on which to load the image (gpu or cpu).
* **Returns:**
  [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) containing the image with an added batch dimension.

## Examples using `load_url_image`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.physics.Physics together with automatic differentiation to estimate unknown parameters of your forward operator.">![](auto_examples/blind-inverse-problems/images/thumb/sphx_glr_demo_optimizing_physics_parameter_thumb.png)

[Calibrating physics operators](https://deepinv.org/auto_examples/blind-inverse-problems/demo_optimizing_physics_parameter.md)

  <div class="sphx-glr-thumbnail-title">Calibrating physics operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo illustrates the impact of different Hadamard pattern ordering algorithms in the Single Pixel Camera (SPC), a computational imaging system that uses a single photodetector to capture images by projecting the scene onto a series of patterns. The SPC is implemented in the DeepInverse library.">![](auto_examples/physics/images/thumb/sphx_glr_demo_spc_thumb.png)

[Pattern Ordering in a Compressive Single Pixel Camera](https://deepinv.org/auto_examples/physics/demo_spc.md)

  <div class="sphx-glr-thumbnail-title">Pattern Ordering in a Compressive Single Pixel Camera</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example compares six approximations of the measurement-matching term used by diffusion posterior samplers.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_noisy_data_fidelity_thumb.png)

[Noisy data-fidelity terms for diffusion posterior sampling](https://deepinv.org/auto_examples/sampling/demo_noisy_data_fidelity.md)

  <div class="sphx-glr-thumbnail-title">Noisy data-fidelity terms for diffusion posterior sampling</div>
</div>
<!-- thumbnail-parent-div-close --></div>
