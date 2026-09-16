# generate_shifts

### deepinv.physics.phase_retrieval.generate_shifts(img_size, n_img=25, fov=None)

Generates the array of probe shifts across the image.
Based on probe radius and field of view.

* **Parameters:**
  * **img_size** ([*Any*](https://docs.python.org/3.9/library/typing.html#typing.Any)) – Size of the image.
  * **n_img** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of shifts (must be a perfect square).
  * **fov** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Field of view for shift computation.
* **Returns:**
  Array of (x, y) shifts.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

## Examples using `generate_shifts`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to create a Ptychography phase retrieval operator and generate phaseless measurements from a given image.">  <div class="sphx-glr-thumbnail-title">Ptychography phase retrieval</div>
</div>
<!-- thumbnail-parent-div-close --></div>
