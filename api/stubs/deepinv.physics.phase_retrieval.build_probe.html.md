# build_probe

### deepinv.physics.phase_retrieval.build_probe(img_size, type='disk', probe_radius=10, device='cpu')

Builds a probe based on the specified type and radius.

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – Shape of the input image.
  * **type** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Type of probe shape, e.g., “disk”.
  * **probe_radius** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Radius of the probe shape.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – Device “cpu” or “gpu”.
* **Returns:**
  Tensor representing the constructed probe.

## Examples using `build_probe`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to create a Ptychography phase retrieval operator and generate phaseless measurements from a given image.">  <div class="sphx-glr-thumbnail-title">Ptychography phase retrieval</div>
</div>
<!-- thumbnail-parent-div-close --></div>
