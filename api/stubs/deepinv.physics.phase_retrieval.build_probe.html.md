# build_probe

### deepinv.physics.phase_retrieval.build_probe(img_size, type='disk', probe_radius=10, device='cpu')

Builds a probe based on the specified type and radius.

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple)) – Shape of the input image.
  * **type** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str)) – Type of probe shape, e.g., “disk”.
  * **probe_radius** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – Radius of the probe shape.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – Device `"cpu"` or `"cuda"`.
* **Returns:**
  Tensor representing the constructed probe.

## Examples using `build_probe`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Ptychography is a coherent imaging technique for recovering a complex-valued object, including both its amplitude and phase, from intensity-only measurements. A localized illumination, called the probe, is scanned across overlapping regions of the object. At each position, a detector records the intensity after the exit wave has propagated from the object plane. The overlap provides redundant measurements of the same object regions, making it possible to recover phase information that is not measured directly by the detector.">![](auto_examples/physics/images/thumb/sphx_glr_demo_ptychography_thumb.png)

[Ptychography phase retrieval](https://deepinv.org/auto_examples/physics/demo_ptychography.html.md)

  <div class="sphx-glr-thumbnail-title">Ptychography phase retrieval</div>
</div>
<!-- thumbnail-parent-div-close --></div>
