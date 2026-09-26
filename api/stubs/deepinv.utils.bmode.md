# bmode

### deepinv.utils.bmode(x, dim=-2, , amplitude_floor_db=-60.0, dynamic_range=None, reference=None, normalize=True)

Compute log-compressed brightness mode (B-Mode) image.

$$
\mathrm{B}(x) = 20 \log_{10} \left(\frac{x_a}{x_\mathrm{ref}} \right),

$$

where $x_a$ is the envelope of $x$, i.e. the modulus of its analytical signal
(see [`deepinv.utils.hilbert()`](https://deepinv.org/api/stubs/deepinv.utils.hilbert.md#deepinv.utils.hilbert)) or its modulus if $x$ is complex-valued, and
$x_\mathrm{ref}$ a reference amplitude. The result is clipped to
$[\mathrm{amplitude\_floor\_db}, \mathrm{amplitude\_floor\_db} + \mathrm{dynamic\_range}]$.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input signal of shape `(B, ...)`
  * **dim** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – dimension along which the envelope is computed. (default: `-2`)
  * **amplitude_floor_db** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – lower bound of the display window, in dB relative to the reference. (default: `-60`)
  * **dynamic_range** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – width of the display window in dB. If `None`, the window ends at 0 dB. (default: `None`)
  * **reference** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – reference amplitude mapped to 0 dB. If `None`, the maximum of the envelope of each element of the batch. (default: `None`)
  * **normalize** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, the display window is linearly mapped to `[0, 1]`, which is convenient for display or for saving the image. (default: `True`)
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) the log-compressed image, in dB or in `[0, 1]` if `normalize` is `True`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

## Examples using `bmode`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example presents the plane-wave ultrafast ultrasound forward physics (deepinv.physics.UltrasoundPlaneWave) available in DeepInverse for pulse-echo imaging problems.">![](auto_examples/physics/images/thumb/sphx_glr_demo_ultrasound_tour_thumb.png)

[Tour of ultrafast ultrasound in DeepInverse](https://deepinv.org/auto_examples/physics/demo_ultrasound_tour.md)

  <div class="sphx-glr-thumbnail-title">Tour of ultrafast ultrasound in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use deepinv.physics.UltrasoundPlaneWave to reconstructs an in-vivo carotid acquisition of the EPFL LTS5 ultrafast ultrasound dataset from raw RF ultrasound data.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_ultrasound_invivo_PnP_thumb.png)

[In-vivo ultrafast ultrasound reconstruction with Plug-and-Play](https://deepinv.org/auto_examples/plug-and-play/demo_ultrasound_invivo_PnP.md)

  <div class="sphx-glr-thumbnail-title">In-vivo ultrafast ultrasound reconstruction with Plug-and-Play</div>
</div>
<!-- thumbnail-parent-div-close --></div>
