# AmplitudeLoss

### *class* deepinv.optim.AmplitudeLoss

Bases: [`DataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.md#deepinv.optim.DataFidelity)

Amplitude loss as the data fidelity term for [`deepinv.physics.PhaseRetrieval()`](https://deepinv.org/api/stubs/deepinv.physics.PhaseRetrieval.md#deepinv.physics.PhaseRetrieval) reconstrunction.

In this case, the data fidelity term is defined as

$$
f(x) = \sum_{i=1}^{m}{(\sqrt{|b_i x|^2}-\sqrt{y_i})^2},
$$

where $b_i$ is the i-th row of the linear operator $B$ of the phase retrieval class and $y_i$ is the i-th entry of the measurements, and $m$ is the number of measurements.

<a id="sphx-glr-backref-deepinv-optim-amplitudeloss"></a>

## Examples using `AmplitudeLoss`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Ptychography is a coherent imaging technique for recovering a complex-valued object, including both its amplitude and phase, from intensity-only measurements. A localized illumination, called the probe, is scanned across overlapping regions of the object. At each position, a detector records the intensity after the exit wave has propagated from the object plane. The overlap provides redundant measurements of the same object regions, making it possible to recover phase information that is not measured directly by the detector.">![](auto_examples/physics/images/thumb/sphx_glr_demo_ptychography_thumb.png)

[Ptychography phase retrieval](https://deepinv.org/auto_examples/physics/demo_ptychography.md)

  <div class="sphx-glr-thumbnail-title">Ptychography phase retrieval</div>
</div>
<!-- thumbnail-parent-div-close --></div>
