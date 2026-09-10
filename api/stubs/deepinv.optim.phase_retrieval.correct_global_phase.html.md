# correct_global_phase

### deepinv.optim.phase_retrieval.correct_global_phase(x_est, x_ref, correct_magnitude=False, dim=(-2, -1), verbose=False)

Corrects the global phase shift (and optionally magnitude scaling) of reconstructed complex signals to match the references.

The optimal global phase shift and magnitude scaling is computed per batch entry and channel as:

$$
\theta = \arg \langle \hat{x}, x \rangle, \quad r = \frac{|\langle \hat{x}, x \rangle|}{\|\hat{x}\|^2},

$$

where $\arg$ denotes the angle of a complex number, $\langle a, b \rangle = a^\mathrm{H} b$ denotes the complex inner product and $\|\cdot\|$ denotes the Euclidean norm.

This computation corresponds to the complex-scalar minimizer $c = r \mathrm{e}^{i \theta}$ of the following program:

$$
\min_{c} \|c \cdot \hat{x} - x\|^2,

$$

where $\hat{x}$ is the reconstructed signal and $x$ is the reference signal.

The correction is then applied to the reconstructed signal per batch entry and channel as:

$$
\hat{x} \leftarrow r' \mathrm{e}^{\mathrm{i} \theta} \cdot \hat{x},

$$

with $r' = r$ if `correct_magnitude` is `True` and $r' = 1$ otherwise.

* **Parameters:**
  * **x_est** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Estimated signals of shape `(N, C, ...)`.
  * **x_ref** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Reference signals of shape `(N, C, ...)`.
  * **correct_magnitude** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, also corrects the magnitude scaling in addition to the phase. Default is `False`.
  * **dim** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – Dimensions of the signals over which the inner product is computed. Default is `(-2, -1)`.
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, prints the applied phase shift and scale factor. Default is `False`.
* **Returns:**
  The phase-corrected (and optionally magnitude-corrected) signals of the same shape as `x_est`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

## Examples using `correct_global_phase`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to create a random phase retrieval operator and generate phaseless measurements from a given image. The example showcases 4 different reconstruction methods to recover the image from the phaseless measurements:">  <div class="sphx-glr-thumbnail-title">Random phase retrieval and reconstruction methods.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to create a Ptychography phase retrieval operator and generate phaseless measurements from a given image.">  <div class="sphx-glr-thumbnail-title">Ptychography phase retrieval</div>
</div>
<!-- thumbnail-parent-div-close --></div>
