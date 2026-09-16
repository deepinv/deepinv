# EquivariantReconstructor

### *class* deepinv.models.EquivariantReconstructor(model, transform=None, eval_transform=None)

Bases: [`Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor)

Equivariant reconstructor

Make a base reconstructor $\tilde{R}$ equivariant by averaging over the transformations, i.e.,

$$
R(y, A) = \frac{1}{|\mathcal{G}|}\sum_{g\in \mathcal{G}} T_g \tilde{R}(y, A T_g)
$$

An equivariant reconstructor is a reconstructor that satisfies <sup>[1](#footcite-sechaud26equivariant)</sup>

$$
R(y, A T_g) = T_g^{-1} R(y, A)
$$

for all $g \in \mathcal{G}$ where $T_g$ is a transform (eg shifts, rotations, etc).

* **Parameters:**
  * **model** ([*Reconstructor*](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor)) – base reconstructor to be made equivariant.
  * **transform** ([*Transform*](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform) *,* *None*) – geometric transformation. By default, it is set to a single random 90° rotation and flip.
  * **eval_transform** ([*Transform*](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform)) – transformations to be used in evaluation mode. It can be used to have true Reynolds averaging at evaluation time and efficient Monte Carlo estimation at training time. By default, if training transformations are specified, evaluation transformations default to them, otherwise they default to the eight 90° rotations and flips.

<hr />

* **References:**

* <a id='footcite-sechaud26equivariant'>**[1]**</a> Victor Sechaud, Jérémy Scanvic, Quentin Barthélemy, Patrice Abry, and Julián Tachella. Equivariant Splitting: Self-supervised learning from incomplete data. In *The Fourteenth International Conference on Learning Representations (ICLR)*. 2026.

#### forward(y, physics, \*reconstructor_args, \*\*reconstructor_kwargs)

Apply the reconstructor to an input

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input measurement.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – physics operator associated with the measurements.
  * **\*reconstructor_args** – args for reconstructor function.
  * **\*\*reconstructor_kwargs** – kwargs for reconstructor function.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) output of the reconstructor.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-models-equivariantreconstructor"></a>

## Examples using `EquivariantReconstructor`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Equivariant splitting consists in minimizing a self-supervised loss to train a reconstruction model using measurement data only :footcitesechaud26Equivariant.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the use of our deepinv.transform module for use in solving imaging problems. These can be used for:">  <div class="sphx-glr-thumbnail-title">Image transforms for equivariance & augmentations</div>
</div>
<!-- thumbnail-parent-div-close --></div>
