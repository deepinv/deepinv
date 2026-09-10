# EquivariantDenoiser

### *class* deepinv.models.EquivariantDenoiser(denoiser, transform=None, eval_transform=None, random=True)

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

Turns the input denoiser into an equivariant denoiser with respect to geometric transforms.

Recall that a denoiser is equivariant with respect to a group of transformations if it commutes with the action of
the group. More precisely, let $\mathcal{G}$ be a group of transformations $\{T_g\}_{g\in \mathcal{G}}$
and $\denoisername$ a denoiser. Then, $\denoisername$ is equivariant with respect to $\mathcal{G}$
if $\denoisername(T_g(x)) = T_g(\denoisername(x))$ for any image $x$ and any $g\in \mathcal{G}$.

The denoiser can be turned into an equivariant denoiser by averaging over the group of transforms, i.e.

$$
\operatorname{D}^{\text{eq}}_{\sigma}(x) = \frac{1}{|\mathcal{G}|}\sum_{g\in \mathcal{G}} T_g^{-1}(\operatorname{D}_{\sigma}(T_g(x))).

$$

Otherwise, as proposed by Terris *et al.*<sup>[1](#footcite-terris2024equivariant)</sup>, a Monte Carlo approximation can be obtained by
sampling $g \sim \mathcal{G}$ at random and applying

$$
\operatorname{D}^{\text{MC}}_{\sigma}(x) = T_g^{-1}(\operatorname{D}_{\sigma}(T_g(x))).

$$

#### NOTE
We have implemented many popular geometric transforms, see [docs](https://deepinv.org/user_guide/training/transforms.html.md#transform). You can set the number of Monte Carlo samples by passing `n_trans`
into the transforms, for example `Rotate(n_trans=2)` will average over 2 samples per call. For rotate and reflect, by setting `n_trans`
to the maximum (e.g. 4 for 90 degree rotations, 2 for 1D reflections), it will average over the whole group, for example:

`Rotate(n_trans=4, multiples=90, positive=True) * Reflect(n_trans=2, dims=[-1])`

#### NOTE
It is customary to sample a single transformation at training time and do a full averaging at evaluation time to ensure true equivariance. This can be done by setting a `eval_transform` that averages over the whole group, while leaving `transform` computing a single random transformation.

See [Image transforms for equivariance & augmentations](https://deepinv.org/auto_examples/transforms-equivariance/demo_transforms.html.md#sphx-glr-auto-examples-transforms-equivariance-demo-transforms-py) for an example.

* **Parameters:**
  * **denoiser** (*Callable*) – Denoiser $\operatorname{D}_{\sigma}$.
  * **transform** ([*Transform*](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform)) – geometric transformation. If None, defaults to rotations of multiples of 90 with horizontal flips (see note above).
    See [docs](https://deepinv.org/user_guide/training/transforms.html.md#transform) for list of available transforms.
  * **eval_transform** ([*Transform*](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform)) – transformations to be used in evaluation mode. It can be used to have true Reynolds averaging at evaluation time and efficient Monte Carlo estimation at training time. If set to `None`, evaluation transformations are the same as training transformations.
  * **random** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if True, the denoiser is applied to a randomly transformed version of the input image
    each time i.e. a Monte-Carlo approximation of an equivariant denoiser.
    If False, the denoiser is applied to the average of all the transformed images, turning the denoiser into an
    equivariant denoiser with respect to the chosen group of transformations. Ignored if `transform` is provided.

<hr />

* **References:**

* <a id='footcite-terris2024equivariant'>**[1]**</a> Matthieu Terris, Thomas Moreau, Nelly Pustelnik, and Julián Tachella. Equivariant plug-and-play image reconstruction. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 25255–25264. 2024.

#### forward(x, \*denoiser_args, \*\*denoiser_kwargs)

Symmetrize the denoiser by the transformation to create an equivariant denoiser and apply to input.

The symmetrization collects the average if multiple samples are used (controlled with `n_trans` in the transform).

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image.
  * **\*denoiser_args** – args for denoiser function e.g. sigma noise level.
  * **\*\*denoiser_kwargs** – kwargs for denoiser function e.g. sigma noise level.
* **Returns:**
  denoised image.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-models-equivariantdenoiser"></a>

## Examples using `EquivariantDenoiser`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the use of our deepinv.transform module for use in solving imaging problems. These can be used for:">  <div class="sphx-glr-thumbnail-title">Image transforms for equivariance & augmentations</div>
</div>
<!-- thumbnail-parent-div-close --></div>
