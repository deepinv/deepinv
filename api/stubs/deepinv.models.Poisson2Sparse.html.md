# Poisson2Sparse

### *class* deepinv.models.Poisson2Sparse(backbone=None, , lr=1e-4, weight_n2n=2.0, weight_l1_regularization=1e-5, num_iter=200, verbose=False)

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

Poisson2Sparse model for Poisson denoising.

This method, introduced by Ta *et al.*<sup>[1](#footcite-ta2022poisson2sparse)</sup>, reconstructs an image corrupted by Poisson noise by learning a sparse non-linear dictionary parametrized by a neural network using a combination of Neighbor2Neighbor Huang *et al.*<sup>[2](#footcite-huang2021neighbor2neighbor)</sup>, of the negative log Poisson likelihood, of the $\ell^1$ pixel distance and of a sparsity-inducing $\ell^1$ regularization function on the weights.

#### NOTE
This method does not use knowledge of the physics model and assumes a Poisson degradation model internally. Therefore, the physics object can be omitted when calling the model and specifying it will have no effect.

#### NOTE
The denoiser expects images with a dynamic range normalized between zero and one.

* **Parameters:**
  * **backbone** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) *,* *None*) – Neural network used as a non-linear dictionary. If `None`, a default [`deepinv.models.ConvLista`](https://deepinv.org/api/stubs/deepinv.models.ConvLista.html.md#deepinv.models.ConvLista) model is used.
  * **lr** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Learning rate of the AdamW optimizer (default: 1e-4).
  * **weight_n2n** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Weight of the Neighbor2Neighbor loss term (default: 2.0).
  * **weight_l1_regularization** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Weight of the sparsity-inducing $\ell^1$ regularization on the weights (default: 1e-5).
  * **num_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of optimization iterations (default: 200).
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, print progress (default: `False`).

<hr />

* **References:**

* <a id='footcite-ta2022poisson2sparse'>**[1]**</a> Calvin-Khang Ta, Abhishek Aich, Akash Gupta, and Amit K Roy-Chowdhury. Poisson2sparse: self-supervised poisson denoising from a single image. In *International Conference on Medical Image Computing and Computer-Assisted Intervention*, 557–567. Springer, 2022.
* <a id='footcite-huang2021neighbor2neighbor'>**[2]**</a> Tao Huang, Songjiang Li, Xu Jia, Huchuan Lu, and Jianzhuang Liu. Neighbor2neighbor: self-supervised denoising from single noisy images. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 14781–14790. 2021.

<a id="sphx-glr-backref-deepinv-models-poisson2sparse"></a>

## Examples using `Poisson2Sparse`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to restore a single image corrupted by Poisson noise using Poisson2Sparse, without requiring external training or knowledge of the noise level.">  <div class="sphx-glr-thumbnail-title">Poisson denoising using Poisson2Sparse</div>
</div>
<!-- thumbnail-parent-div-close --></div>
