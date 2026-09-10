# DeepImagePrior

### *class* deepinv.models.DeepImagePrior(generator, img_size, iterations=2500, learning_rate=1e-2, verbose=False, re_init=False)

Bases: [`Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor)

Deep Image Prior reconstruction.

This method, introduced by Ulyanov *et al.*<sup>[1](#footcite-ulyanov2018deep)</sup>, reconstructs an image by minimizing the loss function

$$
\min_{\theta}  \|y-AG_{\theta}(z)\|^2
$$

where $z$ is a random input and $G_{\theta}$ is a convolutional decoder network with parameters
$\theta$. The minimization should be stopped early to avoid overfitting. The method uses the Adam
optimizer.

#### NOTE
This method only works with certain convolutional decoder networks. We recommend using the
network [`deepinv.models.ConvDecoder`](https://deepinv.org/api/stubs/deepinv.models.ConvDecoder.html.md#deepinv.models.ConvDecoder).

#### NOTE
The number of iterations and learning rate are set to the values used in the original paper. However, these
values may not be optimal for all problems. We recommend experimenting with different values.

* **Parameters:**
  * **generator** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Convolutional decoder network.
  * **img_size** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – Size `(C,H,W)` of the input noise vector $z$.
  * **iterations** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of optimization iterations.
  * **learning_rate** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Learning rate of the Adam optimizer.
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, print progress.
  * **re_init** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, re-initialize the network parameters before each reconstruction.

<hr />

* **References:**

* <a id='footcite-ulyanov2018deep'>**[1]**</a> Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. Deep image prior. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, 9446–9454. 2018.

#### forward(y, physics, \*\*kwargs)

Reconstruct an image from the measurement $y$. The reconstruction is performed by solving a minimization
problem.

#### WARNING
The optimization is run for every test batch. Thus, this method can be slow when tested on a large
number of test batches.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurement.
  * **physics** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Physics model.

<a id="sphx-glr-backref-deepinv-models-deepimageprior"></a>

## Examples using `DeepImagePrior`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to reconstruct a noisy and incomplete image using the deep image prior.">  <div class="sphx-glr-thumbnail-title">Reconstructing an image using the deep image prior.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
