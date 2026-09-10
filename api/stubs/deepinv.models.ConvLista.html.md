# ConvLista

### *class* deepinv.models.ConvLista(, in_channels, out_channels, kernel_size=3, num_filters=512, stride=2, num_iter=10, threshold=1e-2)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Convolutional LISTA network.

The architecture was introduced by Simon and Elad<sup>[1](#footcite-simon2019rethinking)</sup>, and it is well suited as a backbone for Poisson2Sparse (see [`deepinv.models.Poisson2Sparse`](https://deepinv.org/api/stubs/deepinv.models.Poisson2Sparse.html.md#deepinv.models.Poisson2Sparse)).

#### NOTE
The decoder expects images with a dynamic range normalized between zero and one.

* **Parameters:**
  * **in_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of channels in the input image.
  * **out_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of channels in the output image.
  * **kernel_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Size of the convolutional kernels (default: 3).
  * **num_filters** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of filters in the convolutional layers (default: 512).
  * **stride** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Stride of the convolutional layers (default: 2).
  * **num_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of iterations of the LISTA algorithm (default: 10).
  * **threshold** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Initial value for the learned soft-thresholding (default: 1e-2).

<hr />

* **References:**

* <a id='footcite-simon2019rethinking'>**[1]**</a> Dror Simon and Michael Elad. Rethinking the csc model for natural images. *Advances in Neural Information Processing Systems*, 2019.

<a id="sphx-glr-backref-deepinv-models-convlista"></a>

## Examples using `ConvLista`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to restore a single image corrupted by Poisson noise using Poisson2Sparse, without requiring external training or knowledge of the noise level.">  <div class="sphx-glr-thumbnail-title">Poisson denoising using Poisson2Sparse</div>
</div>
<!-- thumbnail-parent-div-close --></div>
