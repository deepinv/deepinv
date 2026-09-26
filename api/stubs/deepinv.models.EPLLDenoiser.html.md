# EPLLDenoiser

### *class* deepinv.models.EPLLDenoiser(GMM=None, n_components=200, pretrained='download', patch_size=6, channels=1, device=torch.device('cpu'))

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

Expected Patch Log Likelihood denoising method.

This class implements the Expected Patch Log Likelihood (EPLL) denoising method from Zoran and Weiss<sup>[1](#footcite-zoran2011learning)</sup>, which is a denoising method based on the minimization problem

$$
\underset{x}{\arg\min} \, \|y-x\|^2 - \sum_i \log p(P_ix)
$$

where the first term is a standard L2 data-fidelity, and the second term represents a patch prior via
Gaussian mixture models, where $P_i$ is a patch operator that extracts the ith (overlapping) patch from the image.

* **Parameters:**
  * **GMM** (*None* *,* [*deepinv.optim.utils.GaussianMixtureModel*](https://deepinv.org/api/stubs/deepinv.optim.utils.GaussianMixtureModel.html.md#deepinv.optim.utils.GaussianMixtureModel)) – Gaussian mixture defining the distribution on the patch space.
    `None` creates a GMM with n_components components of dimension accordingly to the arguments patch_size and channels.
  * **n_components** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of components of the generated GMM if GMM is `None`.
  * **pretrained** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *None*) – Path to pretrained weights of the GMM with file ending `.pt`. None for no pretrained weights,
    `"download"` for pretrained weights on the BSDS500 dataset, `"GMM_lodopab_small"` for the weights from the limited-angle CT example.
    See [pretrained-weights](https://deepinv.org/user_guide/reconstruction/pretrained-models.html.md#pretrained-weights) for more details.
  * **patch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – patch size.
  * **channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of color channels (e.g. 1 for gray-valued images and 3 for RGB images)
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – defines device (`cpu` or `cuda`)

<hr />

* **References:**

* <a id='footcite-zoran2011learning'>**[1]**</a> Daniel Zoran and Yair Weiss. From learning models of natural image patches to whole image restoration. In *2011 international conference on computer vision*, 479–486. IEEE, 2011.

#### forward(x, sigma, betas=None, batch_size=-1)

Denoising method based on the minimization problem.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – noisy image. Shape: batch size x …
  * **physics** ([*deepinv.physics.LinearPhysics*](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)) – Forward linear operator.
  * **betas** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *]*) – parameters from the half-quadratic splitting. `None` uses
    the standard choice `[1,4,8,16,32]/sigma_sq`
  * **batch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – batching the patch estimations for large images. No effect on the output,
    but a small value reduces the memory consumption
    and might increase the computation time. `-1` for considering all patches at once.
