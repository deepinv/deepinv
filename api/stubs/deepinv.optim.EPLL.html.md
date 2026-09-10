# EPLL

### *class* deepinv.optim.EPLL(GMM=None, n_components=200, pretrained='download', patch_size=6, channels=1, device='cpu')

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Expected Patch Log Likelihood reconstruction method.

Reconstruction method based on the minimization problem

$$
\underset{x}{\arg\min} \; \|y-Ax\|^2 - \sum_i \log p(P_ix)
$$

where the first term is a standard $\ell_2$ data-fidelity, and the second term represents a patch prior via
Gaussian mixture models, where $P_i$ is a patch operator that extracts the ith (overlapping) patch from the image.

The reconstruction function is based on the approximated half-quadratic splitting method as in Zoran and Weiss [[172](https://deepinv.org/user_guide/other/biblio.html.md#id41)].

* **Parameters:**
  * **GMM** (*None* *,* [*deepinv.optim.utils.GaussianMixtureModel*](https://deepinv.org/api/stubs/deepinv.optim.utils.GaussianMixtureModel.html.md#deepinv.optim.utils.GaussianMixtureModel)) – Gaussian mixture defining the distribution on the patch space.
    `None` creates a GMM with n_components components of dimension accordingly to the arguments patch_size and channels.
  * **n_components** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of components of the generated GMM if GMM is `None`.
  * **pretrained** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *None*) – Path to pretrained weights of the GMM with file ending `.pt`. None for no pretrained weights,
    `"download"` for pretrained weights on the BSDS500 dataset, `"GMM_lodopab_small"` for the weights from the limited-angle CT example.
    See [pretrained-weights](https://deepinv.org/user_guide/reconstruction/pretrained-models.html.md#pretrained-weights) for more details.
  * **patch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – patch size.
  * **channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of color channels (e.g. 1 for gray-valued images and 3 for RGB images)
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – defines device (`cpu` or `cuda`)

#### forward(y, physics, sigma=None, x_init=None, betas=None, batch_size=-1)

Approximated half-quadratic splitting method for image reconstruction as proposed by Zoran and Weiss.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – tensor of observations. Shape: batch size x …
  * **x_init** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* *None*) – tensor of initializations. If `None` uses initializes with the adjoint of the forward operator.
    Shape: batch size x channels x height x width
  * **physics** ([*deepinv.physics.LinearPhysics*](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)) – Forward linear operator.
  * **betas** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *]*) – parameters from the half-quadratic splitting. `None` uses the standard choice `[1,4,8,16,32]/sigma_sq`
  * **batch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – batching the patch estimations for large images. No effect on the output, but a small value reduces the memory consumption
    but might increase the computation time. -1 for considering all patches at once.

#### negative_log_likelihood(x)

Takes patches and returns the negative log likelihood of the GMM for each patch.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – tensor of patches of shape batch_size x number of patches per batch x patch_dimensions

<a id="sphx-glr-backref-deepinv-optim-epll"></a>

## Examples using `EPLL`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In this example we use the expected patch log likelihood (EPLL) prior :footcitezoran2011learning. for denoising and inpainting of natural images. To this end, we consider the inverse problem y = Ax+\\epsilon, where A is either the identity (for denoising) or a masking operator (for inpainting) and \\epsilon\\sim\\mathcal{N}(0,\\sigma^2 I) is white Gaussian noise with standard deviation \\sigma.">  <div class="sphx-glr-thumbnail-title">Expected Patch Log Likelihood (EPLL) for Denoising and Inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we use patch priors for limited angle computed tomography. More precisely, we consider the inverse problem y = \\mathrm{noisy}(Ax), where A is the discretized Radon transform with 100 equispace angles between 20 and 160 degrees. For the reconstruction, we minimize the variational problem">  <div class="sphx-glr-thumbnail-title">Patch priors for limited-angle computed tomography</div>
</div>
<!-- thumbnail-parent-div-close --></div>
