# PatchPrior

### *class* deepinv.optim.PatchPrior(negative_patch_log_likelihood, n_patches=-1, patch_size=6, pad=False, \*args, \*\*kwargs)

Bases: [`Prior`](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)

Patch prior $g(x) = \sum_i h(P_i x)$ for some prior $h(x)$ on the space of patches.

Given a negative log likelihood (NLL) function on the patch space, this builds a prior by summing
the NLLs of all (overlapping) patches in the image.

* **Parameters:**
  * **negative_patch_log_likelihood** (*Callable*) – NLL function on the patch space
  * **n_patches** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of randomly selected patches for prior evaluation. -1 for taking all patches
  * **patch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – size of the patches
  * **pad** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool) *|* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – whether to use padding on the boundary to avoid undesired boundary effects. If `pad` is a string, it should be a valid padding mode for `torch.nn.functional.pad` (e.g. “reflect”, “constant”, etc.). If `pad` is `True`, the padding mode is set to “reflect”. Default is `False`.

#### fn(x, \*args, \*\*kwargs)

Computes the regularizer

$$
\reg{x} = \sum_i h(P_i x)

$$

for some prior $h(x)$ on the space of patches, where $P_i$ is the operator extracting the $i$-th patch from the image.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the prior is computed.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) prior $g(x)$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-optim-patchprior"></a>

## Examples using `PatchPrior`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In this example we use patch priors for limited angle computed tomography. More precisely, we consider the inverse problem y = \\mathrm{noisy}(Ax), where A is the discretized Radon transform with 100 equispace angles between 20 and 160 degrees. For the reconstruction, we minimize the variational problem">  <div class="sphx-glr-thumbnail-title">Patch priors for limited-angle computed tomography</div>
</div>
<!-- thumbnail-parent-div-close --></div>
