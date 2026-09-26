# load_torch_url

### deepinv.utils.load_torch_url(url, device='cpu', \*\*kwargs)

Load an array from url and read it by torch.load.

* **Parameters:**
  * **url** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – URL of the image file.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – Device on which to load the tensor.
* **Returns:**
  weights or tensors contained in file.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

## Examples using `load_torch_url`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the Astra tomography toolbox with deepinv, a popular toolbox for tomography with GPU acceleration.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_astra_tomography_thumb.png)

[Low-dose CT with ASTRA backend and Total-Variation (TV) prior](https://deepinv.org/auto_examples/external-libraries/demo_astra_tomography.html.md)

  <div class="sphx-glr-thumbnail-title">Low-dose CT with ASTRA backend and Total-Variation (TV) prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we use patch priors for limited angle computed tomography. More precisely, we consider the inverse problem y = \\mathrm{noisy}(Ax), where A is the discretized Radon transform with 100 equispace angles between 20 and 160 degrees. For the reconstruction, we minimize the variational problem">![](auto_examples/optimization/images/thumb/sphx_glr_demo_patch_priors_CT_thumb.png)

[Patch priors for limited-angle computed tomography](https://deepinv.org/auto_examples/optimization/demo_patch_priors_CT.html.md)

  <div class="sphx-glr-thumbnail-title">Patch priors for limited-angle computed tomography</div>
</div>
<!-- thumbnail-parent-div-close --></div>
