# download_example

### deepinv.utils.download_example(name, save_dir)

Download an image from the [DeepInverse HuggingFace](https://huggingface.co/datasets/deepinv/images) to file.

For all available examples, see [`deepinv.utils.load_example()`](https://deepinv.org/api/stubs/deepinv.utils.load_example.md#deepinv.utils.load_example).

* **Parameters:**
  * **name** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – filename of the image from the HuggingFace dataset.
  * **save_dir** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – directory to save image to.

## Examples using `download_example`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use DeepInverse with your own dataset.">![](auto_examples/basics/images/thumb/sphx_glr_demo_custom_dataset_thumb.png)

[Bring your own dataset](https://deepinv.org/auto_examples/basics/demo_custom_dataset.md)

  <div class="sphx-glr-thumbnail-title">Bring your own dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs raw non-Cartesian multicoil kspace data from the FastMRI breast dataset solomonFastMRI2025, for mammography.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_mrinufft_breast_thumb.png)

[Reconstruct accelerated non-Cartesian breast MRI acquisition data](https://deepinv.org/auto_examples/external-libraries/demo_mrinufft_breast.md)

  <div class="sphx-glr-thumbnail-title">Reconstruct accelerated non-Cartesian breast MRI acquisition data</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised (blind) denoising of a low-field MRI scan without ground truth data.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_lowfieldmri_thumb.png)

[Low-field MRI denoising without ground truth](https://deepinv.org/auto_examples/self-supervised-learning/demo_lowfieldmri.md)

  <div class="sphx-glr-thumbnail-title">Low-field MRI denoising without ground truth</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate scan-specific self-supervised learning, that is, learning to reconstruct MRI scans from a single accelerated sample without ground truth.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_scan_specific_thumb.png)

[Scan-specific zero-shot SSDU for MRI](https://deepinv.org/auto_examples/self-supervised-learning/demo_scan_specific.md)

  <div class="sphx-glr-thumbnail-title">Scan-specific zero-shot SSDU for MRI</div>
</div>
<!-- thumbnail-parent-div-close --></div>
