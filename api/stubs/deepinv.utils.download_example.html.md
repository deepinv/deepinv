# download_example

### deepinv.utils.download_example(name, save_dir)

Download an image from the [DeepInverse HuggingFace](https://huggingface.co/datasets/deepinv/images) to file.

For all available examples, see [`deepinv.utils.load_example()`](https://deepinv.org/api/stubs/deepinv.utils.load_example.html.md#deepinv.utils.load_example).

* **Parameters:**
  * **name** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – filename of the image from the HuggingFace dataset.
  * **save_dir** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – directory to save image to.

## Examples using `download_example`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use DeepInverse with your own dataset.">  <div class="sphx-glr-thumbnail-title">Bring your own dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised (blind) denoising of a low-field MRI scan without ground truth data.">  <div class="sphx-glr-thumbnail-title">Low-field MRI denoising without ground truth</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate scan-specific self-supervised learning, that is, learning to reconstruct MRI scans from a single accelerated sample without ground truth.">  <div class="sphx-glr-thumbnail-title">Scan-specific zero-shot SSDU for MRI</div>
</div>
<!-- thumbnail-parent-div-close --></div>
