# BM3D

### *class* deepinv.models.BM3D(use_legacy=True, device='cpu', \*\*kwargs)

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.md#deepinv.models.Denoiser)

BM3D denoiser.

The BM3D denoiser was introduced by Dabov *et al.*<sup>[1](#footcite-dabov2007image)</sup>.

* **Parameters:**
  * **use_legacy** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Whether to use the legacy implementation of BM3D. Default: `True`
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *|* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device to run the fast implementation of BM3D on. Default: `"cpu"`
  * **kwargs** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – additional keyword arguments for the fast implementation of BM3D. See the note below for details.

#### NOTE
Unlike other denoisers from the library, this denoiser is applied sequentially to each noisy image in the batch
(no parallelization). Furthermore, it does not support backpropagation.

#### NOTE
Additional keyword arguments are supported for the fast implementation of BM3D (when `use_legacy=False`), which include:

- `patch_size`: size of each image patch. Default: 8
- `search_radius`: search window radius for block matching. Default: 19
- `search_step`: step size for block matching. Default: 1
- `ref_stride`: stride for selecting reference patches. Default: 3
- `chunk_size`: number of groups to process in parallel. Default: 2048
- `ht_group_size`: group size for stage 1 (hard-thresholding). Default: 16
- `wiener_group_size`: group size for stage 2 (Wiener filtering). Default: 32
- `spatial_ht_transform`: spatial transform for stage 1. Default: `"bior1.5"`
- `spatial_wiener_transform`: spatial transform for stage 2. Default: `"dct"`
- `group_ht_transform`: group transform for stage 1. Default: `"haar"`
- `group_wiener_transform`: group transform for stage 2. Default: `"haar"`
- `hard_threshold`: hard-thresholding parameter for stage 1. Default: 3.0
- `wiener_mu2`: Wiener filtering parameter for stage 2. Default: 0.4

#### WARNING
When `use_legacy=True`, the denoiser calls the BM3D denoiser from the [BM3D python package](https://pypi.org/project/bm3d/).
It can be installed with `pip install bm3d`.
This implementation always runs on the CPU regardless of the device of the input tensor.

When `use_legacy=False`, the denoiser calls a custom re-implementation of BM3D.
It requires `ptwt`, which can be installed with `pip install ptwt`.
It runs on the device specified by the `device` parameter, and is significantly faster than the legacy implementation, especially when the input tensor is on the GPU.
However, it may produce slightly different results than the legacy implementation.

<hr />

* **References:**

* <a id='footcite-dabov2007image'>**[1]**</a> Kostadin Dabov, Alessandro Foi, Vladimir Katkovnik, and Karen Egiazarian. Image denoising by sparse 3-d transform-domain collaborative filtering. *IEEE Transactions on image processing*, 16(8):2080–2095, 2007.

<a id="sphx-glr-backref-deepinv-models-bm3d"></a>

## Examples using `BM3D`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of the denoisers in DeepInverse. A denoiser is a model that takes in a noisy image and outputs a denoised version of it. Basically, it solves the following problem:">![](auto_examples/models/images/thumb/sphx_glr_demo_denoiser_tour_thumb.png)

[Benchmarking pretrained denoisers](https://deepinv.org/auto_examples/models/demo_denoiser_tour.md)

  <div class="sphx-glr-thumbnail-title">Benchmarking pretrained denoisers</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use deepinv.physics.UltrasoundPlaneWave to reconstructs an in-vivo carotid acquisition of the EPFL LTS5 ultrafast ultrasound dataset from raw RF ultrasound data.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_ultrasound_invivo_PnP_thumb.png)

[In-vivo ultrafast ultrasound reconstruction with Plug-and-Play](https://deepinv.org/auto_examples/plug-and-play/demo_ultrasound_invivo_PnP.md)

  <div class="sphx-glr-thumbnail-title">In-vivo ultrafast ultrasound reconstruction with Plug-and-Play</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example despeckles real clinical ultrasound B-mode images with Speckle2Self li2025speckle2self, a model pretrained without clean reference images.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_ultrasound_despeckling_thumb.png)

[Ultrasound despeckling from B-mode images](https://deepinv.org/auto_examples/self-supervised-learning/demo_ultrasound_despeckling.md)

  <div class="sphx-glr-thumbnail-title">Ultrasound despeckling from B-mode images</div>
</div>
<!-- thumbnail-parent-div-close --></div>
