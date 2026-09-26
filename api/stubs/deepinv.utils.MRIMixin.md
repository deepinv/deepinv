# MRIMixin

### *class* deepinv.utils.MRIMixin

Bases: [`object`](https://docs.python.org/3.9/library/functions.html#object)

Mixin base class for MRI functionality.

Base class that provides helper functions for FFT and mask checking.

#### *static* check_mask(mask=None, three_d=False)

Updates MRI mask and verifies mask shape to be B,C,…,H,W where C=2.

* **Parameters:**
  * **mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – MRI subsampling mask.
  * **three_d** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `False` the mask should be min 4 dimensions (B, C, H, W) for 2D data, otherwise if `True` the mask should have 5 dimensions (B, C, D, H, W) for 3D data.

#### crop(x, crop=True, shape=None, rescale=False)

Center crop 2D image according to `img_size`.

This matches the RSS reconstructions of the original raw data in [`deepinv.datasets.FastMRISliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.FastMRISliceDataset.md#deepinv.datasets.FastMRISliceDataset).

If `img_size` has odd height, then adjust by one pixel to match FastMRI data.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor of shape (…,H,W)
  * **crop** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to perform crop, defaults to `True`. If `True`, `rescale` must be `False`.
  * **shape** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – optional shape (…, H,W) to crop to. If `None`, crops to `img_size` attribute.
  * **rescale** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to rescale instead of cropping. If `True`, `crop` must be `False`.
    Note to be careful here as resizing will change aspect ratio.

#### *static* fft(x, dim=(-2, -1), norm='ortho')

Centered, orthogonal fft

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image of complex dtype of shape [B,…] where … is all dims to be transformed
  * **dim** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – fft transform dims, defaults to (-2, -1)
  * **norm** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – fft norm, see docs for [`torch.fft.fftn()`](https://docs.pytorch.org/docs/stable/generated/torch.fft.fftn.html#torch.fft.fftn), defaults to “ortho”

#### *static* from_torch_complex(x)

[B,…,H,W] complex -> [B,2,…,H,W] real

#### *static* ifft(x, dim=(-2, -1), norm='ortho')

Centered, orthogonal ifft

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input kspace of complex dtype of shape [B,…] where … is all dims to be transformed
  * **dim** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – fft transform dims, defaults to (-2, -1)
  * **norm** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – fft norm, see docs for [`torch.fft.fftn()`](https://docs.pytorch.org/docs/stable/generated/torch.fft.fftn.html#torch.fft.fftn), defaults to “ortho”

#### im_to_kspace(x, three_d=False)

Convenience method that wraps fft.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image of shape (B,2,…) of real dtype
  * **three_d** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether MRI data is 3D or not, defaults to False
* **Returns:**
  Tensor: output measurements of shape (B,2,…) of real dtype
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### kspace_to_im(y, three_d=False)

Convenience method that wraps inverse fft.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input measurements of shape (B,2,…) of real dtype
  * **three_d** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether MRI data is 3D or not, defaults to False
* **Returns:**
  Tensor: output image of shape (B,2,…) of real dtype
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### *static* rss(x, multicoil=True, mag=True, three_d=False)

Perform root-sum-square reconstruction on multicoil data, defined as

$$
\operatorname{RSS}(x) = \sqrt{\sum_{n=1}^N |x_n|^2}
$$

where $x_n$ are the coil images of $x$, $|\cdot|$ denotes the magnitude
and $N$ is the number of coils. Note that the sum is performed voxel-wise.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image of shape (B,2,…) where 2 represents
    real and imaginary channels
  * **multicoil** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, assume `x` is of shape (B,2,N,…),
    and reduce over coil dimension N too.
  * **mag** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `False`, do not reduce over the complex dimension. Rarely used.
  * **three_d** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – used only for validating input shape, set to `True` if input is 3D data.

#### *static* to_torch_complex(x)

[B,2,…,H,W] real -> [B,…,H,W] complex

<a id="sphx-glr-backref-deepinv-utils-mrimixin"></a>

## Examples using `MRIMixin`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to get started with DeepInverse in under 5 minutes.">![](auto_examples/basics/images/thumb/sphx_glr_demo_quickstart_thumb.png)

[5 minute quickstart tutorial](https://deepinv.org/auto_examples/basics/demo_quickstart.md)

  <div class="sphx-glr-thumbnail-title">5 minute quickstart tutorial</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various input/output functions provided by DeepInverse for handling medical and scientific imaging formats. We demonstrate loading and plotting from DICOM, NIfTI, ISMRMRD, PyTorch, NumPy and raster data sources.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_io_thumb.png)

[Loading scientific images](https://deepinv.org/auto_examples/external-libraries/demo_io.md)

  <div class="sphx-glr-thumbnail-title">Loading scientific images</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs raw non-Cartesian multicoil kspace data from the FastMRI breast dataset solomonFastMRI2025, for mammography.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_mrinufft_breast_thumb.png)

[Reconstruct accelerated non-Cartesian breast MRI acquisition data](https://deepinv.org/auto_examples/external-libraries/demo_mrinufft_breast.md)

  <div class="sphx-glr-thumbnail-title">Reconstruct accelerated non-Cartesian breast MRI acquisition data</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model terris2025reconstruct to solve inverse problems.">![](auto_examples/models/images/thumb/sphx_glr_demo_foundation_model_thumb.png)

[Inference and fine-tune a foundation model](https://deepinv.org/auto_examples/models/demo_foundation_model.md)

  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo reconstructs undersampled k-space data for 2D cardiac and brain MRI on:">![](auto_examples/models/images/thumb/sphx_glr_demo_mri_pretrained_thumb.png)

[Reconstruct undersampled k-space for cardiac and brain MRI](https://deepinv.org/auto_examples/models/demo_mri_pretrained.md)

  <div class="sphx-glr-thumbnail-title">Reconstruct undersampled k-space for cardiac and brain MRI</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs real prospectively undersampled multicoil brain k-space from yu2022validation.">![](auto_examples/models/images/thumb/sphx_glr_demo_prospective_mri_thumb.png)

[Reconstruct prospectively-undersampled raw multicoil MRI](https://deepinv.org/auto_examples/models/demo_prospective_mri.md)

  <div class="sphx-glr-thumbnail-title">Reconstruct prospectively-undersampled raw multicoil MRI</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">![](auto_examples/physics/images/thumb/sphx_glr_demo_mri_tour_thumb.png)

[Tour of MRI functionality in DeepInverse](https://deepinv.org/auto_examples/physics/demo_mri_tour.md)

  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in here.">![](auto_examples/physics/images/thumb/sphx_glr_demo_physics_tour_thumb.png)

[Tour of forward sensing operators](https://deepinv.org/auto_examples/physics/demo_physics_tour.md)

  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the self-supervised Artifact2Artifact loss for solving an undersampled sequential MRI reconstruction problem without ground truth.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_artifact2artifact_thumb.png)

[Self-supervised MRI reconstruction with Artifact2Artifact](https://deepinv.org/auto_examples/self-supervised-learning/demo_artifact2artifact.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised MRI reconstruction with Artifact2Artifact</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a reconstruction network for an MRI inverse problem on a fully self-supervised way, i.e., using measurement data only.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_equivariant_imaging_thumb.png)

[Self-supervised learning with Equivariant Imaging for MRI.](https://deepinv.org/auto_examples/self-supervised-learning/demo_equivariant_imaging.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Imaging for MRI.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised (blind) denoising of a low-field MRI scan without ground truth data.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_lowfieldmri_thumb.png)

[Low-field MRI denoising without ground truth](https://deepinv.org/auto_examples/self-supervised-learning/demo_lowfieldmri.md)

  <div class="sphx-glr-thumbnail-title">Low-field MRI denoising without ground truth</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate scan-specific self-supervised learning, that is, learning to reconstruct MRI scans from a single accelerated sample without ground truth.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_scan_specific_thumb.png)

[Scan-specific zero-shot SSDU for MRI](https://deepinv.org/auto_examples/self-supervised-learning/demo_scan_specific.md)

  <div class="sphx-glr-thumbnail-title">Scan-specific zero-shot SSDU for MRI</div>
</div>
<!-- thumbnail-parent-div-close --></div>
