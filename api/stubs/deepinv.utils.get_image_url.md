# get_image_url

### deepinv.utils.get_image_url(file_name, dataset='images')

Get URL for image from DeepInverse HuggingFace repository.

* **Parameters:**
  * **file_name** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – image filename in repository
  * **dataset** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – HuggingFace dataset name, defaults to ‘images’
* **Return str:**
  image URL
* **Return type:**
  [str](https://docs.python.org/3.9/library/stdtypes.html#str)

## Examples using `get_image_url`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.physics.Physics together with automatic differentiation to estimate unknown parameters of your forward operator.">![](auto_examples/blind-inverse-problems/images/thumb/sphx_glr_demo_optimizing_physics_parameter_thumb.png)

[Calibrating physics operators](https://deepinv.org/auto_examples/blind-inverse-problems/demo_optimizing_physics_parameter.md)

  <div class="sphx-glr-thumbnail-title">Calibrating physics operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="The data is taken from the 2DeteCT benchmark kiss2025benchmarking and dataset kiss20232detect, which is an industrial CT dataset of various materials acquired using a proprietary scanner from CWI (i.e. sinogram-to-image). The setup is matched exactly to kiss2025benchmarking, such that you can compare DeepInverse image reconstruction methods with the values reported in kiss2025benchmarking.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_astra_2detect_thumb.png)

[Reconstruct real CT sinograms with the 2DeteCT benchmark](https://deepinv.org/auto_examples/external-libraries/demo_astra_2detect.md)

  <div class="sphx-glr-thumbnail-title">Reconstruct real CT sinograms with the 2DeteCT benchmark</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various input/output functions provided by DeepInverse for handling medical and scientific imaging formats. We demonstrate loading and plotting from DICOM, NIfTI, ISMRMRD, PyTorch, NumPy and raster data sources.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_io_thumb.png)

[Loading scientific images](https://deepinv.org/auto_examples/external-libraries/demo_io.md)

  <div class="sphx-glr-thumbnail-title">Loading scientific images</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example, we investigate a simple 2D Radio Interferometry (RI) imaging task with deepinverse. The following example and data are taken from aghabiglou2024r2d2. If you are interested in RI imaging problem and would like to see more examples or try the state-of-the-art algorithms, please check BASPLib.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_ri_basic_thumb.png)

[Radio interferometric imaging with deepinverse](https://deepinv.org/auto_examples/external-libraries/demo_ri_basic.md)

  <div class="sphx-glr-thumbnail-title">Radio interferometric imaging with deepinverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo reconstructs undersampled k-space data for 2D cardiac and brain MRI on:">![](auto_examples/models/images/thumb/sphx_glr_demo_mri_pretrained_thumb.png)

[Reconstruct undersampled k-space for cardiac and brain MRI](https://deepinv.org/auto_examples/models/demo_mri_pretrained.md)

  <div class="sphx-glr-thumbnail-title">Reconstruct undersampled k-space for cardiac and brain MRI</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs real prospectively undersampled multicoil brain k-space from yu2022validation.">![](auto_examples/models/images/thumb/sphx_glr_demo_prospective_mri_thumb.png)

[Reconstruct prospectively-undersampled raw multicoil MRI](https://deepinv.org/auto_examples/models/demo_prospective_mri.md)

  <div class="sphx-glr-thumbnail-title">Reconstruct prospectively-undersampled raw multicoil MRI</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a full-resolution multispectral image from raw snapshot mosaiced measurements using various reconstruction algorithms.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_multispectral_demosaicing_thumb.png)

[Multispectral demosaicing from raw sensor data](https://deepinv.org/auto_examples/optimization/demo_multispectral_demosaicing.md)

  <div class="sphx-glr-thumbnail-title">Multispectral demosaicing from raw sensor data</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">![](auto_examples/physics/images/thumb/sphx_glr_demo_mri_tour_thumb.png)

[Tour of MRI functionality in DeepInverse](https://deepinv.org/auto_examples/physics/demo_mri_tour.md)

  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo illustrates the impact of different Hadamard pattern ordering algorithms in the Single Pixel Camera (SPC), a computational imaging system that uses a single photodetector to capture images by projecting the scene onto a series of patterns. The SPC is implemented in the DeepInverse library.">![](auto_examples/physics/images/thumb/sphx_glr_demo_spc_thumb.png)

[Pattern Ordering in a Compressive Single Pixel Camera](https://deepinv.org/auto_examples/physics/demo_spc.md)

  <div class="sphx-glr-thumbnail-title">Pattern Ordering in a Compressive Single Pixel Camera</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use deepinv.physics.UltrasoundPlaneWave to reconstructs an in-vivo carotid acquisition of the EPFL LTS5 ultrafast ultrasound dataset from raw RF ultrasound data.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_ultrasound_invivo_PnP_thumb.png)

[In-vivo ultrafast ultrasound reconstruction with Plug-and-Play](https://deepinv.org/auto_examples/plug-and-play/demo_ultrasound_invivo_PnP.md)

  <div class="sphx-glr-thumbnail-title">In-vivo ultrafast ultrasound reconstruction with Plug-and-Play</div>
</div>
<!-- thumbnail-parent-div-close --></div>
