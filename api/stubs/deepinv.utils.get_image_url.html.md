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
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.physics.Physics together with automatic differentiation to estimate unknown parameters of your forward operator.">  <div class="sphx-glr-thumbnail-title">Calibrating physics operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="The data is taken from the 2DeteCT benchmark :footcitekiss2025benchmarking and dataset :footcitekiss20232detect, which is an industrial CT dataset of various materials acquired using a proprietary scanner from CWI (i.e. sinogram-to-image). The setup is matched exactly to :footcitekiss2025benchmarking, such that you can compare DeepInverse image reconstruction methods with the values reported in :footcitekiss2025benchmarking.">  <div class="sphx-glr-thumbnail-title">Reconstruct real CT sinograms with the 2DeteCT benchmark</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various input/output functions provided by DeepInverse for handling medical and scientific imaging formats. We demonstrate loading and plotting from DICOM, NIfTI, ISMRMRD, PyTorch, NumPy and raster data sources.">  <div class="sphx-glr-thumbnail-title">Loading scientific images</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example, we investigate a simple 2D Radio Interferometry (RI) imaging task with deepinverse. The following example and data are taken from :footciteaghabiglou2024r2d2. If you are interested in RI imaging problem and would like to see more examples or try the state-of-the-art algorithms, please check BASPLib.">  <div class="sphx-glr-thumbnail-title">Radio interferometric imaging with deepinverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo reconstructs undersampled k-space data for 2D cardiac and brain MRI on:">  <div class="sphx-glr-thumbnail-title">Reconstruct undersampled k-space for cardiac and brain MRI</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs real prospectively undersampled multicoil brain k-space from :footciteyu2022validation.">  <div class="sphx-glr-thumbnail-title">Reconstruct prospectively-undersampled raw multicoil MRI</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a full-resolution multispectral image from raw snapshot mosaiced measurements using various reconstruction algorithms.">  <div class="sphx-glr-thumbnail-title">Multispectral demosaicing from raw sensor data</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo illustrates the impact of different Hadamard pattern ordering algorithms in the Single Pixel Camera (SPC), a computational imaging system that uses a single photodetector to capture images by projecting the scene onto a series of patterns. The SPC is implemented in the DeepInverse library.">  <div class="sphx-glr-thumbnail-title">Pattern Ordering in a Compressive Single Pixel Camera</div>
</div>
<!-- thumbnail-parent-div-close --></div>
