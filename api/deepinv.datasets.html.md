# deepinv.datasets

This module can be used for defining datasets or generating reconstruction datasets from other base datasets.
Please refer to the [user guide](https://deepinv.org/user_guide/training/datasets.html.md#datasets) for more information.

## Base Datasets

**User Guide:** refer to [Base Datasets](https://deepinv.org/user_guide/training/datasets.html.md#base-datasets) for more information.

| [`deepinv.datasets.ImageDataset`](https://deepinv.org/api/stubs/deepinv.datasets.ImageDataset.html.md#deepinv.datasets.ImageDataset)   | Base class for imaging datasets in DeepInverse.     |
|----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| [`deepinv.datasets.ImageFolder`](https://deepinv.org/api/stubs/deepinv.datasets.ImageFolder.html.md#deepinv.datasets.ImageFolder)     | Dataset loading images from files.                  |
| [`deepinv.datasets.TensorDataset`](https://deepinv.org/api/stubs/deepinv.datasets.TensorDataset.html.md#deepinv.datasets.TensorDataset) | Dataset wrapping data explicitly passed as tensors. |

| [`deepinv.datasets.check_dataset`](https://deepinv.org/api/stubs/deepinv.datasets.check_dataset.html.md#deepinv.datasets.check_dataset)   | Check that a torch dataset is compatible with DeepInverse.   |
|------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|

## Generating Datasets

**User Guide:** refer to [Generating Datasets](https://deepinv.org/user_guide/training/datasets.html.md#generating-datasets) for more information.

| [`deepinv.datasets.HDF5Dataset`](https://deepinv.org/api/stubs/deepinv.datasets.HDF5Dataset.html.md#deepinv.datasets.HDF5Dataset)   | DeepInverse HDF5 dataset   |
|--------------------------------------------------------------------------------------------------------------|----------------------------|

| [`deepinv.datasets.generate_dataset`](https://deepinv.org/api/stubs/deepinv.datasets.generate_dataset.html.md#deepinv.datasets.generate_dataset)   | Generates dataset of signal/measurement pairs from base dataset.   |
|------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|

## Image Datasets

**User Guide:** refer to [Predefined Datasets](https://deepinv.org/user_guide/training/datasets.html.md#predefined-datasets) for more information.

| [`deepinv.datasets.DIV2K`](https://deepinv.org/api/stubs/deepinv.datasets.DIV2K.html.md#deepinv.datasets.DIV2K)                                         | Dataset for [DIV2K Image Super-Resolution Challenge](https://data.vision.ee.ethz.ch/cvl/DIV2K).                              |
|----------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| [`deepinv.datasets.Urban100HR`](https://deepinv.org/api/stubs/deepinv.datasets.Urban100HR.html.md#deepinv.datasets.Urban100HR)                               | Dataset for [Urban100](https://paperswithcode.com/dataset/urban100).                                                         |
| [`deepinv.datasets.Set14HR`](https://deepinv.org/api/stubs/deepinv.datasets.Set14HR.html.md#deepinv.datasets.Set14HR)                                     | Dataset for [Set14](https://paperswithcode.com/dataset/set14).                                                               |
| [`deepinv.datasets.Set5HR`](https://deepinv.org/api/stubs/deepinv.datasets.Set5HR.html.md#deepinv.datasets.Set5HR)                                       | Dataset for [Set5](https://paperswithcode.com/dataset/set5).                                                                 |
| [`deepinv.datasets.BSDS500`](https://deepinv.org/api/stubs/deepinv.datasets.BSDS500.html.md#deepinv.datasets.BSDS500)                                     | Dataset for [BSDS500](https://github.com/BIDS/BSDS500).                                                                      |
| [`deepinv.datasets.BSD100HR`](https://deepinv.org/api/stubs/deepinv.datasets.BSD100HR.html.md#deepinv.datasets.BSD100HR)                                   | Dataset for [BSD100](https://paperswithcode.com/dataset/bsd100).                                                             |
| [`deepinv.datasets.McMaster`](https://deepinv.org/api/stubs/deepinv.datasets.McMaster.html.md#deepinv.datasets.McMaster)                                   | Dataset for [McMaster](https://www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.htm).                                            |
| [`deepinv.datasets.Kodak24`](https://deepinv.org/api/stubs/deepinv.datasets.Kodak24.html.md#deepinv.datasets.Kodak24)                                     | Dataset for [Kodak24](http://r0k.us/graphics/kodak/).                                                                        |
| [`deepinv.datasets.CBSD68`](https://deepinv.org/api/stubs/deepinv.datasets.CBSD68.html.md#deepinv.datasets.CBSD68)                                       | Dataset for [CBSD68](https://paperswithcode.com/dataset/cbsd68).                                                             |
| [`deepinv.datasets.FastMRISliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.FastMRISliceDataset.html.md#deepinv.datasets.FastMRISliceDataset)             | Dataset for [fastMRI](https://fastmri.med.nyu.edu/) that provides access to raw MR kspace data.                              |
| [`deepinv.datasets.SimpleFastMRISliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.SimpleFastMRISliceDataset.html.md#deepinv.datasets.SimpleFastMRISliceDataset) | Simple FastMRI image dataset.                                                                                                |
| [`deepinv.datasets.CMRxReconSliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.CMRxReconSliceDataset.html.md#deepinv.datasets.CMRxReconSliceDataset)         | CMRxRecon dynamic MRI dataset.                                                                                               |
| [`deepinv.datasets.SKMTEASliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.SKMTEASliceDataset.html.md#deepinv.datasets.SKMTEASliceDataset)               | SKM-TEA dataset for raw multicoil MRI kspace data.                                                                           |
| [`deepinv.datasets.LidcIdriSliceDataset`](https://deepinv.org/api/stubs/deepinv.datasets.LidcIdriSliceDataset.html.md#deepinv.datasets.LidcIdriSliceDataset)           | Dataset for [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) that provides access to CT image slices. |
| [`deepinv.datasets.Flickr2kHR`](https://deepinv.org/api/stubs/deepinv.datasets.Flickr2kHR.html.md#deepinv.datasets.Flickr2kHR)                               | Dataset for [Flickr2K](https://github.com/limbee/NTIRE2017).                                                                 |
| [`deepinv.datasets.LsdirHR`](https://deepinv.org/api/stubs/deepinv.datasets.LsdirHR.html.md#deepinv.datasets.LsdirHR)                                     | Dataset for [LSDIR](https://ofsoundof.github.io/lsdir-data/).                                                                |
| [`deepinv.datasets.FMD`](https://deepinv.org/api/stubs/deepinv.datasets.FMD.html.md#deepinv.datasets.FMD)                                             | Dataset for [Fluorescence Microscopy Denoising](https://github.com/yinhaoz/denoising-fluorescence).                          |
| [`deepinv.datasets.Kohler`](https://deepinv.org/api/stubs/deepinv.datasets.Kohler.html.md#deepinv.datasets.Kohler)                                       | Dataset for [Recording and Playback of Camera Shake](https://doi.org/10.1007/978-3-642-33786-4_3)                            |
| [`deepinv.datasets.NBUDataset`](https://deepinv.org/api/stubs/deepinv.datasets.NBUDataset.html.md#deepinv.datasets.NBUDataset)                               | NBU remote sensing multispectral satellite imagery dataset.                                                                  |
| [`deepinv.datasets.BrainWebPET`](https://deepinv.org/api/stubs/deepinv.datasets.BrainWebPET.html.md#deepinv.datasets.BrainWebPET)                             | BrainWeb PET phantoms.                                                                                                       |
| [`deepinv.datasets.BrainWebMRI`](https://deepinv.org/api/stubs/deepinv.datasets.BrainWebMRI.html.md#deepinv.datasets.BrainWebMRI)                             | Dataset for [BrainWeb](https://brainweb.bic.mni.mcgill.ca/).                                                                 |

## Other Datasets

| [`deepinv.datasets.PatchDataset`](https://deepinv.org/api/stubs/deepinv.datasets.PatchDataset.html.md#deepinv.datasets.PatchDataset)                         | Builds the dataset of all patches from a tensor of images.     |
|--------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| [`deepinv.datasets.RandomPatchSampler`](https://deepinv.org/api/stubs/deepinv.datasets.RandomPatchSampler.html.md#deepinv.datasets.RandomPatchSampler)             | Dataset for nD images that samples one random patch per image. |
| [`deepinv.datasets.utils.PlaceholderDataset`](https://deepinv.org/api/stubs/deepinv.datasets.utils.PlaceholderDataset.html.md#deepinv.datasets.utils.PlaceholderDataset) | A placeholder dataset for test purposes.                       |

## Data Transforms

**User Guide:** refer to [Data Transforms](https://deepinv.org/user_guide/training/datasets.html.md#data-transforms) for more information.

| [`deepinv.datasets.utils.Rescale`](https://deepinv.org/api/stubs/deepinv.datasets.utils.Rescale.html.md#deepinv.datasets.utils.Rescale)         | Image value rescale torchvision-style transform.                           |
|------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| [`deepinv.datasets.utils.ToComplex`](https://deepinv.org/api/stubs/deepinv.datasets.utils.ToComplex.html.md#deepinv.datasets.utils.ToComplex)     | Torchvision-style transform to add empty imaginary dimension to image.     |
| [`deepinv.datasets.utils.Crop`](https://deepinv.org/api/stubs/deepinv.datasets.utils.Crop.html.md#deepinv.datasets.utils.Crop)               | Torchvision-style transform to take crop in corner or any arbitrary place. |
| [`deepinv.datasets.MRISliceTransform`](https://deepinv.org/api/stubs/deepinv.datasets.MRISliceTransform.html.md#deepinv.datasets.MRISliceTransform) | FastMRI raw data preprocessing.                                            |
