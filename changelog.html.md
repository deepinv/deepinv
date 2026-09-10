# Change Log

This change log is for the `main` branch. It contains changes for each release, with the date and author of each change.

## Current

### New Features

- Publish the docs in `llms.txt` format using the [sphinx-llm](https://github.com/NVIDIA/sphinx-llm) extension ([#1362](https://github.com/deepinv/deepinv/pull/1362) by [Julian Tachella](https://github.com/tachella))
- Add distributed backward propagation and training for samples too large to fit on a single device ([#1088](https://github.com/deepinv/deepinv/pull/1088) by [Benoît Malézieux](https://github.com/bmalezieux))
- Add [`deepinv.physics.TomographyWithAstra.from_astra_geometry()`](https://deepinv.org/api/stubs/deepinv.physics.TomographyWithAstra.html.md#deepinv.physics.TomographyWithAstra.from_astra_geometry) to build the operator directly from pre-created `astra` geometries ([#1102](https://github.com/deepinv/deepinv/pull/1102) by [Margaret Duff](https://github.com/MargaretDuff))
- Add downloadable pretrained weights to [`deepinv.models.FFDNet`](https://deepinv.org/api/stubs/deepinv.models.FFDNet.html.md#deepinv.models.FFDNet) ([#1357](https://github.com/deepinv/deepinv/pull/1357) by [Vicky De Ridder](https://github.com/nucli-vicky))
- Add [`deepinv.utils.plot()`](https://deepinv.org/api/stubs/deepinv.utils.plot.html.md#deepinv.utils.plot) to disable image rescaling with `rescale_mode=None`. ([#1339](https://github.com/deepinv/deepinv/pull/1339) by [Delphine Doutsas](https://github.com/dldou))
- Add [`deepinv.datasets.Set5HR`](https://deepinv.org/api/stubs/deepinv.datasets.Set5HR.html.md#deepinv.datasets.Set5HR), [`deepinv.datasets.BSD100HR`](https://deepinv.org/api/stubs/deepinv.datasets.BSD100HR.html.md#deepinv.datasets.BSD100HR), [`deepinv.datasets.McMaster`](https://deepinv.org/api/stubs/deepinv.datasets.McMaster.html.md#deepinv.datasets.McMaster) and [`deepinv.datasets.Kodak24`](https://deepinv.org/api/stubs/deepinv.datasets.Kodak24.html.md#deepinv.datasets.Kodak24) datasets ([#1382](https://github.com/deepinv/deepinv/pull/1382) by [Vicky De Ridder](https://github.com/nucli-vicky))

### Changed

- [`deepinv.models.FFDNet`](https://deepinv.org/api/stubs/deepinv.models.FFDNet.html.md#deepinv.models.FFDNet) default network parameters changed, to allow pretrained weights by default ([#1357](https://github.com/deepinv/deepinv/pull/1357) by [Vicky De Ridder](https://github.com/nucli-vicky))
- [`deepinv.models.PanNet`](https://deepinv.org/api/stubs/deepinv.models.PanNet.html.md#deepinv.models.PanNet) upsampling preserves intensity properly now. Existing PanNet weights may not perform well, but retraining should give improved performance compared to old weights. ([#1371](https://github.com/deepinv/deepinv/pull/1371) by [Vicky De Ridder](https://github.com/nucli-vicky))

### Fixed

- Fix description of channels in documentation of [`deepinv.datasets.NBUDataset`](https://deepinv.org/api/stubs/deepinv.datasets.NBUDataset.html.md#deepinv.datasets.NBUDataset) and provide link for more information on the dataset ([#1348](https://github.com/deepinv/deepinv/pull/1348) by [Delphine Doutsas](https://github.com/dldou))
- Fix inversion in [`deepinv.transform.Homography`](https://deepinv.org/api/stubs/deepinv.transform.Homography.html.md#deepinv.transform.Homography) transforms ([#1395](https://github.com/deepinv/deepinv/pull/1395) by [Jérémy Scanvic](https://github.com/jscanvic))

## v0.4.2

### New Features

- Add [`deepinv.loss.metric.RecoveryCoefficient`](https://deepinv.org/api/stubs/deepinv.loss.metric.RecoveryCoefficient.html.md#deepinv.loss.metric.RecoveryCoefficient) Recovery Coefficient (RC) metric to evaluate reconstructed activity relative to ground truth within a mask, with dtype-aware numerical stability and a dedicated loss transformation for training ([#1228](https://github.com/deepinv/deepinv/pull/1228) by [Kushagra Shukla](https://github.com/Kushagra481))
- Add 2D and 3D [`deepinv.physics.PET`](https://deepinv.org/api/stubs/deepinv.physics.PET.html.md#deepinv.physics.PET) ([#1099](https://github.com/deepinv/deepinv/pull/1099) by [Julian Tachella](https://github.com/tachella))
- Add support for multi-channel (chromatic) diffraction PSFs in [`deepinv.physics.generator.DiffractionBlurGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.DiffractionBlurGenerator.html.md#deepinv.physics.generator.DiffractionBlurGenerator) with physically consistent wavelength scaling of the pupil cut-off frequency and Zernike coefficients.  ([#1242](https://github.com/deepinv/deepinv/pull/1242) by [Pierre Weiss](https://www.math.univ-toulouse.fr/~weiss/) and [Florian Sarron](https://fsarron.github.io/))
- Add caching to demo/archive downloads ([#1234](https://github.com/deepinv/deepinv/pull/1234) by [Julian Tachella](https://github.com/tachella))
- Add [`deepinv.utils.load_tiff()`](https://deepinv.org/api/stubs/deepinv.utils.load_tiff.html.md#deepinv.utils.load_tiff) to load images/ volumes from TIFF files ([#1249](https://github.com/deepinv/deepinv/pull/1249) by [Andrew Wang](https://andrewwango.github.io/about/))
- Add [`deepinv.utils.plot_napari()`](https://deepinv.org/api/stubs/deepinv.utils.plot_napari.html.md#deepinv.utils.plot_napari) to interactively view 2D images/3D vols with napari ([#1249](https://github.com/deepinv/deepinv/pull/1249) by [Andrew Wang](https://andrewwango.github.io/about/))
- Add support for [`Liu-Jia padding`](https://deepinv.org/api/stubs/deepinv.physics.functional.liu_jia_pad.html.md#deepinv.physics.functional.liu_jia_pad) ([#934](https://github.com/deepinv/deepinv/pull/934) by [Jérémy Scanvic](https://github.com/jscanvic))
- Add support for TV-L1 priors [`deepinv.optim.TVL1Prior`](https://deepinv.org/api/stubs/deepinv.optim.TVL1Prior.html.md#deepinv.optim.TVL1Prior) ([#1236](https://github.com/deepinv/deepinv/pull/1236) by [Sarra Amiri](https://github.com/amirisarra18-jpg))
- Add support for mixed-precision (float16 and bfloat16) to the trainer ([#1208](https://github.com/deepinv/deepinv/pull/1208) by [Vicky De Ridder](https://github.com/nucli-vicky))
- Add [`deepinv.optim.OSEM`](https://deepinv.org/api/stubs/deepinv.optim.OSEM.html.md#deepinv.optim.OSEM) algorithm for tomographic reconstruction and update PET demos to showcase OSEM ([#1255](https://github.com/deepinv/deepinv/pull/1255) by [Thibaut Modrzyk](https://github.com/Tmodrzyk))
- Add utilities for subsetted tomography physics [`deepinv.physics.split_physics()`](https://deepinv.org/api/stubs/deepinv.physics.split_physics.html.md#deepinv.physics.split_physics) and [`deepinv.physics.split_measurements()`](https://deepinv.org/api/stubs/deepinv.physics.split_measurements.html.md#deepinv.physics.split_measurements) ([#1255](https://github.com/deepinv/deepinv/pull/1255) by [Thibaut Modrzyk](https://github.com/Tmodrzyk))
- Add [`deepinv.loss.metric.NRMSE`](https://deepinv.org/api/stubs/deepinv.loss.metric.NRMSE.html.md#deepinv.loss.metric.NRMSE) metric ([#1255](https://github.com/deepinv/deepinv/pull/1255) by [Thibaut Modrzyk](https://github.com/Tmodrzyk))
- Add espirit_crop parameter to control ESPIRiT multicoil MRI map estimation ([#1263](https://github.com/deepinv/deepinv/pull/1263) by [Andrew Wang](https://andrewwango.github.io/about/))
- Add [`deepinv.loss.metric.BRISQUE`](https://deepinv.org/api/stubs/deepinv.loss.metric.BRISQUE.html.md#deepinv.loss.metric.BRISQUE) and [`deepinv.loss.metric.NIMA`](https://deepinv.org/api/stubs/deepinv.loss.metric.NIMA.html.md#deepinv.loss.metric.NIMA) no-reference image quality metrics ([#1310](https://github.com/deepinv/deepinv/pull/1310) by [Julian Tachella](https://github.com/tachella))
- Add [`deepinv.datasets.BrainWebPET`](https://deepinv.org/api/stubs/deepinv.datasets.BrainWebPET.html.md#deepinv.datasets.BrainWebPET) ([#1286](https://github.com/deepinv/deepinv/pull/1286) by [Thibaut Modrzyk](https://github.com/Tmodrzyk))
- Add `mask_first` option to [`deepinv.physics.SpaceVaryingBlur`](https://deepinv.org/api/stubs/deepinv.physics.SpaceVaryingBlur.html.md#deepinv.physics.SpaceVaryingBlur) and [`deepinv.physics.functional.product_convolution2d()`](https://deepinv.org/api/stubs/deepinv.physics.functional.product_convolution2d.html.md#deepinv.physics.functional.product_convolution2d) ([#1347](https://github.com/deepinv/deepinv/pull/1347) by [Julian Tachella](https://github.com/tachella))
- Add `use_dict_output` option to every dataset class, returning a dict `{"x", "y", "params"}` instead of a tuple; propagate support to all deepinv internals ([#1244](https://github.com/deepinv/deepinv/pull/1244) by [Romain Vo](https://github.com/romainvo))

### Changed

- (Breaking) Drop support for deprecated parameters `num_channels` in [`deepinv.physics.generator.PSFGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.PSFGenerator.html.md#deepinv.physics.generator.PSFGenerator), [`deepinv.physics.generator.GaussianBlurGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.GaussianBlurGenerator.html.md#deepinv.physics.generator.GaussianBlurGenerator), [`deepinv.physics.generator.MotionBlurGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.MotionBlurGenerator.html.md#deepinv.physics.generator.MotionBlurGenerator), [`deepinv.physics.generator.DiffractionBlurGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.DiffractionBlurGenerator.html.md#deepinv.physics.generator.DiffractionBlurGenerator), [`deepinv.physics.generator.DiffractionBlurGenerator3D`](https://deepinv.org/api/stubs/deepinv.physics.generator.DiffractionBlurGenerator3D.html.md#deepinv.physics.generator.DiffractionBlurGenerator3D) ([#1242](https://github.com/deepinv/deepinv/pull/1242) by [Pierre Weiss](https://www.math.univ-toulouse.fr/~weiss/) and [Florian Sarron](https://fsarron.github.io/))
- Extend [`DST-I`](https://deepinv.org/api/stubs/deepinv.physics.functional.dst1.html.md#deepinv.physics.functional.dst1) to make it n-dimensional and add an option to have it compute the regular DST-I instead of the non-standard sign-flipped orthogonal variant ([#934](https://github.com/deepinv/deepinv/pull/934) by [Jérémy Scanvic](https://github.com/jscanvic))
- Extend: [`deepinv.optim.MLEM`](https://deepinv.org/api/stubs/deepinv.optim.MLEM.html.md#deepinv.optim.MLEM) now supports [`deepinv.physics.PET`](https://deepinv.org/api/stubs/deepinv.physics.PET.html.md#deepinv.physics.PET) ([#1255](https://github.com/deepinv/deepinv/pull/1255) by [Thibaut Modrzyk](https://github.com/Tmodrzyk))
- (Breaking) Make [`deepinv.optim.TVPrior()`](https://deepinv.org/api/stubs/deepinv.optim.TVPrior.html.md#deepinv.optim.TVPrior) compute an explicit choice of subgradient instead of using autodiff. ([#1271](https://github.com/deepinv/deepinv/pull/1271) by [Thibaut Modrzyk](https://github.com/Tmodrzyk))
- Extend: [`deepinv.models.DEAL`](https://deepinv.org/api/stubs/deepinv.models.DEAL.html.md#deepinv.models.DEAL) now accepts two new arguments: `inner_iter` and `outer_iter`. ([#1335](https://github.com/deepinv/deepinv/pull/1335) by [Paul Bernard](https://github.com/PAUL-BERNARD))

### Fixed

- Fix the `mask_first=False` pretrained [`deepinv.models.KernelIdentificationNetwork`](https://deepinv.org/api/stubs/deepinv.models.KernelIdentificationNetwork.html.md#deepinv.models.KernelIdentificationNetwork) ([#1347](https://github.com/deepinv/deepinv/pull/1347) by [Julian Tachella](https://github.com/tachella))
- Deprecate the `theta` attribute from [`deepinv.physics.Tomography`](https://deepinv.org/api/stubs/deepinv.physics.Tomography.html.md#deepinv.physics.Tomography) ([#1262](https://github.com/deepinv/deepinv/pull/1262) by [Matthieu Terris](https://github.com/matthieutrs))
- Remove redundant parameters `unitary` and `compute_inverse` from [`deepinv.physics.RandomPhaseRetrieval`](https://deepinv.org/api/stubs/deepinv.physics.RandomPhaseRetrieval.html.md#deepinv.physics.RandomPhaseRetrieval) ([#1220](https://github.com/deepinv/deepinv/pull/1220) by [Zhiyuan Hu](https://github.com/zhiyhu1605))
- Add [`deepinv.utils.DownloadError`](https://deepinv.org/api/stubs/deepinv.utils.DownloadError.html.md#deepinv.utils.DownloadError) to avoid CI errors when downloading demos/datasets ([#1234](https://github.com/deepinv/deepinv/pull/1234) by [Julian Tachella](https://github.com/tachella))
- Remove unconditional dtype conversion to `torch.cfloat` in [`deepinv.optim.phase_retrieval.spectral_methods()`](https://deepinv.org/api/stubs/deepinv.optim.phase_retrieval.spectral_methods.html.md#deepinv.optim.phase_retrieval.spectral_methods) ([#1216](https://github.com/deepinv/deepinv/pull/1216) by [Zhiyuan Hu](https://github.com/zhiyhu1605))
- Let quickstart run as default on Apple MPS, and all BM3D and DPIR to be used on MPS ([#1263](https://github.com/deepinv/deepinv/pull/1263) by [Andrew Wang](https://andrewwango.github.io/about/))
- Fix physics and rng devices incorrectly compared in noise and physics and add better device equal checking ([#1263](https://github.com/deepinv/deepinv/pull/1263) by [Andrew Wang](https://andrewwango.github.io/about/))
- Deprecate `deepinv.models.WaveletDenoiser.thresold_2D` and `deepinv.models.WaveletDenoiser.thresold_func` in favor of [`deepinv.models.WaveletDenoiser.threshold_2D()`](https://deepinv.org/api/stubs/deepinv.models.WaveletDenoiser.html.md#deepinv.models.WaveletDenoiser.threshold_2D) and [`deepinv.models.WaveletDenoiser.threshold_func()`](https://deepinv.org/api/stubs/deepinv.models.WaveletDenoiser.html.md#deepinv.models.WaveletDenoiser.threshold_func) ([#1266](https://github.com/deepinv/deepinv/pull/1266) by [Paul Bernard](https://github.com/PAUL-BERNARD))
- Fix kwargs applications in parent constructor calls in the constructors of [`deepinv.sampling.EDMDiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.EDMDiffusionSDE.html.md#deepinv.sampling.EDMDiffusionSDE), [`deepinv.sampling.SongDiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.SongDiffusionSDE.html.md#deepinv.sampling.SongDiffusionSDE) and [`deepinv.sampling.VariancePreservingDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.VariancePreservingDiffusion.html.md#deepinv.sampling.VariancePreservingDiffusion) ([#1278](https://github.com/deepinv/deepinv/pull/1278) by [Jérémy Scanvic](https://github.com/jscanvic))
- Fix [`deepinv.transform.rotate_via_shear()`](https://deepinv.org/api/stubs/deepinv.transform.rotate_via_shear.html.md#deepinv.transform.rotate_via_shear) for angles outside $[0, 2pi)$ ([#1236](https://github.com/deepinv/deepinv/pull/1236) by [Sarra Amiri](https://github.com/amirisarra18-jpg))
- Fix inversion in [`deepinv.transform.Reflect`](https://deepinv.org/api/stubs/deepinv.transform.Reflect.html.md#deepinv.transform.Reflect) ([#1236](https://github.com/deepinv/deepinv/pull/1236) by [Sarra Amiri](https://github.com/amirisarra18-jpg))
- (Breaking) Have `x_shift` represent horizontal shifts and `y_shift` vertical shifts in [`deepinv.transform.Shift`](https://deepinv.org/api/stubs/deepinv.transform.Shift.html.md#deepinv.transform.Shift) ([#1236](https://github.com/deepinv/deepinv/pull/1236) by [Sarra Amiri](https://github.com/amirisarra18-jpg))
- Force trainer non_blocking_transfers=False on MPS and CPU ([#1311](https://github.com/deepinv/deepinv/pull/1311) by [Andrew Wang](https://andrewwango.github.io/about/))
- Fix [`deepinv.physics.PET`](https://deepinv.org/api/stubs/deepinv.physics.PET.html.md#deepinv.physics.PET) incorrect device attribution of attenuation and background on update and incorrect handling of batched attenuation  ([#1331](https://github.com/deepinv/deepinv/pull/1331) by [Thibaut Modrzyk](https://github.com/Tmodrzyk))
- Raise informative error when filter is not set in [`deepinv.physics.Blur`](https://deepinv.org/api/stubs/deepinv.physics.Blur.html.md#deepinv.physics.Blur) and [`deepinv.physics.BlurFFT`](https://deepinv.org/api/stubs/deepinv.physics.BlurFFT.html.md#deepinv.physics.BlurFFT) ([#1337](https://github.com/deepinv/deepinv/pull/1337) by [Romain Vo](https://github.com/romainvo))

## v0.4.1

### New Features

- Add [`deepinv.models.DEAL`](https://deepinv.org/api/stubs/deepinv.models.DEAL.html.md#deepinv.models.DEAL) model ([#1107](https://github.com/deepinv/deepinv/pull/1107) by [Hossein Alimohammadi](https://github.com/Holimmo7))
- Add install guidelines for different platforms (`pixi`, `conda`, `pip`, `uv`) in docs ([#1108](https://github.com/deepinv/deepinv/pull/1108) by [Julian Tachella](https://github.com/tachella))
- Add [`deepinv.optim.SIRT`](https://deepinv.org/api/stubs/deepinv.optim.SIRT.html.md#deepinv.optim.SIRT) algorithm for tomographic reconstruction ([#985](https://github.com/deepinv/deepinv/pull/985) by [Thibaut Modrzyk](https://github.com/Tmodrzyk))
- Add [`deepinv.optim.MLEM`](https://deepinv.org/api/stubs/deepinv.optim.MLEM.html.md#deepinv.optim.MLEM) algorithm for Poisson inverse problems ([#1051](https://github.com/deepinv/deepinv/pull/1051) by [Thibaut Modrzyk](https://github.com/Tmodrzyk))
- Add the equivariant splitting loss [`deepinv.loss.EquivariantSplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.EquivariantSplittingLoss.html.md#deepinv.loss.EquivariantSplittingLoss) with equivariant reconstructors [`deepinv.models.EquivariantReconstructor`](https://deepinv.org/api/stubs/deepinv.models.EquivariantReconstructor.html.md#deepinv.models.EquivariantReconstructor) and virtual physics [`deepinv.physics.VirtualLinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.VirtualLinearPhysics.html.md#deepinv.physics.VirtualLinearPhysics) ([#881](https://github.com/deepinv/deepinv/pull/881) by [Jérémy Scanvic](https://github.com/jscanvic))
- Add `DEEPINV_CACHE_DIR` environment variable to set the cache directory for datasets and pretrained weights ([#1105](https://github.com/deepinv/deepinv/pull/1105) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- Add [`deepinv.models.SRResNet`](https://deepinv.org/api/stubs/deepinv.models.SRResNet.html.md#deepinv.models.SRResNet) (the generator of SRGAN) for single image super-resolution. ([#1164](https://github.com/deepinv/deepinv/pull/1164) by [Vicky De Ridder](https://github.com/nucli-vicky))
- NIQE weight fitting on custom datasets, using [`deepinv.loss.metric.NIQE.create_weights`](https://deepinv.org/api/stubs/deepinv.loss.metric.NIQE.html.md#deepinv.loss.metric.NIQE.create_weights) ([#911](https://github.com/deepinv/deepinv/pull/911) by [Vicky De Ridder](https://github.com/nucli-vicky))
- Add [`deepinv.loss.metric.GMSD`](https://deepinv.org/api/stubs/deepinv.loss.metric.GMSD.html.md#deepinv.loss.metric.GMSD), the Gradient Magnitude Similarity Deviation metric ([#1171](https://github.com/deepinv/deepinv/pull/1171) by [Vicky De Ridder](https://github.com/nucli-vicky))
- Add [`deepinv.models.FFDNet`](https://deepinv.org/api/stubs/deepinv.models.FFDNet.html.md#deepinv.models.FFDNet) for non-blind Gaussian denoising ([#1174](https://github.com/deepinv/deepinv/pull/1174) by [Vicky De Ridder](https://github.com/nucli-vicky))
- Extend [`deepinv.physics.functional.gaussian_blur()`](https://deepinv.org/api/stubs/deepinv.physics.functional.gaussian_blur.html.md#deepinv.physics.functional.gaussian_blur) to 1D and 3D. Add [`deepinv.physics.generator.GaussianBlurGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.GaussianBlurGenerator.html.md#deepinv.physics.generator.GaussianBlurGenerator) ([#1152](https://github.com/deepinv/deepinv/pull/1152) by [Romain Vo](https://github.com/romainvo))
- Add a fast re-implementation of BM3D for [`deepinv.models.BM3D`](https://deepinv.org/api/stubs/deepinv.models.BM3D.html.md#deepinv.models.BM3D) ([#1195](https://github.com/deepinv/deepinv/pull/1195) by [Kaibo Tang](https://github.com/kvttt))

### Changed

- Refactor CI to use `pixi` for compatibility with mixed conda/pip environments ([#1108](https://github.com/deepinv/deepinv/pull/1108) by [Julian Tachella](https://github.com/tachella))
- Add support for arbitrary learning-free reconstructors in the trainer ([#881](https://github.com/deepinv/deepinv/pull/881) by [Jérémy Scanvic](https://github.com/jscanvic))
- Make available every mask used at evaluation for splitting models [`deepinv.loss.SplittingLoss.SplittingModel`](https://deepinv.org/api/stubs/deepinv.loss.SplittingLoss.html.md#deepinv.loss.SplittingLoss.SplittingModel) when `eval_n_samples > 1` ([#881](https://github.com/deepinv/deepinv/pull/881) by [Jérémy Scanvic](https://github.com/jscanvic))
- Deprecate `Loss.name` in favor of the class name as done in the trainer ([#881](https://github.com/deepinv/deepinv/pull/881) by [Jérémy Scanvic](https://github.com/jscanvic))
- Update references for single pixel demo ([#1151](https://github.com/deepinv/deepinv/pull/1151) by [Laura C. Diaz-Delgado](https://github.com/LauraCD2))
- Add changelog section to the contributing guidelines ([#1153](https://github.com/deepinv/deepinv/pull/1153) by [Thibaut Modrzyk](https://github.com/Tmodrzyk))
- Unify patching / tiling and unpatching / un-tiling logic in physics and utils, with support for padding and non-overlapping patches. Add [`deepinv.utils.image_to_patches()`](https://deepinv.org/api/stubs/deepinv.utils.image_to_patches.html.md#deepinv.utils.image_to_patches) and [`deepinv.utils.patches_to_image()`](https://deepinv.org/api/stubs/deepinv.utils.patches_to_image.html.md#deepinv.utils.patches_to_image) utility functions, and refactor physics to use them instead of `unfold` ([#1104](https://github.com/deepinv/deepinv/pull/1104) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- New [`deepinv.loss.metric.NIQE`](https://deepinv.org/api/stubs/deepinv.loss.metric.NIQE.html.md#deepinv.loss.metric.NIQE) implementation, this drops PyIQA requirement, values given by NIQE vary between implementations ([#911](https://github.com/deepinv/deepinv/pull/911) by [Vicky De Ridder](https://github.com/nucli-vicky))
- (Breaking) Drop support for deprecated parameters replaced by `img_size` and `output_size` ([#1210](https://github.com/deepinv/deepinv/pull/1210) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- (Breaking) Drop support of `deepinv.utils.metric` in favor of `deepinv.loss.metric` ([#1210](https://github.com/deepinv/deepinv/pull/1210) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- (Breaking) Drop support of parameter `pinv` in [`deepinv.models.ArtifactRemoval`](https://deepinv.org/api/stubs/deepinv.models.ArtifactRemoval.html.md#deepinv.models.ArtifactRemoval) ([#1210](https://github.com/deepinv/deepinv/pull/1210) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- (Breaking) Drop support of `deepinv.train` in favor of [`deepinv.Trainer`](https://deepinv.org/api/stubs/deepinv.Trainer.html.md#deepinv.Trainer) ([#1210](https://github.com/deepinv/deepinv/pull/1210) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- (Breaking) Drop support of the deprecated parameter `eval_n_samples` in [`deepinv.loss.SplittingLoss.adapt_model()`](https://deepinv.org/api/stubs/deepinv.loss.SplittingLoss.html.md#deepinv.loss.SplittingLoss.adapt_model) ([#1210](https://github.com/deepinv/deepinv/pull/1210) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- (Breaking) Drop support of the deprecated parameter `fast` in [`deepinv.physics.CompressedSensing`](https://deepinv.org/api/stubs/deepinv.physics.CompressedSensing.html.md#deepinv.physics.CompressedSensing) ([#1210](https://github.com/deepinv/deepinv/pull/1210) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- (Breaking) Drop support of `deepinv.Trainer.log_metrics_wandb` ([#1210](https://github.com/deepinv/deepinv/pull/1210) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- (Breaking) Drop support of parameter and attribute `freq_plot` in [`deepinv.Trainer`](https://deepinv.org/api/stubs/deepinv.Trainer.html.md#deepinv.Trainer) ([#1210](https://github.com/deepinv/deepinv/pull/1210) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- (Breaking) Drop support of deprecated value `"old_sequency"` for parameter `ordering` in [`deepinv.physics.SinglePixelCamera`](https://deepinv.org/api/stubs/deepinv.physics.SinglePixelCamera.html.md#deepinv.physics.SinglePixelCamera) ([#1210](https://github.com/deepinv/deepinv/pull/1210) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- (Breaking) Drop support of parameter `sigma` in [`deepinv.loss.R2RLoss`](https://deepinv.org/api/stubs/deepinv.loss.R2RLoss.html.md#deepinv.loss.R2RLoss) in favor of `noise_model` ([#1210](https://github.com/deepinv/deepinv/pull/1210) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))

### Fixed

- Add warning when options `reduce="mean"` and `reduce="none"` are used in [`deepinv.physics.StackedLinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.StackedLinearPhysics.html.md#deepinv.physics.StackedLinearPhysics). Remove `reduction` argument from [`deepinv.distributed.framework.DistributedStackedLinearPhysics`](https://deepinv.org/api/stubs/deepinv.distributed.framework.DistributedStackedLinearPhysics.html.md#deepinv.distributed.framework.DistributedStackedLinearPhysics) ([#1071](https://github.com/deepinv/deepinv/pull/1071) by [Romain Vo](https://github.com/romainvo))
- Correct the value of `Transform.n_trans` in composed transformations ([#881](https://github.com/deepinv/deepinv/pull/881) by [Jérémy Scanvic](https://github.com/jscanvic))
- Add support for computing the evaluation loss for splitting losses like [`deepinv.loss.SplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.SplittingLoss.html.md#deepinv.loss.SplittingLoss) in the trainer ([#881](https://github.com/deepinv/deepinv/pull/881) by [Jérémy Scanvic](https://github.com/jscanvic))
- Add a deprecation warning in [`deepinv.utils.plot_inset()`](https://deepinv.org/api/stubs/deepinv.utils.plot_inset.html.md#deepinv.utils.plot_inset) in favor of [`deepinv.utils.plot()`](https://deepinv.org/api/stubs/deepinv.utils.plot.html.md#deepinv.utils.plot) with `plot_inset=True` ([#1148](https://github.com/deepinv/deepinv/pull/1148) by [Paul Bernard](https://github.com/PAUL-BERNARD)).
- Fix dimensions mismatch in [`deepinv.physics.TomographyWithAstra`](https://deepinv.org/api/stubs/deepinv.physics.TomographyWithAstra.html.md#deepinv.physics.TomographyWithAstra) with 3D phantoms ([#1137](https://github.com/deepinv/deepinv/pull/1137) by [Baptiste Legouix](https://github.com/blegouix))
- Add warning and error handling for negative inputs in BlurFFT and Poisson noise ([#1155](https://github.com/deepinv/deepinv/pull/1155) by [Thibaut Modrzyk](https://github.com/Tmodrzyk))
- Fix a bug in the custom backward of the least-squares solvers for non-leaf tensors ([#1146](https://github.com/deepinv/deepinv/pull/1146) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- Fix [`deepinv.sampling.DPS`](https://deepinv.org/api/stubs/deepinv.sampling.DPS.html.md#deepinv.sampling.DPS) instantiation and refactor to use new SDE interface ([#1127](https://github.com/deepinv/deepinv/pull/1127) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- Fix a bug that caused [`deepinv.models.BM3D`](https://deepinv.org/api/stubs/deepinv.models.BM3D.html.md#deepinv.models.BM3D) to silently break for multi-channel images when the number of channels is not 3 ([#1192](https://github.com/deepinv/deepinv/pull/1192) by [Kaibo Tang](https://github.com/kvttt))
- Fix option “mode” for the wavelet transform, which was not correctly propagated; add this option in [`deepinv.models.WaveletDictDenoiser`](https://deepinv.org/api/stubs/deepinv.models.WaveletDictDenoiser.html.md#deepinv.models.WaveletDictDenoiser) ([#1162](https://github.com/deepinv/deepinv/pull/1162) by [Irène Waldspurger](https://github.com/IWalds))

## v0.4.0

### New Features

- Add [`deepinv.models.WaveletNoiseEstimator`](https://deepinv.org/api/stubs/deepinv.models.WaveletNoiseEstimator.html.md#deepinv.models.WaveletNoiseEstimator) and [`deepinv.models.PatchCovarianceNoiseEstimator`](https://deepinv.org/api/stubs/deepinv.models.PatchCovarianceNoiseEstimator.html.md#deepinv.models.PatchCovarianceNoiseEstimator) for noise level estimation ([#1015](https://github.com/deepinv/deepinv/pull/1015) by [Matthieu Terris](https://github.com/matthieutrs))
- Add [`deepinv.physics.Scattering`](https://deepinv.org/api/stubs/deepinv.physics.Scattering.html.md#deepinv.physics.Scattering) physics for non-linear inverse scattering problems ([#1020](https://github.com/deepinv/deepinv/pull/1020) by [Julian Tachella](https://github.com/tachella))
- Add [`deepinv.physics.Physics.compute_norm()`](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics.compute_norm) local operator norm computation for non-linear physics ([#1020](https://github.com/deepinv/deepinv/pull/1020) by [Julian Tachella](https://github.com/tachella))
- Add [`deepinv.models.BilateralFilter`](https://deepinv.org/api/stubs/deepinv.models.BilateralFilter.html.md#deepinv.models.BilateralFilter) model ([#997](https://github.com/deepinv/deepinv/pull/997) by [Thomas Boulanger](https://github.com/LeRatonLaveurSolitaire))
- Add distributed computing framework with [`deepinv.distributed.DistributedContext`](https://deepinv.org/api/stubs/deepinv.distributed.DistributedContext.html.md#deepinv.distributed.DistributedContext), [`deepinv.distributed.framework.DistributedStackedPhysics`](https://deepinv.org/api/stubs/deepinv.distributed.framework.DistributedStackedPhysics.html.md#deepinv.distributed.framework.DistributedStackedPhysics), [`deepinv.distributed.framework.DistributedProcessing`](https://deepinv.org/api/stubs/deepinv.distributed.framework.DistributedProcessing.html.md#deepinv.distributed.framework.DistributedProcessing), [`deepinv.distributed.framework.DistributedDataFidelity`](https://deepinv.org/api/stubs/deepinv.distributed.framework.DistributedDataFidelity.html.md#deepinv.distributed.framework.DistributedDataFidelity) and [`deepinv.distributed.distribute()`](https://deepinv.org/api/stubs/deepinv.distributed.distribute.html.md#deepinv.distributed.distribute) factory function. Supports multi-GPU/multi-process execution with physics-based and spatial tiling distribution strategies ([#790\`](https://github.com/deepinv/deepinv/pull/790`) by [Benoît Malézieux](https://github.com/bmalezieux))
- Add [`deepinv.loss.metric.CosineSimilarity`](https://deepinv.org/api/stubs/deepinv.loss.metric.CosineSimilarity.html.md#deepinv.loss.metric.CosineSimilarity) to the metrics ([#944](https://github.com/deepinv/deepinv/pull/944) by [Avithal Lautman](https://github.com/avithal))
- New option to initialize 3D networks (DRUNet, DnCNN, DScCP) from pretrained 2D weights ([#958](https://github.com/deepinv/deepinv/pull/958) by [Romain Vo](https://github.com/romainvo))
- Add option to pass a noise level map to DRUNet and RAM ([#1056](https://github.com/deepinv/deepinv/pull/1056) by [Thomas Boulanger](https://github.com/LeRatonLaveurSolitaire))
- Add [`deepinv.physics.TiledSpaceVaryingBlur`](https://deepinv.org/api/stubs/deepinv.physics.TiledSpaceVaryingBlur.html.md#deepinv.physics.TiledSpaceVaryingBlur) physics and [`deepinv.physics.generator.TiledBlurGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.TiledBlurGenerator.html.md#deepinv.physics.generator.TiledBlurGenerator) ([#1033](https://github.com/deepinv/deepinv/pull/1033) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/) and [Paul Escande](https://pescande.perso.math.cnrs.fr/))

### Changed

- (Breaking) Make [`deepinv.physics.BlurFFT`](https://deepinv.org/api/stubs/deepinv.physics.BlurFFT.html.md#deepinv.physics.BlurFFT) compute a true convolution (now) instead of cross-correlation (before). It is now equivalent to [`deepinv.physics.Blur`](https://deepinv.org/api/stubs/deepinv.physics.Blur.html.md#deepinv.physics.Blur) with `padding="circular"` ([#825](https://github.com/deepinv/deepinv/pull/825) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/)). For even kernel sizes, the output is now shifted by one pixel to the top-left compared to before.
- Refactor folder structure of least-squares solvers ([#1011](https://github.com/deepinv/deepinv/pull/1011) by [Julian Tachella](https://github.com/tachella))
- Removed `eps` parameter from [`deepinv.optim.linear.conjugate_gradient()`](https://deepinv.org/api/stubs/deepinv.optim.linear.conjugate_gradient.html.md#deepinv.optim.linear.conjugate_gradient) ([#1011](https://github.com/deepinv/deepinv/pull/1011) by [Julian Tachella](https://github.com/tachella))
- [`deepinv.loss.metric.LPIPS`](https://deepinv.org/api/stubs/deepinv.loss.metric.LPIPS.html.md#deepinv.loss.metric.LPIPS) uses `torchmetrics` instead of `pyiqa`. ([#1041](https://github.com/deepinv/deepinv/pull/1041) by [Andrew Wang](https://andrewwango.github.io/about/))
- Remove `pyiqa` optional dep ([#1041](https://github.com/deepinv/deepinv/pull/1041) by [Andrew Wang](https://andrewwango.github.io/about/))
- Deprecated `verbose_individual_losses` parameter in [`deepinv.Trainer`](https://deepinv.org/api/stubs/deepinv.Trainer.html.md#deepinv.Trainer). Individual losses are now always added to logs when multiple losses are present ([#928](https://github.com/deepinv/deepinv/pull/928) by [Tiberiu Sabau](https://github.com/tibisabau))
- Deprecate historical attributes in HDF5Dataset ([#764](https://github.com/deepinv/deepinv/pull/764) by [Jérémy Scanvic](https://github.com/jscanvic))

### Fixed

- Implement/extend functional for 2D/3D convolution with spatial and FFT ([`deepinv.physics.functional.conv3d()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv3d.html.md#deepinv.physics.functional.conv3d) and  [`deepinv.physics.functional.conv_transpose3d()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv_transpose3d.html.md#deepinv.physics.functional.conv_transpose3d)), support all padding modes with equivalent outputs ([#825](https://github.com/deepinv/deepinv/pull/825) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- Fix ZeroPrior [`deepinv.optim.ZeroPrior`](https://deepinv.org/api/stubs/deepinv.optim.ZeroPrior.html.md#deepinv.optim.ZeroPrior) ([#1001](https://github.com/deepinv/deepinv/pull/1001) by [Victor Sechaud](https://github.com/vsechaud))
- Fix single-disperser CASSI adjointness ([#1029](https://github.com/deepinv/deepinv/pull/1029) by [Andrew Wang](https://andrewwango.github.io/about/))
- Add [`deepinv.physics.ComposedPhysics`](https://deepinv.org/api/stubs/deepinv.physics.ComposedPhysics.html.md#deepinv.physics.ComposedPhysics), [`deepinv.physics.ComposedLinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.ComposedLinearPhysics.html.md#deepinv.physics.ComposedLinearPhysics) to online documentation ([#1000](https://github.com/deepinv/deepinv/pull/1000) by [Romain Vo](https://github.com/romainvo))
- Have test ground truths returned in HDF5Dataset when present ([#764](https://github.com/deepinv/deepinv/pull/764) by [Jérémy Scanvic](https://github.com/jscanvic))
- Dispose of invalid physics parameters in HDF5Dataset loading ([#764](https://github.com/deepinv/deepinv/pull/764) by [Jérémy Scanvic](https://github.com/jscanvic))
- [`deepinv.utils.plot_ortho3D()`](https://deepinv.org/api/stubs/deepinv.utils.plot_ortho3D.html.md#deepinv.utils.plot_ortho3D) doesn’t perform sqrt on images ([#1068](https://github.com/deepinv/deepinv/pull/1068) by [Andrew Wang](https://andrewwango.github.io/about/))
- Deprecate `self.device` in [`deepinv.physics.LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics) and remove it from internal logic. Define it as a property until removed. ([#989](https://github.com/deepinv/deepinv/pull/989) by [Romain Vo](https://github.com/romainvo))
- Sample a unique factor per batch in [`deepinv.physics.generator.DownsamplingGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.DownsamplingGenerator.html.md#deepinv.physics.generator.DownsamplingGenerator) ([#1085](https://github.com/deepinv/deepinv/pull/1085) by [Romain Vo](https://github.com/romainvo))
- Add device argument to [`deepinv.physics.generator.GeneratorMixture`](https://deepinv.org/api/stubs/deepinv.physics.generator.GeneratorMixture.html.md#deepinv.physics.generator.GeneratorMixture) to fix mismatch with generator’s device and extend test suite in `test_generators.py` ([#1093](https://github.com/deepinv/deepinv/pull/1093) by [Romain Vo](https://github.com/romainvo))

## v0.3.7

### New Features

- Add [`deepinv.physics.LaplaceNoise`](https://deepinv.org/api/stubs/deepinv.physics.LaplaceNoise.html.md#deepinv.physics.LaplaceNoise) model ([#921](https://github.com/deepinv/deepinv/pull/921) by [Brayan Monroy](https://github.com/bemc22))
- New way to create optimization models. Standard optimization algorithms (and their unfolded versions) can be created using their class name directly instead of using the `optim_builder` (or `unfolded_builder`) function. ([#592](https://github.com/deepinv/deepinv/pull/592) by [Samuel Hurault](https://github.com/samuro95))
- New `vmin` and `vmax` arguments in [`deepinv.utils.plot()`](https://deepinv.org/api/stubs/deepinv.utils.plot.html.md#deepinv.utils.plot) to set custom clipping bounds when using `rescale_mode='clip'` ([#967](https://github.com/deepinv/deepinv/pull/967) by [Thibaut Modrzyk](https://github.com/Tmodrzyk))
- Added [`deepinv.utils.dirac_comb()`](https://deepinv.org/api/stubs/deepinv.utils.dirac_comb.html.md#deepinv.utils.dirac_comb) and [`deepinv.utils.dirac_comb_like()`](https://deepinv.org/api/stubs/deepinv.utils.dirac_comb_like.html.md#deepinv.utils.dirac_comb_like) ([#946](https://github.com/deepinv/deepinv/pull/946) by [Julian Tachella](https://github.com/tachella))
- Added [testmon](https://www.testmon.org/) and conditional run of sphinx-gallery examples to CI to speed up tests ([#966](https://github.com/deepinv/deepinv/pull/966) by [Julian Tachella](https://github.com/tachella))
- Add [`kernel estimation network`](https://deepinv.org/api/stubs/deepinv.models.KernelIdentificationNetwork.html.md#deepinv.models.KernelIdentificationNetwork) for blind deconvolution ([#971](https://github.com/deepinv/deepinv/pull/971) by [Julian Tachella](https://github.com/tachella))
- Add blind inverse problems section to reconstruction user guide ([#971](https://github.com/deepinv/deepinv/pull/971) by [Julian Tachella](https://github.com/tachella))
- Add [`deepinv.loss.metric.BlurStrength`](https://deepinv.org/api/stubs/deepinv.loss.metric.BlurStrength.html.md#deepinv.loss.metric.BlurStrength) and [`deepinv.loss.metric.SharpnessIndex`](https://deepinv.org/api/stubs/deepinv.loss.metric.SharpnessIndex.html.md#deepinv.loss.metric.SharpnessIndex) no-reference metrics for blind deblurring ([#971](https://github.com/deepinv/deepinv/pull/971) by [Julian Tachella](https://github.com/tachella))

### Changed

- Faster [`deepinv.models.RAM`](https://deepinv.org/api/stubs/deepinv.models.RAM.html.md#deepinv.models.RAM) implementation by avoiding certain redundant computations ([#946](https://github.com/deepinv/deepinv/pull/946) by [Julian Tachella](https://github.com/tachella))
- (Breaking) Change [`deepinv.physics.TomographyWithAstra`](https://deepinv.org/api/stubs/deepinv.physics.TomographyWithAstra.html.md#deepinv.physics.TomographyWithAstra) physics interface to better match the interface of the PyTorch-based `Tomography` physics ([#747](https://github.com/deepinv/deepinv/pull/747) by [Alexander Skorikov](https://github.com/askorikov))
- Add support for Poisson2Sparse ([#677](https://github.com/deepinv/deepinv/pull/677) by [Jérémy Scanvic](https://github.com/jscanvic))
- (Breaking) `Tomography` physics uses the true adjoint by default. `Tomography` and `TomographyWithAstra` implement the pseudo-inverse as the solution of a least-squares problem, with the option to use `fbp`. ([#930](https://github.com/deepinv/deepinv/pull/930) by [Romain Vo](https://github.com/romainvo))
- [`deepinv.models.UNet`](https://deepinv.org/api/stubs/deepinv.models.UNet.html.md#deepinv.models.UNet) now accepts a new (optional) argument `channels_per_scale` to control the number of feature maps at each stage. It now also supports arbitrary number of scales and bias-free batchnorm is supported for 3D variant; also clean-up code ([#976](https://github.com/deepinv/deepinv/pull/976) by [Vicky De Ridder](https://github.com/nucli-vicky))
- Add a check in `deepinv.datasets.FMD` to avoid unnecessary downloads ([#962](https://github.com/deepinv/deepinv/pull/962) by [Jérémy Scanvic](https://github.com/jscanvic))
- Trainer checkpoint loading verbose ([#982](https://github.com/deepinv/deepinv/pull/982) by [Andrew Wang](https://andrewwango.github.io/about/))

### Fixed

- Fixed [`deepinv.sampling.DPS`](https://deepinv.org/api/stubs/deepinv.sampling.DPS.html.md#deepinv.sampling.DPS) initialization when measurements have different size than image ([#946](https://github.com/deepinv/deepinv/pull/946) by [Julian Tachella](https://github.com/tachella))
- Fixed [`deepinv.physics.Ptychography`](https://deepinv.org/api/stubs/deepinv.physics.Ptychography.html.md#deepinv.physics.Ptychography) `A_dagger` initialization bug ([#946](https://github.com/deepinv/deepinv/pull/946) by [Julian Tachella](https://github.com/tachella))
- Reduce CI cache size by using `uv` caching ([#943](https://github.com/deepinv/deepinv/pull/943) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- `generate_dataset` received a general refactor, now supports PIL image datasets and doesn’t break when validation dataset returns `TensorList` ([#948](https://github.com/deepinv/deepinv/pull/948) by [Vicky De Ridder](https://github.com/nucli-vicky))
- test_physics.test_tomography correctly implements the pseudo-inverse test (:gh: `930` by [Romain Vo](https://github.com/romainvo))
- [`deepinv.physics.Tomography`](https://deepinv.org/api/stubs/deepinv.physics.Tomography.html.md#deepinv.physics.Tomography) now correctly handles multi-channel data ([#960](https://github.com/deepinv/deepinv/pull/960) by [Julian Tachella](https://github.com/tachella))

## v0.3.6

### New Features

- Add dataset for patch sampling from (large) nD images without loading entire images into memory ([#806](https://github.com/deepinv/deepinv/pull/806) by [Vicky De Ridder](https://github.com/nucli-vicky))
- Add support for 3D CNNs ([#869](https://github.com/deepinv/deepinv/pull/869) by [Vicky De Ridder](https://github.com/nucli-vicky))
- Add support for complex dtypes in [`deepinv.models.WaveletDenoiser`](https://deepinv.org/api/stubs/deepinv.models.WaveletDenoiser.html.md#deepinv.models.WaveletDenoiser), [`deepinv.models.WaveletDictDenoiser`](https://deepinv.org/api/stubs/deepinv.models.WaveletDictDenoiser.html.md#deepinv.models.WaveletDictDenoiser) and [`deepinv.optim.WaveletPrior`](https://deepinv.org/api/stubs/deepinv.optim.WaveletPrior.html.md#deepinv.optim.WaveletPrior) ([#738](https://github.com/deepinv/deepinv/pull/738) by [Chaithya G R](https://github.com/chaithyagr))
- dinv.io functions for loading DICOM, NIFTI, COS, GEOTIFF etc. ([#768](https://github.com/deepinv/deepinv/pull/768) by [Andrew Wang](https://andrewwango.github.io/about/))
- Add `Open in Colab` button to examples ([#907](https://github.com/deepinv/deepinv/pull/907) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- Better diffraction blur generator with higher Zernike orders, rotation and apodization ([#826](https://github.com/deepinv/deepinv/pull/826) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- Rotation transform via shear operations [`deepinv.transform.rotate.rotate_via_shear()`](https://deepinv.org/api/stubs/deepinv.transform.rotate_via_shear.html.md#deepinv.transform.rotate_via_shear) for reduced interpolation artifacts ([#826](https://github.com/deepinv/deepinv/pull/826) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- Zernike polynomials interface [`deepinv.physics.generator.Zernike`](https://deepinv.org/api/stubs/deepinv.physics.generator.Zernike.html.md#deepinv.physics.generator.Zernike) for any (n, m) order ([#826](https://github.com/deepinv/deepinv/pull/826) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- Integration with HuggingFace Diffusers library to use pretrained diffusion models as denoisers and for posterior sampling ([#893](https://github.com/deepinv/deepinv/pull/893) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))

### Changed

- load_np_url now returns tensors, load_url helper function moved to io ([#768](https://github.com/deepinv/deepinv/pull/768) by [Andrew Wang](https://andrewwango.github.io/about/))
- utils/signal.py renamed to signals.py to avoid stdlib conflict ([#768](https://github.com/deepinv/deepinv/pull/768) by [Andrew Wang](https://andrewwango.github.io/about/))
- utils.get_data_home now creates folder if not exist ([#768](https://github.com/deepinv/deepinv/pull/768) by [Andrew Wang](https://andrewwango.github.io/about/))
- Update attention computation to use [`torch.nn.functional.scaled_dot_product_attention()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention) ([#883](https://github.com/deepinv/deepinv/pull/883) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))

### Fixed

- (Breaking) Make the image saving logic in `Trainer` more conventional ([#904](https://github.com/deepinv/deepinv/pull/904) by [Jérémy Scanvic](https://github.com/jscanvic))
- Blur physics objects now put new filters to physics device regardless of input filter device ([#844](https://github.com/deepinv/deepinv/pull/844) by [Vicky De Ridder](https://github.com/nucli-vicky))
- Set14HR dataset now downloads from a different source (and has slightly different folderstructure), since old link broke. ([#845](https://github.com/deepinv/deepinv/pull/845) by [Vicky De Ridder](https://github.com/nucli-vicky))
- LsdirHR dataset now downloads from a different source (since old link broke) and correctly contains the specific folder images, instead of everything in root ([#866](https://github.com/deepinv/deepinv/pull/866) by [Vicky De Ridder](https://github.com/nucli-vicky))
- Fix typo in docstring of ItohFidelity and SpatialUnwrapping demo ([#860](https://github.com/deepinv/deepinv/pull/860) by [Brayan Monroy](https://github.com/bemc22))
- Fix unhandled import error in CBSD68 if datasets is not installed ([#868](https://github.com/deepinv/deepinv/pull/868) by [Johannes Hertrich](https://johertrich.github.io/))
- Add support for complex signals in PSNR ([#738](https://github.com/deepinv/deepinv/pull/738) by [Jérémy Scanvic](https://github.com/jscanvic))
- Add a warning in SwinIR when upsampling parameters are inconsistent ([#909](https://github.com/deepinv/deepinv/pull/909) by [Jérémy Scanvic](https://github.com/jscanvic))
- Fix formula error in Zernike polynomials, extend to higher order ([#826](https://github.com/deepinv/deepinv/pull/826) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- Fix scaling of measurement and samples in posterior sampling with diffusion SDEs ([#893](https://github.com/deepinv/deepinv/pull/893) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))

## v0.3.5

### New Features

- Add statistics for SAR imaging + fix variance of [`deepinv.physics.GammaNoise`](https://deepinv.org/api/stubs/deepinv.physics.GammaNoise.html.md#deepinv.physics.GammaNoise) in doc ([#740](https://github.com/deepinv/deepinv/pull/740) by [Louise Friot Giroux](https://github.com/Louisefg))
- Add imshow kwargs to plot ([#791](https://github.com/deepinv/deepinv/pull/791) by [Andrew Wang](https://andrewwango.github.io/about/))
- Add [`deepinv.physics.RicianNoise`](https://deepinv.org/api/stubs/deepinv.physics.RicianNoise.html.md#deepinv.physics.RicianNoise) model ([#805](https://github.com/deepinv/deepinv/pull/805) by [Vicky De Ridder](https://github.com/nucli-vicky))
- Add manual physics to reduced resolution loss ([#808](https://github.com/deepinv/deepinv/pull/808) by [Andrew Wang](https://andrewwango.github.io/about/))
- Multi-coil MRI coil-map estimation acceleration via CuPy ([#781](https://github.com/deepinv/deepinv/pull/781) by [Andrew Wang](https://andrewwango.github.io/about/))
- Add SpatialUnwrapping forward model and ItohFidelity data fidelity ([#723](https://github.com/deepinv/deepinv/pull/723) by [Brayan Monroy](https://github.com/bemc22))

### Changed

- (Breaking) Make [`deepinv.datasets.HDF5Dataset`](https://deepinv.org/api/stubs/deepinv.datasets.HDF5Dataset.html.md#deepinv.datasets.HDF5Dataset) similar to [`deepinv.Trainer`](https://deepinv.org/api/stubs/deepinv.Trainer.html.md#deepinv.Trainer) in the unsupervised setting by using NaNs for ground truths instead of a copy of the measurements ([#761](https://github.com/deepinv/deepinv/pull/761) by [Jérémy Scanvic](https://github.com/jscanvic))
- Add `squared` parameter to `LinearPhysics.compute_norm()` and `compute_sqnorm()` method ([#832](https://github.com/deepinv/deepinv/pull/832) by [Jérémy Scanvic](https://github.com/jscanvic))
- Allow self-supervised eval by removing the model.eval() from Trainer.train() ([#777](https://github.com/deepinv/deepinv/pull/777) by [Julian Tachella](https://github.com/tachella))
- Make tqdm progress bar auto-resize ([#835](https://github.com/deepinv/deepinv/pull/835) by [Andrew Wang](https://andrewwango.github.io/about/))
- (Breaking) Normalize the Tomography operator with proper spectral norm computation. Set the default normalization behavior to `True` for both CT operators ([#715](https://github.com/deepinv/deepinv/pull/715) by [Romain Vo](https://github.com/romainvo))

### Fixed

- Use the learning-free model for learning-free metrics in [`deepinv.Trainer`](https://deepinv.org/api/stubs/deepinv.Trainer.html.md#deepinv.Trainer) ([#788](https://github.com/deepinv/deepinv/pull/788) by [Jérémy Scanvic](https://github.com/jscanvic))
- Fix device [`deepinv.utils.dirac_like()`](https://deepinv.org/api/stubs/deepinv.utils.dirac_like.html.md#deepinv.utils.dirac_like) and `deepinv.physics.blur.bilinear_filter`, `deepinv.physics.blur.bicubic_filter` and `deepinv.physics.blur.gaussian_blur` filters ([#785](https://github.com/deepinv/deepinv/pull/785) by [Julian Tachella](https://github.com/tachella))
- Fix device [`deepinv.utils.dirac_like()`](https://deepinv.org/api/stubs/deepinv.utils.dirac_like.html.md#deepinv.utils.dirac_like) and [`deepinv.physics.functional.bilinear_filter()`](https://deepinv.org/api/stubs/deepinv.physics.functional.bilinear_filter.html.md#deepinv.physics.functional.bilinear_filter), [`deepinv.physics.functional.bicubic_filter()`](https://deepinv.org/api/stubs/deepinv.physics.functional.bicubic_filter.html.md#deepinv.physics.functional.bicubic_filter) and [`deepinv.physics.functional.gaussian_blur()`](https://deepinv.org/api/stubs/deepinv.physics.functional.gaussian_blur.html.md#deepinv.physics.functional.gaussian_blur) filters ([#785](https://github.com/deepinv/deepinv/pull/785) by [Julian Tachella](https://github.com/tachella))
- Fix positivity + batching gamma least squares solvers ([#785](https://github.com/deepinv/deepinv/pull/785) by [Julian Tachella](https://github.com/tachella) and [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- Fix and test [`deepinv.models.RAM`](https://deepinv.org/api/stubs/deepinv.models.RAM.html.md#deepinv.models.RAM) scaling issues ([#785](https://github.com/deepinv/deepinv/pull/785) by [Julian Tachella](https://github.com/tachella))
- Reduced CI python version tests ([#746](https://github.com/deepinv/deepinv/pull/746) by [Matthieu Terris](https://github.com/matthieutrs))
- Fix scaling issue in [`deepinv.sampling.DiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSDE.html.md#deepinv.sampling.DiffusionSDE) ([#772](https://github.com/deepinv/deepinv/pull/772) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- All splitting losses fixed to work with changing image sizes and with multicoil MRI ([#778](https://github.com/deepinv/deepinv/pull/778) by [Andrew Wang](https://andrewwango.github.io/about/))
- [`deepinv.Trainer`](https://deepinv.org/api/stubs/deepinv.Trainer.html.md#deepinv.Trainer) treats batch of nans as no ground truth ([#793](https://github.com/deepinv/deepinv/pull/793) by [Andrew Wang](https://andrewwango.github.io/about/))
- Save training loss history ([#777](https://github.com/deepinv/deepinv/pull/777) by [Julian Tachella](https://github.com/tachella))
- Fix docstring formatting in BDSDS500 dataset ([#816](https://github.com/deepinv/deepinv/pull/816) by [Brayan Monroy](https://github.com/bemc22))
- Remove unnecessary tensor cloning from DDRM and DPS ([#834](https://github.com/deepinv/deepinv/pull/834) by [Vicky De Ridder](https://github.com/nucli-vicky))
- Change deprecated `torch.norm` calls to `torch.linalg.vector_norm` ([#840](https://github.com/deepinv/deepinv/pull/840) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))

## v0.3.4

### New Features

- Quickstart tutorials + clean examples ([#622](https://github.com/deepinv/deepinv/pull/622) by [Andrew Wang](https://andrewwango.github.io/about/))
- Dataset base class + [`deepinv.datasets.ImageFolder`](https://deepinv.org/api/stubs/deepinv.datasets.ImageFolder.html.md#deepinv.datasets.ImageFolder) and [`deepinv.datasets.TensorDataset`](https://deepinv.org/api/stubs/deepinv.datasets.TensorDataset.html.md#deepinv.datasets.TensorDataset) classes ([#622](https://github.com/deepinv/deepinv/pull/622) by [Andrew Wang](https://andrewwango.github.io/about/))
- Added GitHub action checking import time ([#680](https://github.com/deepinv/deepinv/pull/680) by [Julian Tachella](https://github.com/tachella))
- Client model for server-side inference for using models in the cloud ([#691](https://github.com/deepinv/deepinv/pull/691) by [Andrew Wang](https://andrewwango.github.io/about/))
- Reduced resolution self-supervised loss ([#735](https://github.com/deepinv/deepinv/pull/735) by [Andrew Wang](https://andrewwango.github.io/about/))
- Add [`deepinv.utils.disable_tex()`](https://deepinv.org/api/stubs/deepinv.utils.disable_tex.html.md#deepinv.utils.disable_tex) to disable LaTeX ([#726](https://github.com/deepinv/deepinv/pull/726) by [Andrew Wang](https://andrewwango.github.io/about/))
- Add BSDS500 dataset ([#749](https://github.com/deepinv/deepinv/pull/749) by [Johannes Hertrich](https://johertrich.github.io/) and [Sebastian Neumayer](https://www.tu-chemnitz.de/mathematik/invimg/index.en.php))
- O(1) memory backprop for linear solver and example ([#739](https://github.com/deepinv/deepinv/pull/739) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))

### Changed

- Move mixins to utils and reduce number of cross-submodule top-level imports ([#680](https://github.com/deepinv/deepinv/pull/680) by [Andrew Wang](https://andrewwango.github.io/about/))
- [`deepinv.datasets.PatchDataset`](https://deepinv.org/api/stubs/deepinv.datasets.PatchDataset.html.md#deepinv.datasets.PatchDataset) returns tensors and not tuples ([#622](https://github.com/deepinv/deepinv/pull/622) by [Andrew Wang](https://andrewwango.github.io/about/))

### Fixed

- Fixed natsorted issue ([#680](https://github.com/deepinv/deepinv/pull/680) by [Julian Tachella](https://github.com/tachella))
- Fix full-reference metrics used with measurement-only dataset ([#622](https://github.com/deepinv/deepinv/pull/622) by [Andrew Wang](https://andrewwango.github.io/about/))
- Batching [`deepinv.physics.generator.DownsamplingGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.DownsamplingGenerator.html.md#deepinv.physics.generator.DownsamplingGenerator) in the case of multiple filters ([#690](https://github.com/deepinv/deepinv/pull/690) by [Matthieu Terris](https://github.com/matthieutrs))
- NaN motion blur generator ([#685](https://github.com/deepinv/deepinv/pull/685) by [Matthieu Terris](https://github.com/matthieutrs))
- Fix the condition for break in compute_norm ([#699](https://github.com/deepinv/deepinv/pull/699) by [Quentin Barthélemy](https://github.com/qbarthelemy))
- Python 3.9 backward compatibility and zip_strict ([#707](https://github.com/deepinv/deepinv/pull/707) by [Andrew Wang](https://andrewwango.github.io/about/))
- Fix numerical instability of [`deepinv.optim.linear.bicgstab()`](https://deepinv.org/api/stubs/deepinv.optim.linear.bicgstab.html.md#deepinv.optim.linear.bicgstab) solver ([#739](https://github.com/deepinv/deepinv/pull/739) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))

## v0.3.3

### New Features

- Automatic A_adjoint, U_adjoint and V computation for user-defined physics ([#658](https://github.com/deepinv/deepinv/pull/658) by [Julian Tachella](https://github.com/tachella))
- Add [`deepinv.models.RAM`](https://deepinv.org/api/stubs/deepinv.models.RAM.html.md#deepinv.models.RAM) model ([#524](https://github.com/deepinv/deepinv/pull/524) by [Matthieu Terris](https://github.com/matthieutrs))
- FastMRI better raw data loading: load targets from different folder for test sets, load mask from test set, prewhitening, normalisation ([#608](https://github.com/deepinv/deepinv/pull/608) by [Andrew Wang](https://andrewwango.github.io/about/))
- SKM-TEA raw MRI dataset ([#608](https://github.com/deepinv/deepinv/pull/608) by [Andrew Wang](https://andrewwango.github.io/about/))
- New downsampling physics that matches MATLAB bicubic imresize ([#608](https://github.com/deepinv/deepinv/pull/608) by [Andrew Wang](https://andrewwango.github.io/about/))

### Changed

- Rename the normalizing function `deepinv.utils.rescale_img` to [`deepinv.utils.normalize_signal()`](https://deepinv.org/api/stubs/deepinv.utils.normalize_signal.html.md#deepinv.utils.normalize_signal) ([#641](https://github.com/deepinv/deepinv/pull/641) by [Jérémy Scanvic](https://github.com/jscanvic))
- Changed default linear solver from `CG` to [`deepinv.optim.linear.lsqr()`](https://deepinv.org/api/stubs/deepinv.optim.linear.lsqr.html.md#deepinv.optim.linear.lsqr) ([#658](https://github.com/deepinv/deepinv/pull/658) by [Julian Tachella](https://github.com/tachella))
- Added positive clipping by default and gain minimum in [`deepinv.physics.PoissonGaussianNoise`](https://deepinv.org/api/stubs/deepinv.physics.PoissonGaussianNoise.html.md#deepinv.physics.PoissonGaussianNoise) ([#658](https://github.com/deepinv/deepinv/pull/658) by [Julian Tachella](https://github.com/tachella)).

### Fixed

- Fix downsampling generator batching ([#608](https://github.com/deepinv/deepinv/pull/608) by [Andrew Wang](https://andrewwango.github.io/about/))
- Fix memory leak in [`deepinv.physics.Tomography`](https://deepinv.org/api/stubs/deepinv.physics.Tomography.html.md#deepinv.physics.Tomography) when using autograd ([#651](https://github.com/deepinv/deepinv/pull/651) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- Fix the circular padded [`deepinv.models.UNet`](https://deepinv.org/api/stubs/deepinv.models.UNet.html.md#deepinv.models.UNet) ([#653](https://github.com/deepinv/deepinv/pull/653) by [Victor Sechaud](https://github.com/vsechaud))
- Clamp constant signals in [`deepinv.utils.normalize_signal()`](https://deepinv.org/api/stubs/deepinv.utils.normalize_signal.html.md#deepinv.utils.normalize_signal) to ensure they are normalized ([#641](https://github.com/deepinv/deepinv/pull/641) by [Jérémy Scanvic](https://github.com/jscanvic))
- Fix ZeroNoise default missing in [`deepinv.physics.ZeroNoise`](https://deepinv.org/api/stubs/deepinv.physics.ZeroNoise.html.md#deepinv.physics.ZeroNoise) ([#658](https://github.com/deepinv/deepinv/pull/658) by [Julian Tachella](https://github.com/tachella))

## v0.3.2

### New Features

- Add support for astra-toolbox CT operators (parallel, fan, cone) with [`deepinv.physics.TomographyWithAstra`](https://deepinv.org/api/stubs/deepinv.physics.TomographyWithAstra.html.md#deepinv.physics.TomographyWithAstra) ([#474](https://github.com/deepinv/deepinv/pull/474) by [Romain Vo](https://github.com/romainvo))
- Add [`deepinv.physics.Physics.clone()`](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics.clone) ([#534](https://github.com/deepinv/deepinv/pull/534) by [Jérémy Scanvic](https://github.com/jscanvic))

### Changed

- Make autograd use the base linear operator for [`deepinv.physics.adjoint_function()`](https://deepinv.org/api/stubs/deepinv.physics.adjoint_function.html.md#deepinv.physics.adjoint_function) ([#519](https://github.com/deepinv/deepinv/pull/519) by [Jérémy Scanvic](https://github.com/jscanvic))
- Parallelize the test suite making it 15% faster ([#522](https://github.com/deepinv/deepinv/pull/522) by [Jérémy Scanvic](https://github.com/jscanvic))
- Adjust backward paths for tomography ([#535](https://github.com/deepinv/deepinv/pull/535) by [Johannes Hertrich](https://johertrich.github.io/))
- Update python version to 3.10+ ([#605](https://github.com/deepinv/deepinv/pull/605) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- Update the library dependencies, issue template, codecov report on linux only ([#654](https://github.com/deepinv/deepinv/pull/654) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))

### Fixed

- Fix the total loss reported by the trainer ([#515](https://github.com/deepinv/deepinv/pull/515) by [Jérémy Scanvic](https://github.com/jscanvic))
- Fix the gradient norm reported by the trainer ([#520](https://github.com/deepinv/deepinv/pull/520) by [Jérémy Scanvic](https://github.com/jscanvic))
- Fix that the max_pixel option in PSNR and SSIM and add analgous min_pixel option ([#535](https://github.com/deepinv/deepinv/pull/535) by [Johannes Hertrich](https://johertrich.github.io/))
- Fix some issues related to denoisers: ICNN grad not working inside torch.no_grad(), batch of image and batch of sigma for some denoisers (DiffUNet, BM3D, TV, Wavemet), EPLL error when batch size > 1 ([#530](https://github.com/deepinv/deepinv/pull/530) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- Batching WaveletPrior and fix iwt ([#530](https://github.com/deepinv/deepinv/pull/530) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- Fix on unreliable/inconsistent automatic choosing GPU with most free VRAM ([#570](https://github.com/deepinv/deepinv/pull/570) by Fedor Goncharov)

## v0.3.1

### New Features

- Added [`deepinv.physics.SaltPepperNoise`](https://deepinv.org/api/stubs/deepinv.physics.SaltPepperNoise.html.md#deepinv.physics.SaltPepperNoise) for impulse noise ([#472](https://github.com/deepinv/deepinv/pull/472) by [Thomas Moreau](https://github.com/tomMoral)).
- Add measurement augmentation VORTEX loss ([#410](https://github.com/deepinv/deepinv/pull/410) by [Andrew Wang](https://andrewwango.github.io/about/))
- Add non-geometric data augmentations (noise, phase errors) ([#410](https://github.com/deepinv/deepinv/pull/410) by [Andrew Wang](https://andrewwango.github.io/about/))
- Make [`deepinv.physics.generator.PhysicsGenerator.average`](https://deepinv.org/api/stubs/deepinv.physics.generator.PhysicsGenerator.html.md#deepinv.physics.generator.PhysicsGenerator.average) use batches ([#488](https://github.com/deepinv/deepinv/pull/488) by [Jérémy Scanvic](https://github.com/jscanvic))
- MRI losses subclass, weighted-SSDU, Robust-SSDU loss functions + more mask generators ([#416](https://github.com/deepinv/deepinv/pull/416) by [Keying Guo](https://github.com/g-keying) and [Andrew Wang](https://andrewwango.github.io/about/))
- Multi-coil MRI estimates sens maps with sigpy ESPIRiT, MRISliceTransform better loads raw data by estimating coil maps and generating masks ([#416](https://github.com/deepinv/deepinv/pull/416) by [Andrew Wang](https://andrewwango.github.io/about/))
- Add HaarPSI metric + metric standardization ([#416](https://github.com/deepinv/deepinv/pull/416) by [Andrew Wang](https://andrewwango.github.io/about/))
- Add ENSURE loss ([#454](https://github.com/deepinv/deepinv/pull/454) by [Andrew Wang](https://andrewwango.github.io/about/))

### Changed

- Added cake_cutting, zig_zag and xy orderings in [`deepinv.physics.SinglePixelCamera`](https://deepinv.org/api/stubs/deepinv.physics.SinglePixelCamera.html.md#deepinv.physics.SinglePixelCamera) physics ([#475](https://github.com/deepinv/deepinv/pull/475) by [Brayan Monroy](https://github.com/bemc22)).

### Fixed

- Fix images not showing in sphinx examples ([#478](https://github.com/deepinv/deepinv/pull/478) by [Matthieu Terris](https://github.com/matthieutrs))
- Fix plot_inset not showing ([#455](https://github.com/deepinv/deepinv/pull/455) by [Andrew Wang](https://andrewwango.github.io/about/))
- Fix latex rendering in [`deepinv.utils.plotting.config_matplotlib()`](https://deepinv.org/api/stubs/deepinv.utils.plotting.config_matplotlib.html.md#deepinv.utils.plotting.config_matplotlib)  ([#452](https://github.com/deepinv/deepinv/pull/452) by [Romain Vo](https://github.com/romainvo))
- Get rid of unnecessary file system writes in [`deepinv.utils.get_freer_gpu()`](https://deepinv.org/api/stubs/deepinv.utils.get_freer_gpu.html.md#deepinv.utils.get_freer_gpu) ([#468](https://github.com/deepinv/deepinv/pull/468) by [Jérémy Scanvic](https://github.com/jscanvic))
- Fixed sequency ordering in [`deepinv.physics.SinglePixelCamera`](https://deepinv.org/api/stubs/deepinv.physics.SinglePixelCamera.html.md#deepinv.physics.SinglePixelCamera) ([#475](https://github.com/deepinv/deepinv/pull/475) by [Brayan Monroy](https://github.com/bemc22))
- Change array operations from numpy to PyTorch in [`deepinv.physics.SinglePixelCamera`](https://deepinv.org/api/stubs/deepinv.physics.SinglePixelCamera.html.md#deepinv.physics.SinglePixelCamera) ([#483](https://github.com/deepinv/deepinv/pull/483) by [Jérémy Scanvic](https://github.com/jscanvic))
- Get rid of commented out code ([#485](https://github.com/deepinv/deepinv/pull/485) by [Jérémy Scanvic](https://github.com/jscanvic))
- Changed [`deepinv.physics.SinglePixelCamera`](https://deepinv.org/api/stubs/deepinv.physics.SinglePixelCamera.html.md#deepinv.physics.SinglePixelCamera) parameters in demos ([#493](https://github.com/deepinv/deepinv/pull/493) by [Brayan Monroy](https://github.com/bemc22))
- Improved code coverage by mocking datasets ([#490](https://github.com/deepinv/deepinv/pull/490) by [Jérémy Scanvic](https://github.com/jscanvic))
- Fix MRI mask generator update img_size on-the-fly not updating n_lines ([#416](https://github.com/deepinv/deepinv/pull/416) by [Andrew Wang](https://andrewwango.github.io/about/))
- Upgrade deprecated typing.T types in the code base ([#501](https://github.com/deepinv/deepinv/pull/501) by [Jérémy Scanvic](https://github.com/jscanvic))

## v0.3

### New Features

- Added early-stopping callback for [`deepinv.Trainer`](https://deepinv.org/api/stubs/deepinv.Trainer.html.md#deepinv.Trainer) and best model saving ([#437](https://github.com/deepinv/deepinv/pull/437) by [Julian Tachella](https://github.com/tachella) and [Andrew Wang](https://andrewwango.github.io/about/))
- Add various generators for the physics module (downsampling, variable masks for inpainting, PoissonGaussian generators etc) ([#384](https://github.com/deepinv/deepinv/pull/384) by [Matthieu Terris](https://github.com/matthieutrs))
- Add minres least squared solver ([#425](https://github.com/deepinv/deepinv/pull/425) by [Sebastian Neumayer](https://www.tu-chemnitz.de/mathematik/invimg/index.en.php) and [Johannes Hertrich](https://johertrich.github.io/))
- New least squared solvers (BiCGStab & LSQR) ([#393](https://github.com/deepinv/deepinv/pull/393) by [Julian Tachella](https://github.com/tachella))
- Typehints are used automatically in the documentation ([#379](https://github.com/deepinv/deepinv/pull/379) by [Julian Tachella](https://github.com/tachella))
- Add Ptychography operator in physics.phase_retrieval ([#351](https://github.com/deepinv/deepinv/pull/351) by [Victor Sechaud](https://github.com/vsechaud))
- Multispectral: NBU satellite image dataset, ERGAS+SAM metrics, PanNet, generalised pansharpening and decolorize ([#371](https://github.com/deepinv/deepinv/pull/371) by [Julian Tachella](https://github.com/tachella) and [Andrew Wang](https://andrewwango.github.io/about/))
- StackedPhysics: class definition, loss and data-fidelity ([#371](https://github.com/deepinv/deepinv/pull/371) by [Julian Tachella](https://github.com/tachella) and [Andrew Wang](https://andrewwango.github.io/about/))
- Added HyperSpectral Unmixing operator ([#353](https://github.com/deepinv/deepinv/pull/353) by Dongdong Chen and [Andrew Wang](https://andrewwango.github.io/about/))
- Add CASSI operator ([#377](https://github.com/deepinv/deepinv/pull/377) by [Andrew Wang](https://andrewwango.github.io/about/))
- Add validation dataset to data generator ([#363](https://github.com/deepinv/deepinv/pull/363) by [Andrew Wang](https://andrewwango.github.io/about/))
- Add Rescale and ToComplex torchvision-style transforms ([#363](https://github.com/deepinv/deepinv/pull/363) by [Andrew Wang](https://andrewwango.github.io/about/))
- Add SimpleFastMRISliceDataset, simplify FastMRISliceDataset, add FastMRI tests ([#363](https://github.com/deepinv/deepinv/pull/363) by [Andrew Wang](https://andrewwango.github.io/about/))
- FastMRI now compatible with MRI and MultiCoilMRI physics ([#363](https://github.com/deepinv/deepinv/pull/363) by [Andrew Wang](https://andrewwango.github.io/about/))
- Add VarNet/E2E-VarNet model and generalise ArtifactRemoval ([#363](https://github.com/deepinv/deepinv/pull/363) by [Andrew Wang](https://andrewwango.github.io/about/))
- [`deepinv.Trainer`](https://deepinv.org/api/stubs/deepinv.Trainer.html.md#deepinv.Trainer) now can log train progress per batch or per epoch ([#388](https://github.com/deepinv/deepinv/pull/388) by [Andrew Wang](https://andrewwango.github.io/about/))
- CMRxRecon dataset and generalised dataset metadata caching ([#385](https://github.com/deepinv/deepinv/pull/385) by [Andrew Wang](https://andrewwango.github.io/about/))
- Online training with noisy physics now can repeat the same noise each epoch ([#414](https://github.com/deepinv/deepinv/pull/414) by [Andrew Wang](https://andrewwango.github.io/about/))
- [`deepinv.Trainer`](https://deepinv.org/api/stubs/deepinv.Trainer.html.md#deepinv.Trainer) test can return unaggregated metrics ([#420](https://github.com/deepinv/deepinv/pull/420) by [Andrew Wang](https://andrewwango.github.io/about/))
- MoDL model ([#435](https://github.com/deepinv/deepinv/pull/435) by [Andrew Wang](https://andrewwango.github.io/about/))
- Add conversion to Hounsfield Units (HUs) for LIDC IDRI ([#459](https://github.com/deepinv/deepinv/pull/459) by [Jérémy Scanvic](https://github.com/jscanvic))
- Add ComposedLinearPhysics (via \_\_mul_\_ method) ([#462](https://github.com/deepinv/deepinv/pull/462) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/) and [Julian Tachella](https://github.com/tachella) )
- Register physics-dependent parameters to module buffers ([#462](https://github.com/deepinv/deepinv/pull/462) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- Add example on optimizing physics parameters ([#462](https://github.com/deepinv/deepinv/pull/462) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- Add `device` property to [`deepinv.utils.TensorList`](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) ([#462](https://github.com/deepinv/deepinv/pull/462) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- Add test physics device transfer and differentiablity ([#462](https://github.com/deepinv/deepinv/pull/462) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))

### Fixed

- Fixed MRI noise bug in kernel of mask ([#384](https://github.com/deepinv/deepinv/pull/384) by [Matthieu Terris](https://github.com/matthieutrs))
- Support for multi-physics / multi-dataset during training fixed ([#384](https://github.com/deepinv/deepinv/pull/384) by [Matthieu Terris](https://github.com/matthieutrs))
- Fixed device bug ([#415](https://github.com/deepinv/deepinv/pull/415) by Dongdong Chen)
- Fixed hyperlinks throughout docs ([#379](https://github.com/deepinv/deepinv/pull/379) by [Julian Tachella](https://github.com/tachella))
- Missing sigma normalization in L2Denoiser ([#371](https://github.com/deepinv/deepinv/pull/371) by [Julian Tachella](https://github.com/tachella) and [Andrew Wang](https://andrewwango.github.io/about/))
- [`deepinv.Trainer`](https://deepinv.org/api/stubs/deepinv.Trainer.html.md#deepinv.Trainer) discards checkpoint after loading ([#385](https://github.com/deepinv/deepinv/pull/385) by [Andrew Wang](https://andrewwango.github.io/about/))
- Fix offline training with noise generator not updating noise params ([#414](https://github.com/deepinv/deepinv/pull/414) by [Andrew Wang](https://andrewwango.github.io/about/))
- Fix wrong reference link in auto examples ([#432](https://github.com/deepinv/deepinv/pull/432) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- Fix paths in LidcIdriSliceDataset ([#446](https://github.com/deepinv/deepinv/pull/446) by [Jérémy Scanvic](https://github.com/jscanvic))
- Fix device inconsistency in test_physics, physics classes and noise models ([#462](https://github.com/deepinv/deepinv/pull/462) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- Fix Ptychography can not handle multi-channels input ([#494](https://github.com/deepinv/deepinv/pull/494) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- Fix argument name (img_size, in_shape, …) inconsistency  ([#494](https://github.com/deepinv/deepinv/pull/494) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))

### Changed

- Add bibtex references ([#575](https://github.com/deepinv/deepinv/pull/575) by [Samuel Hurault](https://github.com/samuro95))
- Set sphinx warnings as errors ([#379](https://github.com/deepinv/deepinv/pull/379) by [Julian Tachella](https://github.com/tachella))
- Added single backquotes default to code mode in docs ([#379](https://github.com/deepinv/deepinv/pull/379) by [Julian Tachella](https://github.com/tachella))
- Changed the \_\_add_\_ method for stack method for stacking physics ([#371](https://github.com/deepinv/deepinv/pull/371) by [Julian Tachella](https://github.com/tachella) and [Andrew Wang](https://andrewwango.github.io/about/))
- Changed the R2R loss to handle multiple noise distributions ([#380](https://github.com/deepinv/deepinv/pull/380) by [Brayan Monroy](https://github.com/bemc22))
- [`deepinv.Trainer.get_samples_online()`](https://deepinv.org/api/stubs/deepinv.Trainer.html.md#deepinv.Trainer.get_samples_online) using physics generator now updates physics params via both [`deepinv.physics.Physics.update_parameters()`](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics.update_parameters) and forward pass ([#386](https://github.com/deepinv/deepinv/pull/386) by [Andrew Wang](https://andrewwango.github.io/about/))
- Deprecate [`deepinv.Trainer`](https://deepinv.org/api/stubs/deepinv.Trainer.html.md#deepinv.Trainer) `freq_plot` in favour of plot_interval ([#388](https://github.com/deepinv/deepinv/pull/388) by [Andrew Wang](https://andrewwango.github.io/about/))

## v0.2.2

### New Features

- Added NCNSpp, ADMUNet model and pretrained weights (by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- Added SDE class (DiffusionSDE (OU Process), VESDE) for image generation (by [Minh Hai Nguyen](https://mh-nguyen712.github.io/) and [Samuel Hurault](https://github.com/samuro95))
- Added SDE solvers (Euler, Heun) (by [Minh Hai Nguyen](https://mh-nguyen712.github.io/) and [Samuel Hurault](https://github.com/samuro95))
- Added example on image generation, working for [`deepinv.models.NCSNpp`](https://deepinv.org/api/stubs/deepinv.models.NCSNpp.html.md#deepinv.models.NCSNpp), [`deepinv.models.ADMUNet`](https://deepinv.org/api/stubs/deepinv.models.ADMUNet.html.md#deepinv.models.ADMUNet), [`deepinv.models.DRUNet`](https://deepinv.org/api/stubs/deepinv.models.DRUNet.html.md#deepinv.models.DRUNet) and [`deepinv.models.DiffUNet`](https://deepinv.org/api/stubs/deepinv.models.DiffUNet.html.md#deepinv.models.DiffUNet) (by [Minh Hai Nguyen](https://mh-nguyen712.github.io/) and [Matthieu Terris](https://github.com/matthieutrs))
- Added VP-SDE for image generation and posterior sampling ([#434](https://github.com/deepinv/deepinv/pull/434) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- global path for datasets `deepinv.utils.get_data_home` ([#347](https://github.com/deepinv/deepinv/pull/347) by [Julian Tachella](https://github.com/tachella) and [Thomas Moreau](https://github.com/tomMoral))
- New docs user guide ([#347](https://github.com/deepinv/deepinv/pull/347) by [Julian Tachella](https://github.com/tachella) and [Thomas Moreau](https://github.com/tomMoral))
- Added UNSURE loss ([#313](https://github.com/deepinv/deepinv/pull/313) by [Julian Tachella](https://github.com/tachella))
- Add transform symmetrisation, further transform arithmetic, and new equivariant denoiser ([#259](https://github.com/deepinv/deepinv/pull/259) by [Andrew Wang](https://andrewwango.github.io/about/))
- New transforms: multi-axis reflect, time-shift and diffeomorphism ([#259](https://github.com/deepinv/deepinv/pull/259) by [Andrew Wang](https://andrewwango.github.io/about/))
- Add wrapper classes for adapting models to take time-sequence 2D+t input ([#296](https://github.com/deepinv/deepinv/pull/296) by [Andrew Wang](https://andrewwango.github.io/about/))
- Add sequential MRI operator ([#296](https://github.com/deepinv/deepinv/pull/296) by [Andrew Wang](https://andrewwango.github.io/about/))
- Add multi-operator equivariant imaging loss ([#296](https://github.com/deepinv/deepinv/pull/296) by [Andrew Wang](https://andrewwango.github.io/about/))
- Add loss schedulers ([#296](https://github.com/deepinv/deepinv/pull/296) by [Andrew Wang](https://andrewwango.github.io/about/))
- Add transform symmetrisation, further transform arithmetic, and new equivariant denoiser ([#259](https://github.com/deepinv/deepinv/pull/259) by [Andrew Wang](https://andrewwango.github.io/about/))
- New transforms: multi-axis reflect, time-shift and diffeomorphism ([#259](https://github.com/deepinv/deepinv/pull/259) by [Andrew Wang](https://andrewwango.github.io/about/))
- Multi-coil MRI, 3D MRI, MRI Mixin ([#287](https://github.com/deepinv/deepinv/pull/287) by [Andrew Wang](https://andrewwango.github.io/about/), Brett Levac)
- Add Metric baseclass, unified params (for complex, norm, reduce), typing, tests, L1L2 metric, QNR metric, metrics docs section, Metric functional wrapper ([#309](https://github.com/deepinv/deepinv/pull/309), [#343](https://github.com/deepinv/deepinv/pull/343) by [Andrew Wang](https://andrewwango.github.io/about/))
- generate_dataset features: complex numbers, save/load physics_generator params, overwrite bool ([#324](https://github.com/deepinv/deepinv/pull/324), [#352](https://github.com/deepinv/deepinv/pull/352) by [Andrew Wang](https://andrewwango.github.io/about/))
- Add the Köhler dataset ([#271](https://github.com/deepinv/deepinv/pull/271) by [Jérémy Scanvic](https://github.com/jscanvic))

### Fixed

- Fixed sphinx warnings ([#347](https://github.com/deepinv/deepinv/pull/347) by [Julian Tachella](https://github.com/tachella) and [Thomas Moreau](https://github.com/tomMoral))
- Fix cache file initialization in FastMRI Dataloader ([#300](https://github.com/deepinv/deepinv/pull/300) by [Pierre-Antoine Comby](https://github.com/paquiteau))
- Fixed prox_l2 no learning option in [`deepinv.Trainer`](https://deepinv.org/api/stubs/deepinv.Trainer.html.md#deepinv.Trainer) ([#304](https://github.com/deepinv/deepinv/pull/304) by [Julian Tachella](https://github.com/tachella))
- Fixed SSIM to use lightweight torchmetrics function + add MSE and NMSE as metrics + allow PSNR & SSIM to set max pixel on the fly ([#296](https://github.com/deepinv/deepinv/pull/296) by [Andrew Wang](https://andrewwango.github.io/about/))
- Fix generate_dataset error with physics_generator and batch_size != 1. ([#315](https://github.com/deepinv/deepinv/pull/315) by apolychronou)
- Fix generate_dataset error not using random physics generator ([#324](https://github.com/deepinv/deepinv/pull/324) by [Andrew Wang](https://andrewwango.github.io/about/))
- Fix Scale transform rng device error ([#324](https://github.com/deepinv/deepinv/pull/324) by [Andrew Wang](https://andrewwango.github.io/about/))
- Fix bug when using cuda device in dinv.datasets.generate_dataset  ([#334](https://github.com/deepinv/deepinv/pull/334) by [Tobias Liaudat](https://github.com/tobias-liaudat))
- Update outdated links in the readme ([#366](https://github.com/deepinv/deepinv/pull/366) by [Jérémy Scanvic](https://github.com/jscanvic))

### Changed

- Added direct option to ArtifactRemoval ([#347](https://github.com/deepinv/deepinv/pull/347) by [Julian Tachella](https://github.com/tachella) and [Thomas Moreau](https://github.com/tomMoral))
- Sphinx template to pydata ([#347](https://github.com/deepinv/deepinv/pull/347) by [Julian Tachella](https://github.com/tachella) and [Thomas Moreau](https://github.com/tomMoral))
- Remove metrics from utils and consolidate complex and normalisation options ([#309](https://github.com/deepinv/deepinv/pull/309) by [Andrew Wang](https://andrewwango.github.io/about/))
- get_freer_gpu falls back to torch.cuda when nvidia-smi fails ([#352](https://github.com/deepinv/deepinv/pull/352) by [Andrew Wang](https://andrewwango.github.io/about/))
- libcpab now is a PyPi package for diffeomorphisms, add rngs and devices to transforms ([#370](https://github.com/deepinv/deepinv/pull/370) by [Andrew Wang](https://andrewwango.github.io/about/))

## v0.2.1

### New Features

- Mirror Descent algorithm with Bregman potentials ([#282](https://github.com/deepinv/deepinv/pull/282) by [Samuel Hurault](https://github.com/samuro95))
- Added Gaussian-weighted splitting mask (from Yaman et al.), Artifact2Artifact (Liu et al.) and Phase2Phase (Eldeniz et al.) ([#279](https://github.com/deepinv/deepinv/pull/279) by [Andrew Wang](https://andrewwango.github.io/about/))
- Added time-agnostic network wrapper ([#279](https://github.com/deepinv/deepinv/pull/279) by [Andrew Wang](https://andrewwango.github.io/about/))
- Add sinc filter ([#280](https://github.com/deepinv/deepinv/pull/280) by [Julian Tachella](https://github.com/tachella))
- Add Noise2Score method ([#280](https://github.com/deepinv/deepinv/pull/280) by [Julian Tachella](https://github.com/tachella))
- Add Gamma Noise ([#280](https://github.com/deepinv/deepinv/pull/280) by [Julian Tachella](https://github.com/tachella))
- Add 3D Blur physics operator, with 3D diffraction microscope blur generators ([#277](https://github.com/deepinv/deepinv/pull/277) by [Florian Sarron](https://fsarron.github.io/), [Pierre Weiss](https://www.math.univ-toulouse.fr/~weiss/), `Paul Escande`, [Minh Hai Nguyen](https://mh-nguyen712.github.io/)) - 12/07/2024
- Add ICNN model ([#281](https://github.com/deepinv/deepinv/pull/281) by [Samuel Hurault](https://github.com/samuro95))
- Dynamic MRI physics operator ([#242](https://github.com/deepinv/deepinv/pull/242) by [Andrew Wang](https://andrewwango.github.io/about/))
- Add support for adversarial losses and models (GANs) ([#183](https://github.com/deepinv/deepinv/pull/183) by [Andrew Wang](https://andrewwango.github.io/about/))
- Base transform class for transform arithmetic ([#240](https://github.com/deepinv/deepinv/pull/240) by [Andrew Wang](https://andrewwango.github.io/about/)) - 26/06/2024.
- Plot video/animation functionality ([#245](https://github.com/deepinv/deepinv/pull/245) by [Andrew Wang](https://andrewwango.github.io/about/))
- Added update_parameters for parameter-dependent physics ([#241](https://github.com/deepinv/deepinv/pull/241) by Julian Tachella) - 11/06/2024
- Added evaluation functions for R2R and Splitting losses ([#241](https://github.com/deepinv/deepinv/pull/241) by Julian Tachella) - 11/06/2024
- Added a new `Physics` class for the Radio Interferometry problem ([#230](https://github.com/deepinv/deepinv/pull/230) by [Chao Tang](https://github.com/ChaoTang0330), [Tobias Liaudat](https://github.com/tobias-liaudat)) - 07/06/2024
- Add projective and affine transformations for EI or data augmentation ([#173](https://github.com/deepinv/deepinv/pull/173) by [Andrew Wang](https://andrewwango.github.io/about/))
- Add k-t MRI mask generators using Gaussian, random uniform and equispaced sampling stratgies ([#206](https://github.com/deepinv/deepinv/pull/206) by [Andrew Wang](https://andrewwango.github.io/about/))
- Added Lidc-Idri buit-in datasets ([#270](https://github.com/deepinv/deepinv/pull/270) by Maxime SONG) - 12/07/2024
- Added Flickr2k / LSDIR / Fluorescent Microscopy Denoising  buit-in datasets ([#276](https://github.com/deepinv/deepinv/pull/276) by Maxime SONG) - 15/07/2024
- Added `rng` a random number generator to each [`deepinv.physics.generator.PhysicsGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.PhysicsGenerator.html.md#deepinv.physics.generator.PhysicsGenerator) and a `seed` number argument to [`deepinv.physics.generator.PhysicsGenerator.step()`](https://deepinv.org/api/stubs/deepinv.physics.generator.PhysicsGenerator.html.md#deepinv.physics.generator.PhysicsGenerator.step) (by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))
- Added an equivalent of [`deepinv.physics.functional.random_choice()`](https://deepinv.org/api/stubs/deepinv.physics.functional.random_choice.html.md#deepinv.physics.functional.random_choice) in torch, available as [`deepinv.physics.functional.random_choice()`](https://deepinv.org/api/stubs/deepinv.physics.functional.random_choice.html.md#deepinv.physics.functional.random_choice) (by [Minh Hai Nguyen](https://mh-nguyen712.github.io/))

### Fixed

- Disable unecessary gradient computation to prevent memory explosion ([#301](https://github.com/deepinv/deepinv/pull/301) by `Dylan Sechet`, `Samuel Hurault`)
- Wandb logging ([#280](https://github.com/deepinv/deepinv/pull/280) by [Julian Tachella](https://github.com/tachella))
- SURE improvements ([#280](https://github.com/deepinv/deepinv/pull/280) by [Julian Tachella](https://github.com/tachella))
- Fixed padding in conv_transpose2d and made conv_2d a true convolution (by [Florian Sarron](https://fsarron.github.io/), [Pierre Weiss](https://www.math.univ-toulouse.fr/~weiss/), Paul Escande, [Minh Hai Nguyen](https://mh-nguyen712.github.io/)) - 12/07/2024
- Fixed the gradient stopping in EILoss ([#263](https://github.com/deepinv/deepinv/pull/263) by [Jérémy Scanvic](https://github.com/jscanvic)) - 27/06/2024
- Fixed averaging loss over epochs [`deepinv.Trainer`](https://deepinv.org/api/stubs/deepinv.Trainer.html.md#deepinv.Trainer) ([#241](https://github.com/deepinv/deepinv/pull/241) by Julian Tachella) - 11/06/2024
- Fixed [`deepinv.Trainer`](https://deepinv.org/api/stubs/deepinv.Trainer.html.md#deepinv.Trainer) save_path timestamp problem on Windows ([#245](https://github.com/deepinv/deepinv/pull/245) by [Andrew Wang](https://andrewwango.github.io/about/))
- Fixed inpainting/SplittingLoss mask generation + more flexible tensor size handling + pixelwise masking ([#267](https://github.com/deepinv/deepinv/pull/267) by [Andrew Wang](https://andrewwango.github.io/about/))
- Fixed the [`deepinv.physics.generator.ProductConvolutionBlurGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.ProductConvolutionBlurGenerator.html.md#deepinv.physics.generator.ProductConvolutionBlurGenerator), allowing for batch generation (previously does not work) by ([Minh Hai Nguyen](https://mh-nguyen712.github.io/))

### Changed

- Redefine Prior, DataFidelity and Bregman with a common parent class Potential ([#282](https://github.com/deepinv/deepinv/pull/282) by [Samuel Hurault](https://github.com/samuro95))
- Changed to Python 3.9+ ([#280](https://github.com/deepinv/deepinv/pull/280) by [Julian Tachella](https://github.com/tachella))
- Improved support for parameter-dependent operators ([#227](https://github.com/deepinv/deepinv/pull/227) by [Jérémy Scanvic](https://github.com/jscanvic)) - 28/05/2024
- Added a divergence check in the conjugate gradient implementation ([#225](https://github.com/deepinv/deepinv/pull/225) by [Jérémy Scanvic](https://github.com/jscanvic)) - 22/05/2024

## v0.2.0

Many of the features in this version were developed by [Minh Hai Nguyen](https://mh-nguyen712.github.io/),
[Pierre Weiss](https://www.math.univ-toulouse.fr/~weiss/), [Florian Sarron](https://fsarron.github.io/), [Julian Tachella](https://github.com/tachella) and [Matthieu Terris](https://github.com/matthieutrs) during the IDRIS hackathon.

### New Features

- Added a parameterization of the operators and noiselevels for the physics class
- Added a physics.functional submodule
- Modified the Blur class to handle color, grayscale, single and multi-batch images
- Added a PhysicsGenerator class to synthetize parameters for the forward operators
- Added the possibility to sum generators
- Added a MotionBlur generator
- Added a DiffractionBlur generator
- Added a MaskGenerator for MRI
- Added a SigmaGenerator for the Gaussian noise
- Added a tour of blur operators
- Added ProductConvolution expansions
- Added a ThinPlateSpline interpolation function
- Added d-dimensional histograms
- Added GeneratorMixture to mix physics generators
- Added the SpaceVarying blur class
- Added the SpaceVarying blur generators
- Added pytests and examples for all the new features
- A few speed ups by carefully profiling the training codes
- made sigma in drunet trainable
- Added [`deepinv.Trainer`](https://deepinv.org/api/stubs/deepinv.Trainer.html.md#deepinv.Trainer), [`deepinv.loss.Loss`](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss) class and eval metric (LPIPS, NIQE, SSIM) ([#181](https://github.com/deepinv/deepinv/pull/181) by [Julian Tachella](https://github.com/tachella)) - 02/04/2024
- PhaseRetrieval class ([#176](https://github.com/deepinv/deepinv/pull/176) by [Zhiyuan Hu](https://github.com/zhiyhu1605)) - 20/03/2024
- Added 3D wavelets ([#164](https://github.com/deepinv/deepinv/pull/164) by [Matthieu Terris](https://github.com/matthieutrs)) - 07/03/2024
- Added patch priors loss ([#164](https://github.com/deepinv/deepinv/pull/164) by [Johannes Hertrich](https://johertrich.github.io/)) - 07/03/2024
- Added Restormer model ([#185](https://github.com/deepinv/deepinv/pull/185) by Antoine Regnier and Maxime SONG) - 18/04/2024
- Added DIV2K built-in dataset ([#203](https://github.com/deepinv/deepinv/pull/203) by Maxime SONG) - 03/05/2024
- Added Urban100 built-in dataset ([#237](https://github.com/deepinv/deepinv/pull/237) by Maxime SONG) - 07/06/2024
- Added Set14 / CBSD68 / fastMRI buit-in datasets ([#248](https://github.com/deepinv/deepinv/pull/248) [#249](https://github.com/deepinv/deepinv/pull/249) [#229](https://github.com/deepinv/deepinv/pull/229) by Maxime SONG) - 25/06/2024

### Fixed

- Fixed the None prior ([#233](https://github.com/deepinv/deepinv/pull/233) by [Samuel Hurault](https://github.com/samuro95)) - 04/06/2024
- Fixed the conjugate gradient torch.nograd for teh demos, accelerated)
- Fixed torch.nograd in demos for faster generation of the doc
- Corrected the padding for the convolution
- Solved pan-sharpening issues
- Many docstring fixes
- Fixed slow drunet sigma and batched conjugate gradient  ([#181](https://github.com/deepinv/deepinv/pull/181) by [Minh Hai Nguyen](https://mh-nguyen712.github.io/)) - 02/04/2024
- Fixed g dependence on sigma in optim docs ([#165](https://github.com/deepinv/deepinv/pull/165) by [Julian Tachella](https://github.com/tachella)) - 28/02/2024

### Changed

- Refactored the documentation completely for the physics
- Refactor unfolded docs ([#181](https://github.com/deepinv/deepinv/pull/181) by [Julian Tachella](https://github.com/tachella)) - 02/04/2024
- Refactor model docs ([#172](https://github.com/deepinv/deepinv/pull/172) by [Julian Tachella](https://github.com/tachella)) - 12/03/2024
- Changed WaveletPrior to WaveletDenoiser ([#165](https://github.com/deepinv/deepinv/pull/165) by [Julian Tachella](https://github.com/tachella)) - 28/02/2024
- Move from torchwavelets to ptwt ([#162](https://github.com/deepinv/deepinv/pull/162) by [Matthieu Terris](https://github.com/matthieutrs)) - 22/02/2024

## v0.1.1

### New Features

- Added r2r loss ([#148](https://github.com/deepinv/deepinv/pull/148) by [Brayan Monroy](https://github.com/bemc22)) - 30/01/2024
- Added scale transform ([#135](https://github.com/deepinv/deepinv/pull/135) by [Jérémy Scanvic](https://github.com/jscanvic)) - 19/12/2023
- Added priors for total variation and l12 mixed norm ([#156](https://github.com/deepinv/deepinv/pull/156) by [Nils Laurent](https://nils-laurent.github.io/)) - 09/02/2023

### Fixed

- Fixed issue in noise forward of Decomposable class ([#154](https://github.com/deepinv/deepinv/pull/154) by [Matthieu Terris](https://github.com/matthieutrs)) - 08/02/2024
- Fixed new black version 24.1.1 style changes ([#151](https://github.com/deepinv/deepinv/pull/151) by [Julian Tachella](https://github.com/tachella)) - 31/01/2024
- Fixed test for sigma as torch tensor with gpu enable ([#145](https://github.com/deepinv/deepinv/pull/145) by [Brayan Monroy](https://github.com/bemc22)) - 23/12/2023
- Fixed [#139](https://github.com/deepinv/deepinv/pull/139) BM3D tensor format grayscale ([#140](https://github.com/deepinv/deepinv/pull/140) by [Matthieu Terris](https://github.com/matthieutrs)) - 23/12/2023
- Fixed [#136](https://github.com/deepinv/deepinv/pull/136) noise additive model for DecomposablePhysics ([#138](https://github.com/deepinv/deepinv/pull/138) by [Matthieu Terris](https://github.com/matthieutrs)) - 22/12/2023
- Importing `deepinv` does not modify matplotlib config anymore (:gh\`1501\` by [Thomas Moreau](https://github.com/tomMoral)) - 30/01/2024

### Changed

- Rephrased the README ([#142](https://github.com/deepinv/deepinv/pull/142) by [Jérémy Scanvic](https://github.com/jscanvic)) - 09/01/2024

## v0.1.0

### New Features

- Added autoadjoint capabilities ([#151](https://github.com/deepinv/deepinv/pull/151) by [Julian Tachella](https://github.com/tachella)) - 31/01/2024
- Added equivariant transforms ([#125](https://github.com/deepinv/deepinv/pull/125) by [Matthieu Terris](https://github.com/matthieutrs)) - 07/12/2023
- Moved datasets and weights to HuggingFace ([#121](https://github.com/deepinv/deepinv/pull/121) by [Samuel Hurault](https://github.com/samuro95)) - 01/12/2023
- Added L1 prior, change distance in DataFidelity ([#108](https://github.com/deepinv/deepinv/pull/108) by [Samuel Hurault](https://github.com/samuro95)) - 03/11/2023
- Added Kaiming init ([#102](https://github.com/deepinv/deepinv/pull/102) by [Matthieu Terris](https://github.com/matthieutrs)) - 29/10/2023
- Added Anderson Acceleration ([#86](https://github.com/deepinv/deepinv/pull/86) by [Samuel Hurault](https://github.com/samuro95)) - 23/10/2023
- Added [`deepinv.sampling.DPS()`](https://deepinv.org/api/stubs/deepinv.sampling.DPS.html.md#deepinv.sampling.DPS) diffusion method ([#92](https://github.com/deepinv/deepinv/pull/92) by [Julian Tachella](https://github.com/tachella) and [Hyungjin Chung](https://www.hj-chung.com/)) - 20/10/2023
- Added on-the-fly physics computations in training ([#88](https://github.com/deepinv/deepinv/pull/88) by [Matthieu Terris](https://github.com/matthieutrs)) - 10/10/2023
- Added `no_grad` parameter ([#80](https://github.com/deepinv/deepinv/pull/80) by [Jérémy Scanvic](https://github.com/jscanvic)) - 20/08/2023
- Added prox of TV ([#79](https://github.com/deepinv/deepinv/pull/79) by [Matthieu Terris](https://github.com/matthieutrs)) - 16/08/2023
- Added diffpir demo + model ([#77](https://github.com/deepinv/deepinv/pull/77) by [Matthieu Terris](https://github.com/matthieutrs)) - 08/08/2023
- Added SwinIR model ([#76](https://github.com/deepinv/deepinv/pull/76) by [Jérémy Scanvic](https://github.com/jscanvic)) - 02/08/2023
- Added hard-threshold ([#71](https://github.com/deepinv/deepinv/pull/71) by [Matthieu Terris](https://github.com/matthieutrs)) - 18/07/2023
- Added discord server ([#64](https://github.com/deepinv/deepinv/pull/64) by [Julian Tachella](https://github.com/tachella)) - 10/07/2023
- Added changelog file ([#64](https://github.com/deepinv/deepinv/pull/64) by [Julian Tachella](https://github.com/tachella)) - 10/07/2023

### Fixed

- doc fixes + training fixes ([#124](https://github.com/deepinv/deepinv/pull/124) by [Julian Tachella](https://github.com/tachella)) - 06/12/2023
- Add doc weights ([#97](https://github.com/deepinv/deepinv/pull/97) by [Matthieu Terris](https://github.com/matthieutrs)) - 24/10/2023
- Fix BlurFFT adjoint ([#89](https://github.com/deepinv/deepinv/pull/89) by [Matthieu Terris](https://github.com/matthieutrs)) - 15/10/2023
- Doc typos ([#88](https://github.com/deepinv/deepinv/pull/88) by [Matthieu Terris](https://github.com/matthieutrs)) - 10/10/2023
- Minor fixes DiffPIR + other typos ([#81](https://github.com/deepinv/deepinv/pull/81) by [Matthieu Terris](https://github.com/matthieutrs)) - 10/09/2023
- Call `wandb.init` only when needed ([#78](https://github.com/deepinv/deepinv/pull/78) by [Jérémy Scanvic](https://github.com/jscanvic)) - 09/08/2023
- Log epoch loss instead of batch loss ([#73](https://github.com/deepinv/deepinv/pull/73) by [Jérémy Scanvic](https://github.com/jscanvic)) - 21/07/2023
- Automatically disable backtracking is no explicit cost ([#68](https://github.com/deepinv/deepinv/pull/68) by [Samuel Hurault](https://github.com/samuro95)) - 12/07/2023
- Added missing indent ([#63](https://github.com/deepinv/deepinv/pull/63) by [Jérémy Scanvic](https://github.com/jscanvic)) - 12/07/2023
- Fixed get_freer_gpu grep statement to work for different versions of nvidia-smi ([#82](https://github.com/deepinv/deepinv/pull/82) by [Alexander Mehta](https://github.com/alexmehta)) - 20/09/2023
- Fixed get_freer_gpu to work on different operating systems ([#87](https://github.com/deepinv/deepinv/pull/87) by [Andrea Sebastiani](https://github.com/sedaboni)) - 10/10/2023
- Fixed Discord server and contributiong links  ([#87](https://github.com/deepinv/deepinv/pull/87) by [Andrea Sebastiani](https://github.com/sedaboni)) - 10/10/2023

### Changed

- Update CI ([#95](https://github.com/deepinv/deepinv/pull/95) [#99](https://github.com/deepinv/deepinv/pull/99) by [Thomas Moreau](https://github.com/tomMoral)) - 24/10/2023
- Changed normalization CS and SPC to 1/m ([#72](https://github.com/deepinv/deepinv/pull/72) by [Julian Tachella](https://github.com/tachella)) - 21/07/2023
- Update docstring ([#68](https://github.com/deepinv/deepinv/pull/68) by [Samuel Hurault](https://github.com/samuro95)) - 12/07/2023
