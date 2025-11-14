Change Log
=================
This change log is for the `main` branch. It contains changes for each release, with the date and author of each change.


Current
-------

New Features
^^^^^^^^^^^^

Changed
^^^^^^^

Fixed
^^^^^


v0.3.6
------
New Features
^^^^^^^^^^^^
- Add dataset for patch sampling from (large) nD images without loading entire images into memory (:gh:`806` by `Vicky De Ridder`_)
- Add support for 3D CNNs (:gh:`869` by `Vicky De Ridder`)
- Add support for complex dtypes in WaveletDenoiser, WaveletDictDenoiser and WaveletPrior (:gh:`738` by `Chaithya G R`_)
- dinv.io functions for loading DICOM, NIFTI, COS, GEOTIFF etc. (:gh:`768` by `Andrew Wang`_)

Changed
^^^^^^^
- load_np_url now returns tensors, load_url helper function moved to io (:gh:`768` by `Andrew Wang`_)
- utils/signal.py renamed to signals.py to avoid stdlib conflict (:gh:`768` by `Andrew Wang`_)
- utils.get_data_home now creates folder if not exist (:gh:`768` by `Andrew Wang`_)

Fixed
^^^^^
- Blur physics objects now put new filters to physics device regardless of input filter device (:gh:`844` by `Vicky De Ridder`_)
- Set14HR dataset now downloads from a different source (and has slightly different folderstructure), since old link broke. (:gh:`845` by `Vicky De Ridder`_)
- LsdirHR dataset now downloads from a different source (since old link broke) and correctly contains the specific folder images, instead of everything in root (:gh:`866` by `Vicky De Ridder`_)
- Fix typo in docstring of ItohFidelity and SpatialUnwrapping demo (:gh:`860` by `Brayan Monroy`_)
- Fix unhandled import error in CBSD68 if datasets is not installed (:gh:`868` by `Johannes Hertrich`_)
- Add support for complex signals in PSNR (:gh:`738` by `Jérémy Scanvic`_)
- Add a warning in SwinIR when upsampling parameters are inconsistent (:gh:`909` by `Jérémy Scanvic`_)



v0.3.5
------
New Features
^^^^^^^^^^^^
- Add statistics for SAR imaging + fix variance of :class:`deepinv.physics.GammaNoise` in doc (:gh:`740` by `Louise Friot Giroux`_)
- Add imshow kwargs to plot (:gh:`791` by `Andrew Wang`_)
- Add :class:`deepinv.physics.RicianNoise` model (:gh:`805` by `Vicky De Ridder`_)
- Add manual physics to reduced resolution loss (:gh:`808` by `Andrew Wang`_)
- Multi-coil MRI coil-map estimation acceleration via CuPy (:gh:`781` by `Andrew Wang`_)
- Add SpatialUnwrapping forward model and ItohFidelity data fidelity (:gh:`723` by `Brayan Monroy`_)

Changed
^^^^^^^
- (Breaking) Make :class:`deepinv.datasets.HDF5Dataset` similar to :class:`deepinv.Trainer` in the unsupervised setting by using NaNs for ground truths instead of a copy of the measurements (:gh:`761` by `Jérémy Scanvic`_)
- Add ``squared`` parameter to ``LinearPhysics.compute_norm()`` and ``compute_sqnorm()`` method (:gh:`832` by `Jérémy Scanvic`_)
- Allow self-supervised eval by removing the model.eval() from Trainer.train() (:gh:`777` by `Julian Tachella`_)
- Make tqdm progress bar auto-resize (:gh:`835` by `Andrew Wang`_)
- (Breaking) Normalize the Tomography operator with proper spectral norm computation. Set the default normalization behavior to ``True`` for both CT operators (:gh:`715` by `Romain Vo`_)

Fixed
^^^^^
- Use the learning-free model for learning-free metrics in :class:`deepinv.Trainer` (:gh:`788` by `Jérémy Scanvic`_)
- Fix device :func:`deepinv.utils.dirac_like` and :func:`deepinv.physics.blur.bilinear_filter`, :func:`deepinv.physics.blur.bicubic_filter` and :func:`deepinv.physics.blur.gaussian_blur` filters (:gh:`785` by `Julian Tachella`_)
- Fix positivity + batching gamma least squares solvers (:gh:`785` by `Julian Tachella`_ and `Minh Hai Nguyen`_)
- Fix and test :class:`deepinv.models.RAM` scaling issues (:gh:`785` by `Julian Tachella`_)
- Reduced CI python version tests (:gh:`746` by `Matthieu Terris`_)
- Fix scaling issue in :class:`deepinv.sampling.DiffusionSDE` (:gh:`772` by `Minh Hai Nguyen`_)
- All splitting losses fixed to work with changing image sizes and with multicoil MRI (:gh:`778` by `Andrew Wang`_)
- :class:`deepinv.Trainer` treats batch of nans as no ground truth (:gh:`793` by `Andrew Wang`_)
- Save training loss history (:gh:`777` by `Julian Tachella`_)
- Fix docstring formatting in BDSDS500 dataset (:gh:`816` by `Brayan Monroy`_)
- Remove unnecessary tensor cloning from DDRM and DPS (:gh:`834` by `Vicky De Ridder`_)
- Change deprecated `torch.norm` calls to `torch.linalg.vector_norm` (:gh:`840` by `Minh Hai Nguyen`_)


v0.3.4
------
New Features
^^^^^^^^^^^^
- Quickstart tutorials + clean examples (:gh:`622` by `Andrew Wang`_)
- Dataset base class + :class:`deepinv.datasets.ImageFolder` and :class:`deepinv.datasets.TensorDataset` classes (:gh:`622` by `Andrew Wang`_)
- Added GitHub action checking import time (:gh:`680` by `Julian Tachella`_)
- Client model for server-side inference for using models in the cloud (:gh:`691` by `Andrew Wang`_)
- Reduced resolution self-supervised loss (:gh:`735` by `Andrew Wang`_)
- Add :func:`deepinv.utils.disable_tex` to disable LaTeX (:gh:`726` by `Andrew Wang`_)
- Add BSDS500 dataset (:gh:`749` by `Johannes Hertrich`_ and `Sebastian Neumayer`_)
- O(1) memory backprop for linear solver and example (:gh:`739` by `Minh Hai Nguyen`_)

Changed
^^^^^^^
- Move mixins to utils and reduce number of cross-submodule top-level imports (:gh:`680` by `Andrew Wang`_)
- :class:`deepinv.datasets.PatchDataset` returns tensors and not tuples (:gh:`622` by `Andrew Wang`_)

Fixed
^^^^^
- Fixed natsorted issue (:gh:`680` by `Julian Tachella`_)
- Fix full-reference metrics used with measurement-only dataset (:gh:`622` by `Andrew Wang`_)
- Batching :class:`deepinv.physics.generator.DownsamplingGenerator` in the case of multiple filters (:gh:`690` by `Matthieu Terris`_)
- NaN motion blur generator (:gh:`685` by `Matthieu Terris`_)
- Fix the condition for break in compute_norm (:gh:`699` by `Quentin Barthélemy`_)
- Python 3.9 backward compatibility and zip_strict (:gh:`707` by `Andrew Wang`_)
- Fix numerical instability of :func:`deepinv.optim.utils.bicgstab` solver (:gh:`739` by `Minh Hai Nguyen`_)


v0.3.3
------
New Features
^^^^^^^^^^^^

- Automatic A_adjoint, U_adjoint and V computation for user-defined physics (:gh:`658` by `Julian Tachella`_)
- Add :class:`deepinv.models.RAM` model (:gh:`524` by `Matthieu Terris`_)
- FastMRI better raw data loading: load targets from different folder for test sets, load mask from test set, prewhitening, normalisation (:gh:`608` by `Andrew Wang`_)
- SKM-TEA raw MRI dataset (:gh:`608` by `Andrew Wang`_)
- New downsampling physics that matches MATLAB bicubic imresize (:gh:`608` by `Andrew Wang`_)

Changed
^^^^^^^
- Rename the normalizing function `deepinv.utils.rescale_img` to :func:`deepinv.utils.normalize_signal` (:gh:`641` by `Jérémy Scanvic`_)
- Changed default linear solver from `CG` to :func:`deepinv.optim.utils.lsqr` (:gh:`658` by `Julian Tachella`_)
- Added positive clipping by default and gain minimum in :class:`deepinv.physics.PoissonGaussianNoise` (:gh:`658` by `Julian Tachella`_).

Fixed
^^^^^

- Fix downsampling generator batching (:gh:`608` by `Andrew Wang`_)
- Fix memory leak in :mod:`deepinv.physics.Tomography` when using autograd (:gh:`651` by `Minh Hai Nguyen`_)
- Fix the circular padded :class:`deepinv.models.UNet` (:gh:`653` by `Victor Sechaud`_)
- Clamp constant signals in :func:`deepinv.utils.normalize_signal` to ensure they are normalized (:gh:`641` by `Jérémy Scanvic`_)
- Fix ZeroNoise default missing in :class:`deepinv.physics.ZeroNoise` (:gh:`658` by `Julian Tachella`_)



v0.3.2
------
New Features
^^^^^^^^^^^^
- Add support for astra-toolbox CT operators (parallel, fan, cone) with :class:`deepinv.physics.TomographyWithAstra` (:gh:`474` by `Romain Vo`_)
- Add :meth:`deepinv.physics.Physics.clone` (:gh:`534` by `Jérémy Scanvic`_)

Changed
^^^^^^^
- Make autograd use the base linear operator for :func:`deepinv.physics.adjoint_function` (:gh:`519` by `Jérémy Scanvic`_)
- Parallelize the test suite making it 15% faster (:gh:`522` by `Jérémy Scanvic`_)
- Adjust backward paths for tomography (:gh:`535` by `Johannes Hertrich`_)
- Update python version to 3.10+ (:gh:`605` by `Minh Hai Nguyen`_)
- Update the library dependencies, issue template, codecov report on linux only (:gh:`654` by `Minh Hai Nguyen`_)

Fixed
^^^^^
- Fix the total loss reported by the trainer (:gh:`515` by `Jérémy Scanvic`_)
- Fix the gradient norm reported by the trainer (:gh:`520` by `Jérémy Scanvic`_)
- Fix that the max_pixel option in PSNR and SSIM and add analgous min_pixel option (:gh:`535` by `Johannes Hertrich`_)
- Fix some issues related to denoisers: ICNN grad not working inside torch.no_grad(), batch of image and batch of sigma for some denoisers (DiffUNet, BM3D, TV, Wavemet), EPLL error when batch size > 1 (:gh:`530` by `Minh Hai Nguyen`_)
- Batching WaveletPrior and fix iwt (:gh:`530` by `Minh Hai Nguyen`_)
- Fix on unreliable/inconsistent automatic choosing GPU with most free VRAM (:gh:`570` by Fedor Goncharov)



v0.3.1
----------------

New Features
^^^^^^^^^^^^

- Added :class:`deepinv.physics.SaltPepperNoise` for impulse noise (:gh:`472` by `Thomas Moreau`_).
- Add measurement augmentation VORTEX loss (:gh:`410` by `Andrew Wang`_)
- Add non-geometric data augmentations (noise, phase errors) (:gh:`410` by `Andrew Wang`_)
- Make :class:`deepinv.physics.generator.PhysicsGenerator.average` use batches (:gh:`488` by `Jérémy Scanvic`_)
- MRI losses subclass, weighted-SSDU, Robust-SSDU loss functions + more mask generators (:gh:`416` by `Keying Guo`_ and `Andrew Wang`_)
- Multi-coil MRI estimates sens maps with sigpy ESPIRiT, MRISliceTransform better loads raw data by estimating coil maps and generating masks (:gh:`416` by `Andrew Wang`_)
- Add HaarPSI metric + metric standardization (:gh:`416` by `Andrew Wang`_)
- Add ENSURE loss (:gh:`454` by `Andrew Wang`_)

Changed
^^^^^^^
- Added cake_cutting, zig_zag and xy orderings in :class:`deepinv.physics.SinglePixelCamera` physics (:gh:`475` by `Brayan Monroy`_).

Fixed
^^^^^
- Fix images not showing in sphinx examples (:gh:`478` by `Matthieu Terris`_)
- Fix plot_inset not showing (:gh:`455` by `Andrew Wang`_)
- Fix latex rendering in :func:`deepinv.utils.plotting.config_matplotlib`  (:gh:`452` by `Romain Vo`_)
- Get rid of unnecessary file system writes in :func:`deepinv.utils.get_freer_gpu` (:gh:`468` by `Jérémy Scanvic`_)
- Fixed sequency ordering in :class:`deepinv.physics.SinglePixelCamera` (:gh:`475` by `Brayan Monroy`_)
- Change array operations from numpy to PyTorch in :class:`deepinv.physics.SinglePixelCamera` (:gh:`483` by `Jérémy Scanvic`_)
- Get rid of commented out code (:gh:`485` by `Jérémy Scanvic`_)
- Changed :class:`deepinv.physics.SinglePixelCamera` parameters in demos (:gh:`493` by `Brayan Monroy`_)
- Improved code coverage by mocking datasets (:gh:`490` by `Jérémy Scanvic`_)
- Fix MRI mask generator update img_size on-the-fly not updating n_lines (:gh:`416` by `Andrew Wang`_)
- Upgrade deprecated typing.T types in the code base (:gh:`501` by `Jérémy Scanvic`_)

v0.3
----------------

New Features
^^^^^^^^^^^^
- Added early-stopping callback for :class:`deepinv.Trainer` and best model saving (:gh:`437` by `Julian Tachella`_ and `Andrew Wang`_)
- Add various generators for the physics module (downsampling, variable masks for inpainting, PoissonGaussian generators etc) (:gh:`384` by `Matthieu Terris`_)
- Add minres least squared solver (:gh:`425` by `Sebastian Neumayer`_ and `Johannes Hertrich`_)
- New least squared solvers (BiCGStab & LSQR) (:gh:`393` by `Julian Tachella`_)
- Typehints are used automatically in the documentation (:gh:`379` by `Julian Tachella`_)
- Add Ptychography operator in physics.phase_retrieval (:gh:`351` by `Victor Sechaud`_)
- Multispectral: NBU satellite image dataset, ERGAS+SAM metrics, PanNet, generalised pansharpening and decolorize (:gh:`371` by `Julian Tachella`_ and `Andrew Wang`_)
- StackedPhysics: class definition, loss and data-fidelity (:gh:`371` by `Julian Tachella`_ and `Andrew Wang`_)
- Added HyperSpectral Unmixing operator (:gh:`353` by Dongdong Chen and `Andrew Wang`_)
- Add CASSI operator (:gh:`377` by `Andrew Wang`_)

- Add validation dataset to data generator (:gh:`363` by `Andrew Wang`_)
- Add Rescale and ToComplex torchvision-style transforms (:gh:`363` by `Andrew Wang`_)
- Add SimpleFastMRISliceDataset, simplify FastMRISliceDataset, add FastMRI tests (:gh:`363` by `Andrew Wang`_)
- FastMRI now compatible with MRI and MultiCoilMRI physics (:gh:`363` by `Andrew Wang`_)
- Add VarNet/E2E-VarNet model and generalise ArtifactRemoval (:gh:`363` by `Andrew Wang`_)
- :class:`deepinv.Trainer` now can log train progress per batch or per epoch (:gh:`388` by `Andrew Wang`_)
- CMRxRecon dataset and generalised dataset metadata caching (:gh:`385` by `Andrew Wang`_)
- Online training with noisy physics now can repeat the same noise each epoch (:gh:`414` by `Andrew Wang`_)
- :class:`deepinv.Trainer` test can return unaggregated metrics (:gh:`420` by `Andrew Wang`_)
- MoDL model (:gh:`435` by `Andrew Wang`_)
- Add conversion to Hounsfield Units (HUs) for LIDC IDRI (:gh:`459` by `Jérémy Scanvic`_)
- Add ComposedLinearPhysics (via __mul__ method) (:gh:`462` by `Minh Hai Nguyen`_ and `Julian Tachella`_ )
- Register physics-dependent parameters to module buffers (:gh:`462` by `Minh Hai Nguyen`_)
- Add example on optimizing physics parameters (:gh:`462` by `Minh Hai Nguyen`_)
- Add `device` property to :class:`deepinv.utils.TensorList` (:gh:`462` by `Minh Hai Nguyen`_)
- Add test physics device transfer and differentiablity (:gh:`462` by `Minh Hai Nguyen`_)

Fixed
^^^^^
- Fixed MRI noise bug in kernel of mask (:gh:`384` by `Matthieu Terris`_)
- Support for multi-physics / multi-dataset during training fixed (:gh:`384` by `Matthieu Terris`_)
- Fixed device bug (:gh:`415` by Dongdong Chen)
- Fixed hyperlinks throughout docs (:gh:`379` by `Julian Tachella`_)
- Missing sigma normalization in L2Denoiser (:gh:`371` by `Julian Tachella`_ and `Andrew Wang`_)
- :class:`deepinv.Trainer` discards checkpoint after loading (:gh:`385` by `Andrew Wang`_)
- Fix offline training with noise generator not updating noise params (:gh:`414` by `Andrew Wang`_)
- Fix wrong reference link in auto examples (:gh:`432` by `Minh Hai Nguyen`_)
- Fix paths in LidcIdriSliceDataset (:gh:`446` by `Jérémy Scanvic`_)
- Fix device inconsistency in test_physics, physics classes and noise models (:gh:`462` by `Minh Hai Nguyen`_)
- Fix Ptychography can not handle multi-channels input (:gh:`494` by `Minh Hai Nguyen`_)
- Fix argument name (img_size, in_shape, ...) inconsistency  (:gh:`494` by `Minh Hai Nguyen`_)

Changed
^^^^^^^
- Add bibtex references (:gh:`575` by `Samuel Hurault`_)
- Set sphinx warnings as errors (:gh:`379` by `Julian Tachella`_)
- Added single backquotes default to code mode in docs (:gh:`379` by `Julian Tachella`_)
- Changed the __add__ method for stack method for stacking physics (:gh:`371` by `Julian Tachella`_ and `Andrew Wang`_)
- Changed the R2R loss to handle multiple noise distributions (:gh:`380` by `Brayan Monroy`_)
- :func:`deepinv.Trainer.get_samples_online` using physics generator now updates physics params via both :meth:`deepinv.physics.Physics.update_parameters` and forward pass (:gh:`386` by `Andrew Wang`_)
- Deprecate :class:`deepinv.Trainer` `freq_plot` in favour of plot_interval (:gh:`388` by `Andrew Wang`_)

v0.2.2
----------------

New Features
^^^^^^^^^^^^
- Added NCNSpp, ADMUNet model and pretrained weights (by `Minh Hai Nguyen`_)
- Added SDE class (DiffusionSDE (OU Process), VESDE) for image generation (by `Minh Hai Nguyen`_ and `Samuel Hurault`_)
- Added SDE solvers (Euler, Heun) (by `Minh Hai Nguyen`_ and `Samuel Hurault`_)
- Added example on image generation, working for :class:`deepinv.models.NCSNpp`, :class:`deepinv.models.ADMUNet`, :class:`deepinv.models.DRUNet` and :class:`deepinv.models.DiffUNet` (by `Minh Hai Nguyen`_ and `Matthieu Terris`_)
- Added VP-SDE for image generation and posterior sampling (:gh:`434` by `Minh Hai Nguyen`_)

- global path for datasets :func:`deepinv.utils.get_data_home` (:gh:`347` by `Julian Tachella`_ and `Thomas Moreau`_)
- New docs user guide (:gh:`347` by `Julian Tachella`_ and `Thomas Moreau`_)
- Added UNSURE loss (:gh:`313` by `Julian Tachella`_)
- Add transform symmetrisation, further transform arithmetic, and new equivariant denoiser (:gh:`259` by `Andrew Wang`_)
- New transforms: multi-axis reflect, time-shift and diffeomorphism (:gh:`259` by `Andrew Wang`_)


- Add wrapper classes for adapting models to take time-sequence 2D+t input (:gh:`296` by `Andrew Wang`_)
- Add sequential MRI operator (:gh:`296` by `Andrew Wang`_)
- Add multi-operator equivariant imaging loss (:gh:`296` by `Andrew Wang`_)
- Add loss schedulers (:gh:`296` by `Andrew Wang`_)
- Add transform symmetrisation, further transform arithmetic, and new equivariant denoiser (:gh:`259` by `Andrew Wang`_)
- New transforms: multi-axis reflect, time-shift and diffeomorphism (:gh:`259` by `Andrew Wang`_)
- Multi-coil MRI, 3D MRI, MRI Mixin (:gh:`287` by `Andrew Wang`_, Brett Levac)
- Add Metric baseclass, unified params (for complex, norm, reduce), typing, tests, L1L2 metric, QNR metric, metrics docs section, Metric functional wrapper (:gh:`309`, :gh:`343` by `Andrew Wang`_)
- generate_dataset features: complex numbers, save/load physics_generator params, overwrite bool (:gh:`324`, :gh:`352` by `Andrew Wang`_)
- Add the Köhler dataset (:gh:`271` by `Jérémy Scanvic`_)

Fixed
^^^^^
- Fixed sphinx warnings (:gh:`347` by `Julian Tachella`_ and `Thomas Moreau`_)
- Fix cache file initialization in FastMRI Dataloader (:gh:`300` by `Pierre-Antoine Comby`_)
- Fixed prox_l2 no learning option in :class:`deepinv.Trainer` (:gh:`304` by `Julian Tachella`_)

- Fixed SSIM to use lightweight torchmetrics function + add MSE and NMSE as metrics + allow PSNR & SSIM to set max pixel on the fly (:gh:`296` by `Andrew Wang`_)
- Fix generate_dataset error with physics_generator and batch_size != 1. (:gh:`315` by apolychronou)
- Fix generate_dataset error not using random physics generator (:gh:`324` by `Andrew Wang`_)
- Fix Scale transform rng device error (:gh:`324` by `Andrew Wang`_)
- Fix bug when using cuda device in dinv.datasets.generate_dataset  (:gh:`334` by `Tobias Liaudat`_)
- Update outdated links in the readme (:gh:`366` by `Jérémy Scanvic`_)

Changed
^^^^^^^
- Added direct option to ArtifactRemoval (:gh:`347` by `Julian Tachella`_ and `Thomas Moreau`_)
- Sphinx template to pydata (:gh:`347` by `Julian Tachella`_ and `Thomas Moreau`_)
- Remove metrics from utils and consolidate complex and normalisation options (:gh:`309` by `Andrew Wang`_)
- get_freer_gpu falls back to torch.cuda when nvidia-smi fails (:gh:`352` by `Andrew Wang`_)
- libcpab now is a PyPi package for diffeomorphisms, add rngs and devices to transforms (:gh:`370` by `Andrew Wang`_)

v0.2.1
----------------

New Features
^^^^^^^^^^^^
- Mirror Descent algorithm with Bregman potentials (:gh:`282` by `Samuel Hurault`_)
- Added Gaussian-weighted splitting mask (from Yaman et al.), Artifact2Artifact (Liu et al.) and Phase2Phase (Eldeniz et al.) (:gh:`279` by `Andrew Wang`_)
- Added time-agnostic network wrapper (:gh:`279` by `Andrew Wang`_)
- Add sinc filter (:gh:`280` by `Julian Tachella`_)
- Add Noise2Score method (:gh:`280` by `Julian Tachella`_)
- Add Gamma Noise (:gh:`280` by `Julian Tachella`_)
- Add 3D Blur physics operator, with 3D diffraction microscope blur generators (:gh:`277` by `Florian Sarron`_, `Pierre Weiss`_, `Paul Escande`, `Minh Hai Nguyen`_) - 12/07/2024
- Add ICNN model (:gh:`281` by `Samuel Hurault`_)
- Dynamic MRI physics operator (:gh:`242` by `Andrew Wang`_)
- Add support for adversarial losses and models (GANs) (:gh:`183` by `Andrew Wang`_)
- Base transform class for transform arithmetic (:gh:`240` by `Andrew Wang`_) - 26/06/2024.
- Plot video/animation functionality (:gh:`245` by `Andrew Wang`_)
- Added update_parameters for parameter-dependent physics (:gh:`241` by Julian Tachella) - 11/06/2024
- Added evaluation functions for R2R and Splitting losses (:gh:`241` by Julian Tachella) - 11/06/2024
- Added a new `Physics` class for the Radio Interferometry problem (:gh:`230` by `Chao Tang`_, `Tobias Liaudat`_) - 07/06/2024
- Add projective and affine transformations for EI or data augmentation (:gh:`173` by `Andrew Wang`_)
- Add k-t MRI mask generators using Gaussian, random uniform and equispaced sampling stratgies (:gh:`206` by `Andrew Wang`_)
- Added Lidc-Idri buit-in datasets (:gh:`270` by Maxime SONG) - 12/07/2024
- Added Flickr2k / LSDIR / Fluorescent Microscopy Denoising  buit-in datasets (:gh:`276` by Maxime SONG) - 15/07/2024
- Added `rng` a random number generator to each :class:`deepinv.physics.generator.PhysicsGenerator` and a `seed` number argument to :func:`deepinv.physics.generator.PhysicsGenerator.step` (by `Minh Hai Nguyen`_)
- Added an equivalent of :func:`deepinv.physics.functional.random_choice` in torch, available as :func:`deepinv.physics.functional.random_choice` (by `Minh Hai Nguyen`_)

Fixed
^^^^^
- Disable unecessary gradient computation to prevent memory explosion (:gh:`301` by `Dylan Sechet`, `Samuel Hurault`)
- Wandb logging (:gh:`280` by `Julian Tachella`_)
- SURE improvements (:gh:`280` by `Julian Tachella`_)
- Fixed padding in conv_transpose2d and made conv_2d a true convolution (by `Florian Sarron`_, `Pierre Weiss`_, Paul Escande, `Minh Hai Nguyen`_) - 12/07/2024
- Fixed the gradient stopping in EILoss (:gh:`263` by `Jérémy Scanvic`_) - 27/06/2024
- Fixed averaging loss over epochs :class:`deepinv.Trainer` (:gh:`241` by Julian Tachella) - 11/06/2024
- Fixed :class:`deepinv.Trainer` save_path timestamp problem on Windows (:gh:`245` by `Andrew Wang`_)
- Fixed inpainting/SplittingLoss mask generation + more flexible tensor size handling + pixelwise masking (:gh:`267` by `Andrew Wang`_)
- Fixed the :class:`deepinv.physics.generator.ProductConvolutionBlurGenerator`, allowing for batch generation (previously does not work) by (`Minh Hai Nguyen`_)

Changed
^^^^^^^
- Redefine Prior, DataFidelity and Bregman with a common parent class Potential (:gh:`282` by `Samuel Hurault`_)
- Changed to Python 3.9+ (:gh:`280` by `Julian Tachella`_)
- Improved support for parameter-dependent operators (:gh:`227` by `Jérémy Scanvic`_) - 28/05/2024
- Added a divergence check in the conjugate gradient implementation (:gh:`225` by `Jérémy Scanvic`_) - 22/05/2024



v0.2.0
----------------
Many of the features in this version were developed by `Minh Hai Nguyen`_,
`Pierre Weiss`_, `Florian Sarron`_, `Julian Tachella`_ and `Matthieu Terris`_ during the IDRIS hackathon.

New Features
^^^^^^^^^^^^
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
- Added :class:`deepinv.Trainer`, :class:`deepinv.loss.Loss` class and eval metric (LPIPS, NIQE, SSIM) (:gh:`181` by `Julian Tachella`_) - 02/04/2024
- PhaseRetrieval class (:gh:`176` by `Zhiyuan Hu`_) - 20/03/2024
- Added 3D wavelets (:gh:`164` by `Matthieu Terris`_) - 07/03/2024
- Added patch priors loss (:gh:`164` by `Johannes Hertrich`_) - 07/03/2024
- Added Restormer model (:gh:`185` by Antoine Regnier and Maxime SONG) - 18/04/2024
- Added DIV2K built-in dataset (:gh:`203` by Maxime SONG) - 03/05/2024
- Added Urban100 built-in dataset (:gh:`237` by Maxime SONG) - 07/06/2024
- Added Set14 / CBSD68 / fastMRI buit-in datasets (:gh:`248` :gh:`249` :gh:`229` by Maxime SONG) - 25/06/2024

Fixed
^^^^^
- Fixed the None prior (:gh:`233` by `Samuel Hurault`_) - 04/06/2024
- Fixed the conjugate gradient torch.nograd for teh demos, accelerated)
- Fixed torch.nograd in demos for faster generation of the doc
- Corrected the padding for the convolution
- Solved pan-sharpening issues
- Many docstring fixes
- Fixed slow drunet sigma and batched conjugate gradient  (:gh:`181` by `Minh Hai Nguyen`_) - 02/04/2024
- Fixed g dependence on sigma in optim docs (:gh:`165` by `Julian Tachella`_) - 28/02/2024



Changed
^^^^^^^
- Refactored the documentation completely for the physics
- Refactor unfolded docs (:gh:`181` by `Julian Tachella`_) - 02/04/2024
- Refactor model docs (:gh:`172` by `Julian Tachella`_) - 12/03/2024
- Changed WaveletPrior to WaveletDenoiser (:gh:`165` by `Julian Tachella`_) - 28/02/2024
- Move from torchwavelets to ptwt (:gh:`162` by `Matthieu Terris`_) - 22/02/2024

v0.1.1
----------------

New Features
^^^^^^^^^^^^
- Added r2r loss (:gh:`148` by `Brayan Monroy`_) - 30/01/2024
- Added scale transform (:gh:`135` by `Jérémy Scanvic`_) - 19/12/2023
- Added priors for total variation and l12 mixed norm (:gh:`156` by `Nils Laurent`_) - 09/02/2023


Fixed
^^^^^
- Fixed issue in noise forward of Decomposable class (:gh:`154` by `Matthieu Terris`_) - 08/02/2024
- Fixed new black version 24.1.1 style changes (:gh:`151` by `Julian Tachella`_) - 31/01/2024
- Fixed test for sigma as torch tensor with gpu enable (:gh:`145` by `Brayan Monroy`_) - 23/12/2023
- Fixed :gh:`139` BM3D tensor format grayscale (:gh:`140` by `Matthieu Terris`_) - 23/12/2023
- Fixed :gh:`136` noise additive model for DecomposablePhysics (:gh:`138` by `Matthieu Terris`_) - 22/12/2023
- Importing `deepinv` does not modify matplotlib config anymore (:gh`1501` by `Thomas Moreau`_) - 30/01/2024


Changed
^^^^^^^
- Rephrased the README (:gh:`142` by `Jérémy Scanvic`_) - 09/01/2024


v0.1.0
----------------

New Features
^^^^^^^^^^^^
- Added autoadjoint capabilities (:gh:`151` by `Julian Tachella`_) - 31/01/2024
- Added equivariant transforms (:gh:`125` by `Matthieu Terris`_) - 07/12/2023
- Moved datasets and weights to HuggingFace (:gh:`121` by `Samuel Hurault`_) - 01/12/2023
- Added L1 prior, change distance in DataFidelity (:gh:`108` by `Samuel Hurault`_) - 03/11/2023
- Added Kaiming init (:gh:`102` by `Matthieu Terris`_) - 29/10/2023
- Added Anderson Acceleration (:gh:`86` by `Samuel Hurault`_) - 23/10/2023
- Added :func:`deepinv.sampling.DPS` diffusion method (:gh:`92` by `Julian Tachella`_ and `Hyungjin Chung`_) - 20/10/2023
- Added on-the-fly physics computations in training (:gh:`88` by `Matthieu Terris`_) - 10/10/2023
- Added `no_grad` parameter (:gh:`80` by `Jérémy Scanvic`_) - 20/08/2023
- Added prox of TV (:gh:`79` by `Matthieu Terris`_) - 16/08/2023
- Added diffpir demo + model (:gh:`77` by `Matthieu Terris`_) - 08/08/2023
- Added SwinIR model (:gh:`76` by `Jérémy Scanvic`_) - 02/08/2023
- Added hard-threshold (:gh:`71` by `Matthieu Terris`_) - 18/07/2023
- Added discord server (:gh:`64` by `Julian Tachella`_) - 10/07/2023
- Added changelog file (:gh:`64` by `Julian Tachella`_) - 10/07/2023

Fixed
^^^^^
- doc fixes + training fixes (:gh:`124` by `Julian Tachella`_) - 06/12/2023
- Add doc weights (:gh:`97` by `Matthieu Terris`_) - 24/10/2023
- Fix BlurFFT adjoint (:gh:`89` by `Matthieu Terris`_) - 15/10/2023
- Doc typos (:gh:`88` by `Matthieu Terris`_) - 10/10/2023
- Minor fixes DiffPIR + other typos (:gh:`81` by `Matthieu Terris`_) - 10/09/2023
- Call `wandb.init` only when needed (:gh:`78` by `Jérémy Scanvic`_) - 09/08/2023
- Log epoch loss instead of batch loss (:gh:`73` by `Jérémy Scanvic`_) - 21/07/2023
- Automatically disable backtracking is no explicit cost (:gh:`68` by `Samuel Hurault`_) - 12/07/2023
- Added missing indent (:gh:`63` by `Jérémy Scanvic`_) - 12/07/2023
- Fixed get_freer_gpu grep statement to work for different versions of nvidia-smi (:gh:`82` by `Alexander Mehta`_) - 20/09/2023
- Fixed get_freer_gpu to work on different operating systems (:gh:`87` by `Andrea Sebastiani`_) - 10/10/2023
- Fixed Discord server and contributiong links  (:gh:`87` by `Andrea Sebastiani`_) - 10/10/2023


Changed
^^^^^^^
- Update CI (:gh:`95` :gh:`99` by `Thomas Moreau`_) - 24/10/2023
- Changed normalization CS and SPC to 1/m (:gh:`72` by `Julian Tachella`_) - 21/07/2023
- Update docstring (:gh:`68` by `Samuel Hurault`_) - 12/07/2023



.. _Julian Tachella: https://github.com/tachella
.. _Jérémy Scanvic: https://github.com/jscanvic
.. _Samuel Hurault: https://github.com/samuro95
.. _Matthieu Terris: https://github.com/matthieutrs
.. _Alexander Mehta: https://github.com/alexmehta
.. _Andrea Sebastiani: https://github.com/sedaboni
.. _Thomas Moreau: https://github.com/tomMoral
.. _Hyungjin Chung: https://www.hj-chung.com/
.. _Eliott Bourrigan: https://github.com/eliottbourrigan
.. _Riyad Chamekh: https://github.com/riyadchk
.. _Jules Dumouchel: https://github.com/Ruli0
.. _Brayan Monroy: https://github.com/bemc22
.. _Nils Laurent: https://nils-laurent.github.io/
.. _Johannes Hertrich: https://johertrich.github.io/
.. _Minh Hai Nguyen: https://mh-nguyen712.github.io/
.. _Florian Sarron: https://fsarron.github.io/
.. _Pierre Weiss: https://www.math.univ-toulouse.fr/~weiss/
.. _Zhiyuan Hu: https://github.com/zhiyhu1605
.. _Chao Tang: https://github.com/ChaoTang0330
.. _Tobias Liaudat: https://github.com/tobias-liaudat
.. _Andrew Wang: https://andrewwango.github.io/about/
.. _Pierre-Antoine Comby: https://github.com/paquiteau
.. _Victor Sechaud: https://github.com/vsechaud
.. _Keying Guo: https://github.com/g-keying
.. _Sebastian Neumayer: https://www.tu-chemnitz.de/mathematik/invimg/index.en.php
.. _Romain Vo: https://github.com/romainvo
.. _Quentin Barthélemy: https://github.com/qbarthelemy
.. _Louise Friot Giroux: https://github.com/Louisefg
.. _Vicky De Ridder: https://github.com/nucli-vicky
.. _Chaithya G R: https://github.com/chaithyagr
