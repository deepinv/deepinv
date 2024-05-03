=================
Change Log
=================
This change log is for the `main` branch. It contains changes for each release, with the date and author of each change.



Current
----------------


New Features
^^^^^^^^^^^^


Fixed
^^^^^


Changed
^^^^^^^



v0.2.0
----------------
Many of the features in this version were developed by `Minh Hai Nguyen`_,
`Pierre Weiss`_, `Florian Sarron`_, `Julian Tachella`_ and `Matthieu Terris`_ during the IDRIS hackathon.

New Features
^^^^^^^^^^^^
- Added a parameterization of the operators and noiselevels for the physics class
- Added a physics.functional submodule
- Modified the Blur class to handle color, grayscale, single and multi-batch images
- Added a PhyisicsGenerator class to synthetize parameters for the forward operators
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
- Added Trainer, Loss class and eval metric (LPIPS, NIQE, SSIM) (:gh:`181` by `Julian Tachella`_) - 02/04/2024
- PhaseRetrieval class (:gh:`176` by `Zhiyuan Hu`_) - 20/03/2024
- Added 3D wavelets (:gh:`164` by `Matthieu Terris`_) - 07/03/2024
- Added patch priors loss (:gh:`164` by `Johannes Hertrich`_) - 07/03/2024
- Added Restormer model (:gh:`185` by Antoine Regnier and Maxime SONG) - 18/04/2024
- Added DIV2K built-in dataset (:gh:`203` by Maxime SONG) - 03/05/2024

Fixed
^^^^^
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
- Added `DPS` diffusion method (:gh:`92` by `Julian Tachella`_ and `Hyungjin Chung`_) - 20/10/2023
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
- Fixed get_freer_gpu grep statement to work for different versions of nvidia-smi (:gh: `82` by `Alexander Mehta`_) - 20/09/2023
- Fixed get_freer_gpu to work on different operating systems (:gh: `87` by `Andrea Sebastiani`_) - 10/10/2023
- Fixed Discord server and contributiong links  (:gh: `87` by `Andrea Sebastiani`_) - 10/10/2023


Changed
^^^^^^^
- Update CI (:gh:`95` :gh:`99` by `Thomas Moreau`_) - 24/10/2023
- Changed normalization CS and SPC to 1/m (:gh:`72` by `Julian Tachella`_) - 21/07/2023
- Update docstring (:gh:`68` by `Samuel Hurault`_) - 12/07/2023


Authors
^^^^^^^

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
.. _Minh Hai Nguyen: https://fr.linkedin.com/in/minh-hai-nguyen-7120
.. _Florian Sarron: https://fsarron.github.io/
.. _Pierre Weiss: https://www.math.univ-toulouse.fr/~weiss/
.. _Zhiyuan Hu: https://github.com/zhiyhu1605