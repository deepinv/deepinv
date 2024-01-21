=================
Change Log
=================
This change log is for the `main` branch. It contains changes for each release, with the date and author of each change.

Current
----------------

New Features
^^^^^^^^^^^^
- Added scale transform (:gh:`135` by `Jérémy Scanvic`_) - 19/12/2023


Fixed
^^^^^
- Fixed test for sigma as torch tensor with gpu enable (:gh:`145` by `Brayan Monroy`_) - 23/12/2023
- Fixed :gh:`139` BM3D tensor format grayscale (:gh:`140` by `Matthieu Terris`_) - 23/12/2023
- Fixed :gh:`136` noise additive model for DecomposablePhysics (:gh:`138` by `Matthieu Terris`_) - 22/12/2023


Changed
^^^^^^^
- Rephrased the README (:gh:`142` by `Jérémy Scanvic`_) - 09/01/2024


v0.1.0
----------------

New Features
^^^^^^^^^^^^
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
