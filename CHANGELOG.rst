=================
Change Log
=================
This change log is for the `main` branch. It contains changes for each release, with the date and author of each change.

Current (v0.0.2)
----------------

New Features
^^^^^^^^^^^^
- Added SwinIR model (:gh:`76` by `Jérémy Scanvic`_) - 02/08/2023
- Added hard-threshold (:gh:`71` by `Matthieu Terris`_) - 18/07/2023
- Added discord server (:gh:`64` by `Julian Tachella`_) - 10/07/2023
- Added changelog file (:gh:`64` by `Julian Tachella`_) - 10/07/2023

Fixed
^^^^^
- Call `wandb.init` only when needed (:gh:`78` by `Jérémy Scanvic`_) - 09/08/2023
- Log epoch loss instead of batch loss (:gh:`73` by `Jérémy Scanvic`_) - 21/07/2023
- Automatically disable backtracking is no explicit cost (:gh:`68` by `Samuel Hurault`_) - 12/07/2023
- Added missing indent (:gh:`63` by `Jérémy Scanvic`_) - 12/07/2023


Changed
^^^^^^^
- Changed normalization CS and SPC to 1/m (:gh:`72` by `Julian Tachella`_) - 21/07/2023
- Update docstring (:gh:`68` by `Samuel Hurault`_) - 12/07/2023


Authors
^^^^^^^

.. _Julian Tachella: https://github.com/tachella
.. _Jérémy Scanvic: https://github.com/jscanvic
.. _Samuel Hurault: https://github.com/samuro95
.. _Matthieu Terris: https://github.com/matthieutrs
