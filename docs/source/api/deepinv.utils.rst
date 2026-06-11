deepinv.utils
=============

This module provides various plotting and utility functions.
Please refer to the :ref:`user guide <utils>` for more information.

Plotting
--------
.. userguide:: plotting

.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

        deepinv.utils.plot
        deepinv.utils.plot_curves
        deepinv.utils.plot_parameters
        deepinv.utils.plot_inset
        deepinv.utils.plot_videos
        deepinv.utils.save_videos
        deepinv.utils.plot_ortho3D
        deepinv.utils.disable_tex
        deepinv.utils.enable_tex
        deepinv.utils.normalize_signal
        deepinv.utils.plotting.config_matplotlib


TensorList
----------
.. userguide:: tensorlist

.. autosummary::
   :toctree: stubs
   :template: myclas_template.rst
   :nosignatures:

    deepinv.utils.TensorList

.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

    deepinv.utils.zeros_like
    deepinv.utils.ones_like
    deepinv.utils.randn_like
    deepinv.utils.rand_like

Logging
-------
.. userguide:: logging

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

        deepinv.utils.AverageMeter
        deepinv.utils.ProgressMeter
        deepinv.utils.get_timestamp

Mixins
------
.. userguide:: mixin

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

        deepinv.utils.MRIMixin
        deepinv.utils.TimeMixin
        deepinv.utils.TiledMixin2d

Image Loading
-------------
.. userguide:: io-utils

.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

    deepinv.utils.load_dicom
    deepinv.utils.load_nifti
    deepinv.utils.load_url
    deepinv.utils.load_np
    deepinv.utils.load_torch
    deepinv.utils.load_mat
    deepinv.utils.load_raster
    deepinv.utils.load_ismrmd

Demo Utils
----------

.. userguide:: demo-utils

.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

    deepinv.utils.load_image
    deepinv.utils.load_url_image
    deepinv.utils.load_np_url
    deepinv.utils.load_torch_url
    deepinv.utils.load_example
    deepinv.utils.download_example
    deepinv.utils.get_cache_home
    deepinv.utils.get_image_url
    deepinv.utils.get_degradation_url
    deepinv.utils.load_dataset
    deepinv.utils.load_degradation

Phantoms
--------

.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

    deepinv.utils.phantoms.generate_shepp_logan
    deepinv.utils.phantoms.generate_random_phantom
    deepinv.utils.phantoms.generate_pet_phantom


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.utils.phantoms.SheppLoganDataset
    deepinv.utils.phantoms.RandomPhantomDataset

Tiling / Untiling (Patching and Unpatching)
-------------------------------------------

.. userguide:: tiling-utils

.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

    deepinv.utils.patch_extractor
    deepinv.utils.image_to_patches
    deepinv.utils.patches_to_image
    deepinv.utils.patchify

Other
-----

.. userguide:: other-utils

.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

    deepinv.utils.get_freer_gpu
    deepinv.utils.dirac
    deepinv.utils.dirac_like
    deepinv.utils.dirac_comb
    deepinv.utils.dirac_comb_like