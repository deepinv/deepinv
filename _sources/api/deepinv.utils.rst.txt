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

Other
-----
.. userguide:: other-utils

.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

    deepinv.utils.get_freer_gpu
    deepinv.utils.get_data_home
    deepinv.utils.get_image_url
    deepinv.utils.get_degradation_url
    deepinv.utils.load_url_image
    deepinv.utils.load_example
    deepinv.utils.download_example
    deepinv.utils.load_image
    deepinv.utils.load_dataset
    deepinv.utils.load_degradation
    deepinv.utils.load_torch_url
    deepinv.utils.load_np_url
    deepinv.utils.dirac_like