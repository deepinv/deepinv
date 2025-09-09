deepinv.datasets
================

This module can be used for defining datasets or generating reconstruction datasets from other base datasets.
Please refer to the :ref:`user guide <datasets>` for more information.

Base Datasets
-------------
.. userguide:: base-datasets

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.datasets.ImageDataset
    deepinv.datasets.ImageFolder
    deepinv.datasets.TensorDataset

.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

   deepinv.datasets.check_dataset

Generating Datasets
-------------------
.. userguide:: generating-datasets

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.datasets.HDF5Dataset


.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

   deepinv.datasets.generate_dataset


Image Datasets
--------------
.. userguide:: predefined-datasets

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.datasets.DIV2K
    deepinv.datasets.Urban100HR
    deepinv.datasets.Set14HR
    deepinv.datasets.CBSD68
    deepinv.datasets.FastMRISliceDataset
    deepinv.datasets.SimpleFastMRISliceDataset
    deepinv.datasets.CMRxReconSliceDataset
    deepinv.datasets.SKMTEASliceDataset
    deepinv.datasets.LidcIdriSliceDataset
    deepinv.datasets.Flickr2kHR
    deepinv.datasets.LsdirHR
    deepinv.datasets.FMD
    deepinv.datasets.Kohler
    deepinv.datasets.NBUDataset


Other Datasets
--------------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.datasets.PatchDataset
    deepinv.datasets.utils.PlaceholderDataset


Data Transforms
---------------
.. userguide:: data-transforms

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.datasets.utils.Rescale
    deepinv.datasets.utils.ToComplex
    deepinv.datasets.utils.Crop
    deepinv.datasets.MRISliceTransform
