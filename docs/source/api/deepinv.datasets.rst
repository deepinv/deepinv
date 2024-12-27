deepinv.datasets
================

This subpackage can be used for generating reconstruction datasets from other base datasets.
Please refer to the :ref:`user guide <datasets>` for more information.


Generating Datasets
-------------------
.. userguide:: generating-datasets

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.datasets.HDF5Dataset
    deepinv.datasets.generate_dataset


PatchDataset
------------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.datasets.PatchDataset

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
    deepinv.datasets.LidcIdriSliceDataset
    deepinv.datasets.Flickr2kHR
    deepinv.datasets.LsdirHR
    deepinv.datasets.FMD
    deepinv.datasets.Kohler
    deepinv.datasets.NBUDataset