deepinv.datasets
================

This subpackage can be used for generating reconstruction datasets from other base datasets (e.g. MNIST or CelebA).


HD5Dataset
----------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.datasets.HDF5Dataset
    deepinv.datasets.generate_dataset


PatchDataset
------------

Generate a dataset of all patches out of a tensor of images.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.datasets.PatchDataset

Image Datasets
--------------

Ready-made datasets available in the `deepinv.datasets` module.

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
