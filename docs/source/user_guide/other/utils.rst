.. _utils:

Utils
=====

.. _plotting:

Plotting
--------
We provide some plotting functions that are adapted to inverse problems.
The main plotting function is :class:`deepinv.utils.plot`,
which can be used to quickly plot a list of tensor images.

.. doctest::

    >>> from deepinv.utils import plot
    >>> import torch
    >>> x1 = torch.rand(4, 3, 16, 16)
    >>> x2 = torch.rand(4, 3, 16, 16)
    >>> plot([x1, x2], titles=['x1', 'x2'])


We provide other plotting functions that are useful for inverse problems:

.. list-table:: Utility Functions and Descriptions
   :header-rows: 1

   * - **Function**
     - **Description**
   * - :func:`deepinv.utils.plot`
     - Plots a list of tensor images with optional titles.
   * - :func:`deepinv.utils.plot_curves`
     - Plots curves for visualizing metrics over optimization iterations.
   * - :func:`deepinv.utils.plot_parameters`
     - Visualizes model parameters over optimization iterations.
   * - :func:`deepinv.utils.plot_inset`
     - Plots a list of images with zoomed-in insets extracted from the images.
   * - :func:`deepinv.utils.plot_videos`
     - Plots and animates a list of image sequences.
   * - :func:`deepinv.utils.plot_ortho3D`
     - Plots 3D orthographic projections for analyzing data or model outputs in three dimensions.


.. _other-utils:

Other
-----
We provide some useful utility functions:

.. list-table:: Utility Functions and Descriptions
   :header-rows: 1

   * - **Function**
     - **Description**
   * - :func:`deepinv.utils.get_freer_gpu`
     - Finds the GPU with the most available memory for optimized computation allocation.
   * - :func:`deepinv.utils.load_url_image`
     - Loads an image directly from a URL for use in experiments or demos.
   * - :func:`deepinv.utils.get_data_home`
     - Retrieves the path to the default directory for storing datasets.
   * - :func:`deepinv.utils.load_image`
     - Loads a local image file for processing and analysis.
   * - :func:`deepinv.utils.demo.demo_mri_model`
     - Demo MRI reconstruction model for use in relevant examples.


.. _tensorlist:

TensorList
----------
The :class:`deepinv.utils.TensorList` class is a wrapper around a list of tensors. It allows performing
elementary operations on the list of tensors, such as sum, multiplication, etc.:

.. doctest::

    >>> from deepinv.utils import TensorList
    >>> import torch
    >>> x1 = torch.ones(2, 3, 2, 2)
    >>> x2 = torch.ones(2, 1, 3, 3)
    >>> t1 = TensorList([x1, x2])
    >>> t2 = TensorList([x1*2, x2/2])
    >>> t3 = t1 + t2

