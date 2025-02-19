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

.. _logging:

Logging
-------
.. list-table:: Logging functionality
   :header-rows: 1

   * - **Function/class**
     - **Description**
   * - :func:`deepinv.utils.AverageMeter`
     - Store values and keep track of average and std.
   * - :func:`deepinv.utils.get_timestamp`
     - Get current timestamp string.

.. _other-utils:

Other
-----
We provide some useful utility and demo functions:

.. list-table:: Utility and demo Functions
   :header-rows: 1

   * - **Function**
     - **Description**
   * - :func:`deepinv.utils.get_freer_gpu`
     - Finds the GPU with the most available memory.
   * - :func:`deepinv.utils.get_data_home`
     - Get the path to the default directory for storing datasets.
   * - :func:`deepinv.utils.get_image_url`
     - Get URL for image from DeepInverse HuggingFace repository.
   * - :func:`deepinv.utils.get_degradation_url`
     - Get URL for degradation from DeepInverse HuggingFace repository.
   * - :func:`deepinv.utils.load_url_image`
     - Loads an image directly from a URL for experiments or demos.
   * - :func:`deepinv.utils.load_image`
     - Loads a local image file for experiments or demos.
   * - :func:`deepinv.utils.load_dataset`
     - Loads an ImageFolder dataset from DeepInverse HuggingFace repository.
   * - :func:`deepinv.utils.load_degradation`
     - Loads a degradation tensor from DeepInverse HuggingFace repository.


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

