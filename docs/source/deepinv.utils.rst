.. _utils:

Utils
=====


Training and Testing
--------------------
Functions to train and test the model, and save the model and the results.

.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

        deepinv.train
        deepinv.test
        deepinv.training_utils.train_normalizing_flow


Plotting
--------
We provide some basic plotting functions that are adapted to inverse problems.

.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

        deepinv.utils.plot
        deepinv.utils.plot_curves
        deepinv.utils.plot_parameters


TensorList
----------
The TensorList class is a wrapper around a list of tensors.
It can be used to represent signals or measurements that are naturally better
represented as a list of tensors of different sizes, rather than a single tensor.
TensorLists can be added, multiplied by a scalar, concatenated, etc., in a similar fashion to
torch.tensor.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

        deepinv.utils.TensorList

We also provide functions to quickly create TensorLists of zeros, ones, or random values.

.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

        deepinv.utils.zeros_like
        deepinv.utils.ones_like
        deepinv.utils.randn_like
        deepinv.utils.rand_like


Other
-----


.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

        deepinv.utils.cal_psnr
        deepinv.utils.get_freer_gpu
        deepinv.utils.load_url_image

