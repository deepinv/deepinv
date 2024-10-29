User Guide
==========

.. grid:: 3
    :gutter: 1

    .. grid-item-card::
        :link: physics
        :link-type: ref

        :octicon:`device-camera-video` **Physics**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Define the physics operators and forward models.

    .. grid-item-card::
        :link: datasets
        :link-type: ref

        :octicon:`database` **Database**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Utilities to load and preprocess datasets.

    .. grid-item-card::
        :link: loss
        :link-type: ref

        :octicon:`graph` **Loss**
        ^^^^^^^^^^^^^^^^^^^^^^^^^
        Losses to train the models.

    .. grid-item-card::
        :link: metric
        :link-type: ref

        :octicon:`goal` **Metrics**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Metrics to evaluate the models.

    .. grid-item-card::
        :link: denoisers
        :link-type: ref

        :octicon:`package` **Denoisers**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Denoiser models.

    .. grid-item-card::
        :link: optim
        :link-type: ref

        :octicon:`law` **Optimization**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Optimization algorithms.


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents:

   deepinv.physics
   deepinv.datasets
   deepinv.utils
   deepinv.loss
   deepinv.metric
   deepinv.transform
   deepinv.denoisers
   deepinv.optim
   deepinv.iterative
   deepinv.unfolded
   deepinv.sampling
   deepinv.other_models