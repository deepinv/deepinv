User Guide
==========

Imaging inverse problems are described by the equation :math:`y = \noise{\forw{x}}` where
:math:`x` is an unknown image we want to recover,
:math:`y` are the observed measurements, :math:`A` is a
deterministic (linear or non-linear) operator capturing the physics of the acquisition and
:math:`N` characterizes the noise affecting the measurements.

Operators
~~~~~~~~~

The library provides a large variety of imaging operators ``physics`` modelling :math:`\noise{\forw{\cdot}}`,
which can simulate the observation process:

::

    x = load_image()
    y = physics(x) # simulate observation

.. grid:: 3
    :gutter: 1

    .. grid-item-card::
        :link: physics_intro
        :link-type: ref

        :octicon:`telescope-fill` **Introduction**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Introduction to the physics package.

    .. grid-item-card::
        :link: physics
        :link-type: ref

        :octicon:`device-camera-video` **Operators**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Forward operators and noise distributions.

    .. grid-item-card::
        :link: physics_defining
        :link-type: ref

        :octicon:`tools` **Defining your operator**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        How to define your own forward operator,
        if the existing ones are not enough.

    .. grid-item-card::
        :link: physics_functional
        :link-type: ref

        :octicon:`package` **Functional**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Various utilities for forward
        operators.


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Forward Operators

   user_guide/operators/intro
   user_guide/operators/physics
   user_guide/operators/defining
   user_guide/operators/functional



Reconstruction
~~~~~~~~~~~~~~

In order to recover an image from its measurements, the library provides many
reconstruction methods :math:`\hat{x}=R(y, A)`, which often leverage knowledge of the acquisition ``physics``. Given a restoration model ``model``, the reconstruction is therefore provided as

::

    x_hat = model(y, physics)


.. grid:: 3
    :gutter: 1

    .. grid-item-card::
        :link: denoisers
        :link-type: ref

        :octicon:`package` **Denoisers**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Classical and deep denoisers with
        pretrained weights.

    .. grid-item-card::
        :link: artifact
        :link-type: ref

        :octicon:`law` **Artifact Removal**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Learned filtered back-projection.

    .. grid-item-card::
        :link: optim
        :link-type: ref

        :octicon:`rocket` **Optimization**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Priors and data-fidelity functions,
        and optimization algorithms.

    .. grid-item-card::
        :link: unfolded
        :link-type: ref

        :octicon:`diff-renamed` **Unfolded Algorithms**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Unfolded architectures.

    .. grid-item-card::
        :link: iterative
        :link-type: ref

        :octicon:`play` **Iterative Reconstruction**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Plug-and-play, RED, and deep image prior.


    .. grid-item-card::
        :link: sampling
        :link-type: ref

        :octicon:`flame` **Sampling**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Diffusion and Langevin algorithms.


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reconstruction Methods

   user_guide/denoisers
   user_guide/other_models
   user_guide/optimization
   user_guide/iterative
   user_guide/sampling
   user_guide/unfolded
   user_guide/weights


Training, Testing and Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All the tools from the library, from measurement operator to restoration methods,
are implemented as :class:`torch.nn.Module` and therefore natively support backpropagation.
Reconstruction networks ``model`` can be trained on datasets to improve their performance:

::

    trainer = Trainer(model, loss, metric, train_dataset, ...)
    trainer.train()
    trainer.test(test_dataset)


.. grid:: 3
    :gutter: 1

    .. grid-item-card::
        :link: training
        :link-type: ref

        :octicon:`zap` **Training**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Training and testing reconstruction
        models.

    .. grid-item-card::
        :link: datasets
        :link-type: ref

        :octicon:`database` **Datasets**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Utilities to generate and load datasets
        for training and testing.

    .. grid-item-card::
        :link: loss
        :link-type: ref

        :octicon:`beaker` **Loss**
        ^^^^^^^^^^^^^^^^^^^^^^^^^
        Supervised and self-supervised losses to train the models.

    .. grid-item-card::
        :link: metric
        :link-type: ref

        :octicon:`goal` **Metrics**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Distortion and perceptual metrics to evaluate reconstructions.


    .. grid-item-card::
        :link: transform
        :link-type: ref

        :octicon:`eye` **Transforms**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Transforms for data augmentation
        and self-supervised learning.


    .. grid-item-card::
        :link: utils
        :link-type: ref

        :octicon:`graph` **Utils**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Plotting and other utilities.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Training and Testing

   user_guide/datasets
   user_guide/loss
   user_guide/metric
   user_guide/transforms
   user_guide/multigpu

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Other

   user_guide/utils
   user_guide/notation