.. _user_guide:

User Guide
==========

Imaging inverse problems are described by the equation :math:`y = \noise{\forw{x}}` where
:math:`x` is an unknown signal (image, volume, etc.) we want to recover,
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

.. grid:: 1 2 3 3
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

   user_guide/physics/intro
   user_guide/physics/physics
   user_guide/physics/defining
   user_guide/physics/functional



Reconstruction
~~~~~~~~~~~~~~

In order to recover an image from its measurements, the library provides many
reconstruction methods :math:`\hat{x}=R(y, A)`, which often leverage knowledge of the acquisition ``physics``.
Given a restoration model ``model``, the reconstruction is therefore provided as

::

    x_hat = model(y, physics) # reconstruct signal


.. grid:: 1 2 3 3
    :gutter: 1

    .. grid-item-card::
        :link: reconstructors
        :link-type: ref

        :octicon:`telescope-fill` **Introduction**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Introduction and summary of reconstruction algorithms.

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
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Reconstruction networks from denoisers
        and other image-to-image networks.

    .. grid-item-card::
        :link: optim
        :link-type: ref

        :octicon:`rocket` **Optimization**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Priors and data-fidelity functions,
        and optimization algorithms.

    .. grid-item-card::
        :link: unfolded
        :link-type: ref

        :octicon:`diff-renamed` **Unfolded Algorithms**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Unfolded architectures.

    .. grid-item-card::
        :link: iterative
        :link-type: ref

        :octicon:`play` **Iterative Reconstruction**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Plug-and-play, RED, variational methods.

    .. grid-item-card::
        :link: sampling
        :link-type: ref

        :octicon:`flame` **Sampling**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Diffusion and MCMC algorithms.

    .. grid-item-card::
        :link: adversarial
        :link-type: ref

        :octicon:`webhook` **Adversarial Reconstruction**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Conditional, unconditional GANs and deep image prior.

    .. grid-item-card::
        :link: specific
        :link-type: ref

        :octicon:`sun` **Custom Reconstruction Networks**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Custom reconstruction methods and networks.


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reconstruction Methods

   user_guide/reconstruction/introduction
   user_guide/reconstruction/denoisers
   user_guide/reconstruction/artifact
   user_guide/reconstruction/optimization
   user_guide/reconstruction/iterative
   user_guide/reconstruction/sampling
   user_guide/reconstruction/unfolded
   user_guide/reconstruction/adversarial
   user_guide/reconstruction/specific
   user_guide/reconstruction/weights


Training, Testing and Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All the tools from the library, from measurement operator to restoration methods,
are implemented as :class:`torch.nn.Module` and therefore natively support backpropagation.
Reconstruction networks ``model`` can be trained on datasets to improve their performance:

::

    trainer = Trainer(model, loss, optimizer, metric, train_dataset, ...)
    trainer.train()
    trainer.test(test_dataset)


.. grid:: 1 2 3 3
    :gutter: 1

    .. grid-item-card::
        :link: trainer
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

   user_guide/training/trainer
   user_guide/training/datasets
   user_guide/training/loss
   user_guide/training/metric
   user_guide/training/transforms
   user_guide/training/multigpu

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Other

   user_guide/other/utils
   user_guide/other/notation