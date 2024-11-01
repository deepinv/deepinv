.. _physics:

Physics
=======

This package contains a large collection of forward operators appearing in imaging applications.
The acquisition models are of the form

.. math::

    y = \noise{\forw{x}}

where :math:`x\in\xset` is an image, :math:`y\in\yset` are the measurements, :math:`A:\xset\mapsto \yset` is a
deterministic (linear or non-linear) operator capturing the physics of the acquisition and
:math:`N:\yset\mapsto \yset` is a mapping which characterizes the noise affecting the measurements.


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents:

   deepinv.physics.intro
   deepinv.physics.forward
   deepinv.physics.noise
   deepinv.physics.functional
   deepinv.physics.defining


.. grid:: 3
    :gutter: 1

    .. grid-item-card::
        :link: physics_intro
        :link-type: ref

        :octicon:`telescope-fill` **Introduction**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Introduction to the physics package.

    .. grid-item-card::
        :link: physics_functional
        :link-type: ref

        :octicon:`package` **Functional**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Various utilities for forward
        operators.


    .. grid-item-card::
        :link: forward_operators
        :link-type: ref

        :octicon:`device-camera-video` **Operators**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Forward operators for imaging.

    .. grid-item-card::
        :link: noise_distributions
        :link-type: ref

        :octicon:`law` **Noise**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Noise distributions for imaging.

    .. grid-item-card::
        :link: physics_defining
        :link-type: ref

        :octicon:`tools` **Defining your operator**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        How to define your own forward operator,
        if the existing ones are not enough.
