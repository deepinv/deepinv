.. deepinverse documentation master file, created by
   sphinx-quickstart on Wed Jan  4 19:22:10 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DeepInverse: a Pytorch library for imaging with deep learning
==================================================================

Deep Inverse is a Pytorch based library for solving imaging inverse problems with deep learning.

Github repository: `<https://github.com/deepinv/deepinv>`_.


Featuring
==================

* |:camera_with_flash:|  Large collection of :ref:`predefined imaging operators <Physics>` (MRI, CT, deblurring, inpainting, etc.)
* |:book:| :ref:`Training losses <Loss>` for inverse problems (self-supervised learning, regularization, etc.).
* |:boomerang:| Many :ref:`pretrained deep denoisers <Models>` which can be used for :ref:`plug-and-play restoration <Optim>`.
* |:book:| Framework for :ref:`building datasets <Datasets>` for inverse problems.
* |:building_construction:| Easy-to-build :ref:`unfolded architectures <Unfolded>` (ADMM, forward-backward, deep equilibrium, etc.).
* |:microscope:| :ref:`Sampling algorithms <Sampling>` for uncertainty quantification (Langevin, diffusion, etc.).
* |:books:| A large number of well-explained :ref:`examples <Examples>`.


.. image:: figures/deepinv_schematic.png
   :width: 1000
   :align: center


Installation
==================

Install the latest version of ``deepinv`` via pip:

.. code-block:: bash

    pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv


Getting Started
==================

Try out one of the following deblurring examples (or pick from :ref:`full list of examples <Examples>`):

.. minigallery:: deepinv.physics.BlurFFT

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents:

   deepinv.physics
   deepinv.datasets
   deepinv.utils
   deepinv.models
   deepinv.loss
   deepinv.optim
   deepinv.unfolded
   deepinv.sampling
   auto_examples/index
   deepinv.notation


Contributing
============================

DeepInverse is a community-driven project and welcomes contributions of all forms.
We are ultimately aiming for a comprehensive library of inverse problems and deep learning,
and we need your help to get there!
The preferred way to contribute to ``deepinv`` is to fork the `main
repository <https://github.com/deepinv/deepinv/>`__ on GitHub,
then submit a "Pull Request" (PR).


Lead Developers
============================

`Julian Tachella <https://tachella.github.io/>`_, `Dongdong Chen <http://dongdongchen.com/>`_, `Samuel Hurault <https://github.com/samuro95/>`_ and `Matthieu Terris <https://matthieutrs.github.io>`_.
