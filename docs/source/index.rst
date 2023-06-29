DeepInverse: a Pytorch library for imaging with deep learning
==================================================================

Deep Inverse is a Pytorch based library for solving imaging inverse problems with deep learning.

Github repository: `<https://github.com/deepinv/deepinv>`_.


Featuring
==================

* |:camera_with_flash:|  Large collection of :ref:`predefined imaging operators <physics>` (MRI, CT, deblurring, inpainting, etc.)
* |:book:| :ref:`Training losses <loss>` for inverse problems (self-supervised learning, regularization, etc.).
* |:boomerang:| Many :ref:`pretrained deep denoisers <models>` which can be used for :ref:`plug-and-play restoration <pnp>`.
* |:book:| Framework for :ref:`building datasets <datasets>` for inverse problems.
* |:building_construction:| Easy-to-build :ref:`unfolded architectures <unfolded>` (ADMM, forward-backward, deep equilibrium, etc.).
* |:microscope:| :ref:`Sampling algorithms <sampling>` for uncertainty quantification (Langevin, diffusion, etc.).
* |:books:| A large number of well-explained :ref:`examples <examples>`, from basics to state-of-the-art methods.


.. image:: figures/deepinv_schematic.png
   :width: 1000
   :align: center


Installation
==================

Install the latest version of ``deepinv`` via pip:

.. code-block:: bash

    pip install deepinv

You can also install the latest version of ``deepinv`` directly from github:

.. code-block:: bash

    pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv

Getting Started
==================

Try out one of the following deblurring examples (or pick from :ref:`full list of examples <examples>`):

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
   deepinv.pnp
   deepinv.unfolded
   deepinv.sampling
   auto_examples/index
   deepinv.notation
   deepinv.contributing



Lead Developers
============================

`Julian Tachella <https://tachella.github.io/>`_, `Dongdong Chen <http://dongdongchen.com/>`_,
`Samuel Hurault <https://github.com/samuro95/>`_ and `Matthieu Terris <https://matthieutrs.github.io>`_.
