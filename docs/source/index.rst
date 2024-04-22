DeepInverse: a Pytorch library for imaging with deep learning
==================================================================

|Test Status| |Docs Status| |Python 3.6+| |codecov| |Black|

Deep Inverse is a Pytorch based library for solving imaging inverse problems with deep learning.

Github repository: `<https://github.com/deepinv/deepinv>`_.


Featuring
==================

* |:camera_with_flash:|  Large collection of :ref:`predefined imaging operators <physics>` (MRI, CT, deblurring, inpainting, etc.)
* |:book:| :ref:`Training losses <loss>` for inverse problems (self-supervised learning, regularization, etc.).
* |:boomerang:| Many :ref:`pretrained deep denoisers <denoisers>` which can be used for :ref:`plug-and-play restoration <iterative>`.
* |:book:| Framework for :ref:`building datasets <datasets>` for inverse problems.
* |:building_construction:| Easy-to-build :ref:`unfolded architectures <unfolded>` (ADMM, forward-backward, deep equilibrium, etc.).
* |:microscope:| :ref:`Diffusion algorithms <sampling>` for image restoration and uncertainty quantification (Langevin, diffusion, etc.).
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
   deepinv.loss
   deepinv.denoisers
   deepinv.optim
   deepinv.iterative
   deepinv.unfolded
   deepinv.sampling
   deepinv.other_models
   auto_examples/index
   deepinv.multigpu
   deepinv.notation
   deepinv.contributing

Finding Help
==================
If you have any questions or suggestions, please join the conversation in our
`Discord server <https://discord.gg/qBqY5jKw3p>`_. The recommended way to get in touch with the developers is to open an issue on the
`issue tracker <https://github.com/deepinv/deepinv/issues>`_.


Lead Developers
============================

`Julian Tachella <https://tachella.github.io/>`_, `Dongdong Chen <http://dongdongchen.com/>`_,
`Samuel Hurault <https://github.com/samuro95/>`_ and `Matthieu Terris <https://matthieutrs.github.io>`_.



.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
.. |Test Status| image:: https://github.com/deepinv/deepinv/actions/workflows/test.yml/badge.svg
   :target: https://github.com/deepinv/deepinv/actions/workflows/test.yml
.. |Docs Status| image:: https://github.com/deepinv/deepinv/actions/workflows/documentation.yml/badge.svg
   :target: https://github.com/deepinv/deepinv/actions/workflows/documentation.yml
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
.. |codecov| image:: https://codecov.io/gh/deepinv/deepinv/branch/main/graph/badge.svg?token=77JRvUhQzh
   :target: https://codecov.io/gh/deepinv/deepinv
