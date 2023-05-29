.. raw:: html

   <div align="center">
     <img src="docs/source/figures/deepinv_logolarge.png" width="500"/>
     <div>&nbsp;</div>

|Test Status| |Docs Status| |Python 3.6+| |codecov| |Black|

.. raw:: html

     </div>

Introduction
------------
Deep Inverse is an open-source pytorch library for solving imaging inverse problems using deep learning. The goal of ``deepinv`` is to accelerate the development of deep learning based methods for imaging inverse problems, by combining popular learning-based reconstruction approaches in a common and simplified framework, standarizing forward imaging models and simplifying the creation of imaging datasets. 

With ``deepinv`` you can:


* `:camera:` Large collection of `predefined imaging operators <https://deepinv.github.io/deepinv/deepinv.physics.html>`_ (MRI, CT, deblurring, inpainting, etc.)
* `:book:` `Training losses <https://deepinv.github.io/deepinv/deepinv.loss.html>`_ for inverse problems (self-supervised learning, regularization, etc.).
* `:boomerang:` Many `pretrained deep denoisers <https://deepinv.github.io/deepinv/deepinv.models.html>`_ which can be used for `plug-and-play restoration <https://deepinv.github.io/deepinv/deepinv.optim.html>`_.
* `:book:` Framework for `building datasets <https://deepinv.github.io/deepinv/deepinv.models.html>`_ for inverse problems.
* `:building_construction:` Easy-to-build `unfolded architectures <https://deepinv.github.io/deepinv/deepinv.unfolded.html>`_ (ADMM, forward-backward, deep equilibrium, etc.).
* `:microscope:` `Sampling algorithms <https://deepinv.github.io/deepinv/deepinv.sampling.html>`_ for uncertainty quantification (Langevin, diffusion, etc.).
* `:books:` A large number of well-explained `examples <https://deepinv.github.io/deepinv/auto_examples/index.html>`_.


.. raw:: html

   <div align="center">
     <img src="docs/source/figures/deepinv_schematic.png" width="1000"/>
    </div>


Documentation
-------------

Read the documentation and examples at `https://deepinv.github.io <https://deepinv.github.io>`_.

Install
-------

If you can't wait until the next release, install the latest version of ``deepinv`` from source:

.. code-block:: bash

    pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv

Getting Started
---------------

Try out `one of the examples <https://deepinv.github.io/deepinv/auto_examples/index.html>`_ to get started.

Contributing
------------

DeepInverse is a community-driven project and welcomes contributions of all forms.
We are ultimately aiming for a comprehensive library of inverse problems and deep learning,
and we need your help to get there!
The preferred way to contribute to ``deepinv`` is to fork the `main
repository <https://github.com/deepinv/deepinv/>`__ on GitHub,
then submit a "Pull Request" (PR).


Finding help
------------

The recommended way to get in touch with the developers is to open an issue on the
`issue tracker <https://github.com/deepinv/deepinv/issues>`_.


.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
.. |Test Status| image:: https://github.com/deepinv/deepinv/actions/workflows/test.yml/badge.svg
   :target: https://github.com/deepinv/deepinv/actions/workflows/test.yml
.. |Docs Status| image:: https://github.com/deepinv/deepinv/actions/workflows/documentation.yaml/badge.svg
   :target: https://github.com/deepinv/deepinv/actions/workflows/documentation.yaml
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
.. |codecov| image:: https://codecov.io/gh/deepinv/deepinv/branch/main/graph/badge.svg?token=77JRvUhQzh
   :target: https://codecov.io/gh/deepinv/deepinv
