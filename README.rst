.. image:: https://github.com/deepinv/deepinv/raw/main/docs/source/figures/deepinv_logolarge.png
   :width: 500px
   :alt: deepinv logo
   :align: center


|Test Status| |Docs Status| |Python Version| |Black| |codecov| |pip install| |discord| |colab|   


Introduction
------------
`DeepInverse <https://deepinv.github.io/deepinv>`_ is an open-source PyTorch-based library for solving imaging inverse problems with deep learning. ``deepinv`` accelerates deep learning research across imaging domains, enhances research reproducibility via a common modular framework of problems and algorithms, and lowers the entrance bar to new practitioners.


.. image:: https://github.com/deepinv/deepinv/raw/main/docs/source/figures/deepinv_schematic.png
   :width: 1000px
   :alt: deepinv schematic
   :align: center

Get started
-----------

Read our **documentation** at `deepinv.github.io <https://deepinv.github.io>`_. Check out our `5 minute quickstart tutorial <https://deepinv.github.io/deepinv/auto_examples/basics/demo_quickstart.html>`_, our `comprehensive examples <https://deepinv.github.io/deepinv/auto_examples/index.html>`_, or our `User Guide <https://deepinv.github.io/deepinv/user_guide.html>`_.

``deepinv`` features

* A large framework of `predefined imaging operators <https://deepinv.github.io/deepinv/user_guide/physics/physics.html>`_
* Many `state-of-the-art deep neural networks <https://deepinv.github.io/deepinv/user_guide/reconstruction/introduction.html>`_, including pretrained out-of-the-box `reconstruction models <https://deepinv.github.io/deepinv/user_guide/reconstruction/introduction.html#pretrained-models>`_ and `denoisers <https://deepinv.github.io/deepinv/user_guide/reconstruction/denoisers.html>`_ 
* Comprehensive frameworks for `plug-and-play restoration <https://deepinv.github.io/deepinv/user_guide/reconstruction/iterative.html>`_, `optimization <https://deepinv.github.io/deepinv/user_guide/reconstruction/optimization.html>`_ and `unfolded architectures <https://deepinv.github.io/deepinv/user_guide/reconstruction/unfolded.html>`_
* `Training losses <https://deepinv.github.io/deepinv/user_guide/training/loss.html>`_ for inverse problems
* `Sampling algorithms and diffusion models <https://deepinv.github.io/deepinv/user_guide/reconstruction/sampling.html>`_ for uncertainty quantification
* A framework for `building datasets <https://deepinv.github.io/deepinv/user_guide/training/datasets.html>`_ for inverse problems

Install
-------

Install the latest stable release of ``deepinv``:

.. code-block:: bash

   pip install deepinv

   # Or:

   uv pip install deepinv # faster

   # Or, for additional dependencies:

   pip install deepinv[dataset,denoisers]

Since ``deepinv`` is under active development, you can install the latest nightly version using:

.. code-block:: bash

   pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv

   # Or, for updating:

   pip install --upgrade --force-reinstall --no-deps git+https://github.com/deepinv/deepinv.git#egg=deepinv


Contributing
------------

DeepInverse is a community-driven project and we encourage contributions of all forms.
We are building a comprehensive library of inverse problems and deep learning,
and we need your help to get there!

Interested? Check out our `contributing guide <https://deepinv.github.io/deepinv/contributing.html>`_.


Finding help
------------

If you have any questions or suggestions, please join the conversation in our
`Discord server <https://discord.gg/qBqY5jKw3p>`_. The recommended way to get in touch with the developers about any bugs or issues is to
`open an issue <https://github.com/deepinv/deepinv/issues>`_.

Maintainers
-----------

Get in touch with our `MAINTAINERS <https://github.com/deepinv/deepinv/blob/main/MAINTAINERS.md>`_.


Citation
--------
If you use DeepInverse in your research, please cite `our paper on arXiv <https://arxiv.org/abs/2505.20160>`_:


.. code-block:: bash

    @software{tachella2025deepinverse,
          title={DeepInverse: A Python package for solving imaging inverse problems with deep learning},
          author={Julián Tachella and Matthieu Terris and Samuel Hurault and Andrew Wang and Dongdong Chen and Minh-Hai Nguyen and Maxime Song and Thomas Davies and Leo Davy and Jonathan Dong and Paul Escande and Johannes Hertrich and Zhiyuan Hu and Tobías I. Liaudat and Nils Laurent and Brett Levac and Mathurin Massias and Thomas Moreau and Thibaut Modrzyk and Brayan Monroy and Sebastian Neumayer and Jérémy Scanvic and Florian Sarron and Victor Sechaud and Georg Schramm and Romain Vo and Pierre Weiss},
          year={2025},
          eprint={2505.20160},
          archivePrefix={arXiv},
          primaryClass={eess.IV},
          url={https://arxiv.org/abs/2505.20160},
    }


Star history
------------

.. image:: https://api.star-history.com/svg?repos=deepinv/deepinv&type=Date
   :alt: Star History Chart
   :target: https://www.star-history.com/#deepinv/deepinv&Date


.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
.. |Test Status| image:: https://github.com/deepinv/deepinv/actions/workflows/test_recurrent_main.yml/badge.svg
   :target: https://github.com/deepinv/deepinv/actions/workflows/test_recurrent_main.yml
.. |Docs Status| image:: https://github.com/deepinv/deepinv/actions/workflows/documentation.yml/badge.svg
   :target: https://github.com/deepinv/deepinv/actions/workflows/documentation.yml
.. |Python Version| image:: https://img.shields.io/badge/python-3.10%2B-blue
   :target: https://www.python.org/downloads/release/python-3100/
.. |codecov| image:: https://codecov.io/gh/deepinv/deepinv/branch/main/graph/badge.svg?token=77JRvUhQzh
   :target: https://codecov.io/gh/deepinv/deepinv
.. |pip install| image:: https://img.shields.io/pypi/dm/deepinv.svg?logo=pypi&label=pip%20install&color=fedcba
   :target: https://pypistats.org/packages/deepinv
.. |discord| image:: https://dcbadge.limes.pink/api/server/qBqY5jKw3p?style=flat
   :target: https://discord.gg/qBqY5jKw3p
.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/11YKc_fq4VS70fL8mFzmWgWpZJ7iTE9tI?usp=sharing
