.. image:: https://github.com/deepinv/deepinv/raw/main/docs/source/figures/deepinv_logolarge.png
   :width: 500px
   :alt: deepinv logo
   :align: center


|Test Status| |GPU Test Status|  |Docs Status| |GPU Docs Status| |Python Version| |Black| |codecov| |pip install| |discord| |colab| |youtube| |paper|


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
* Many `state-of-the-art deep neural networks <https://deepinv.github.io/deepinv/user_guide/reconstruction/introduction.html>`_, including pretrained out-of-the-box `reconstruction models <https://deepinv.github.io/deepinv/user_guide/reconstruction/pretrained-models.html>`_ and `denoisers <https://deepinv.github.io/deepinv/user_guide/reconstruction/denoisers.html>`_ 
* Comprehensive frameworks for `plug-and-play restoration <https://deepinv.github.io/deepinv/user_guide/reconstruction/iterative.html>`_, `optimization <https://deepinv.github.io/deepinv/user_guide/reconstruction/optimization.html>`_ and `unfolded architectures <https://deepinv.github.io/deepinv/user_guide/reconstruction/unfolded.html>`_
* `Training losses <https://deepinv.github.io/deepinv/user_guide/training/loss.html>`_ for inverse problems
* `Sampling algorithms and diffusion models <https://deepinv.github.io/deepinv/user_guide/reconstruction/sampling.html>`_ for uncertainty quantification
* A framework for `building datasets <https://deepinv.github.io/deepinv/user_guide/training/datasets.html>`_ for inverse problems

Mailing list
~~~~~~~~~~~~

`Join our mailing list <https://forms.gle/TFyT7M2HAWkJYfvQ7>`_ for occasional updates on releases and new features.

Install
-------

Install the latest stable release of ``deepinv``:

.. code-block:: bash

   pip install deepinv

Or, use `uv` for a faster install:

.. code-block:: bash

   uv pip install deepinv

Or, to also install optional dependencies:

.. code-block:: bash

   pip install deepinv[dataset,denoisers]

Since ``deepinv`` is under active development, you can install the latest nightly version using:

.. code-block:: bash

   pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv

Or, for updating an existing installation:

.. code-block:: bash

   pip install --upgrade --force-reinstall --no-deps git+https://github.com/deepinv/deepinv.git#egg=deepinv


Finding help
------------

If you have any questions or suggestions, please join the conversation in our
`Discord server <https://discord.gg/qBqY5jKw3p>`_. The recommended way to get in touch with the developers about any bugs or issues is to
`open an issue <https://github.com/deepinv/deepinv/issues>`_.

Maintainers
~~~~~~~~~~~

Get in touch with our `MAINTAINERS <https://github.com/deepinv/deepinv/blob/main/MAINTAINERS.md>`_.


Contributing
------------

DeepInverse is a `community-driven project <https://deepinv.github.io/deepinv/community.html>`_ and we encourage contributions of all forms.
We are building a comprehensive library of inverse problems and deep learning,
and we need your help to get there! 

Interested? `Check out how you can contribute <https://deepinv.github.io/deepinv/contributing.html>`_!

Citation
--------
If you use DeepInverse in your research, please cite `our paper on JOSS <https://joss.theoj.org/papers/10.21105/joss.08923>`_:


.. code-block:: bash

    @article{tachella2025deepinverse,
        title = {DeepInverse: A Python package for solving imaging inverse problems with deep learning},
        journal = {Journal of Open Source Software},
        doi = {10.21105/joss.08923},
        url = {https://doi.org/10.21105/joss.08923},
        year = {2025},
        publisher = {The Open Journal},
        volume = {10},
        number = {115},
        pages = {8923},
        author = {Tachella, Julián and Terris, Matthieu and Hurault, Samuel and Wang, Andrew and Davy, Leo and Scanvic, Jérémy and Sechaud, Victor and Vo, Romain and Moreau, Thomas and Davies, Thomas and Chen, Dongdong and Laurent, Nils and Monroy, Brayan and Dong, Jonathan and Hu, Zhiyuan and Nguyen, Minh-Hai and Sarron, Florian and Weiss, Pierre and Escande, Paul and Massias, Mathurin and Modrzyk, Thibaut and Levac, Brett and Liaudat, Tobías I. and Song, Maxime and Hertrich, Johannes and Neumayer, Sebastian and Schramm, Georg},
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
.. |GPU Test Status| image:: https://github.com/deepinv/deepinv/actions/workflows/test_gpu.yml/badge.svg?branch=main&event=push
   :target: https://github.com/deepinv/deepinv/actions/workflows/test_gpu.yml
.. |Docs Status| image:: https://github.com/deepinv/deepinv/actions/workflows/documentation.yml/badge.svg
   :target: https://github.com/deepinv/deepinv/actions/workflows/documentation.yml
.. |GPU Docs Status| image:: https://github.com/deepinv/deepinv/actions/workflows/gpu_docs.yml/badge.svg?branch=main&event=push
   :target: https://github.com/deepinv/deepinv/actions/workflows/gpu_docs.yml
.. |Python Version| image:: https://img.shields.io/badge/python-3.10%2B-blue
   :target: https://www.python.org/downloads/release/python-3100/
.. |codecov| image:: https://codecov.io/gh/deepinv/deepinv/branch/main/graph/badge.svg?token=77JRvUhQzh
   :target: https://codecov.io/gh/deepinv/deepinv
.. |pip install| image:: https://img.shields.io/pypi/dm/deepinv.svg?logo=pypi&label=pip%20install&color=fedcba
   :target: https://pypistats.org/packages/deepinv
.. |discord| image:: https://dcbadge.limes.pink/api/server/qBqY5jKw3p?style=flat
   :target: https://discord.gg/qBqY5jKw3p
.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/deepinv/deepinv/blob/gh-pages/auto_examples/_notebooks/basics/demo_quickstart.ipynb
.. |youtube| image:: https://img.shields.io/badge/YouTube-deepinv-red?logo=youtube
   :target: https://www.youtube.com/@deepinv
.. |paper| image:: https://joss.theoj.org/papers/10.21105/joss.08923/status.svg
   :target: https://doi.org/10.21105/joss.08923
