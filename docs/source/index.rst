:html_theme.sidebar_secondary.remove:

DeepInverse: a PyTorch library for imaging with deep learning
=============================================================

|Test Status| |Docs Status| |Python Version| |Black| |codecov| |pip install| |discord| |colab|

DeepInverse is a PyTorch-based library for solving imaging inverse problems with deep learning.

Github repository: `<https://github.com/deepinv/deepinv>`_.


**Featuring**

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


.. toctree::
   :maxdepth: 3
   :hidden:

   quickstart
   auto_examples/index
   user_guide
   API
   finding_help
   contributing
   community


**Maintainers**

View our current and former maintainers, and how to get in touch, at `MAINTAINERS <https://github.com/deepinv/deepinv/blob/main/MAINTAINERS.md>`_.

**Citation**

If you use DeepInverse in your research, please cite the following paper (available on `arXiv <https://arxiv.org/abs/2505.20160>`_):

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

**Star history**

.. raw:: html

   <a href="https://www.star-history.com/#deepinv/deepinv&Date">
    <picture>
      <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=deepinv/deepinv&type=Date" />
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=deepinv/deepinv&type=Date&theme=dark" />
      <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=deepinv/deepinv&type=Date" />
    </picture>
   </a>


.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
.. |Test Status| image:: https://github.com/deepinv/deepinv/actions/workflows/test.yml/badge.svg
   :target: https://github.com/deepinv/deepinv/actions/workflows/test.yml
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
   :target: https://colab.research.google.com/drive/1XhCO5S1dYN3eKm4NEkczzVU7ZLBuE42J
