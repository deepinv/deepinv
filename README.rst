.. raw:: html

   <div align="center">
     <img src="docs/source/figures/deepinv_logolarge.png" width="500"/>
     <div>&nbsp;</div>

|Test Status| |Docs Status| |Python 3.6+| |codecov| |Black|

.. raw:: html

     </div>

Introduction
-------
Deep Inverse is an open-source pytorch library for solving imaging inverse problems using deep learning. The goal of ``deepinv`` is to accelerate the development of deep learning based methods for imaging inverse problems, by combining popular learning-based reconstruction approaches in a common and simplified framework, standarizing forward imaging models and simplifying the creation of imaging datasets. 

With ``deepinv`` you can:

* Use deep learning for solving your inverse problem. You only need to create a ``physics`` class that captures your imaging problem. You can try self-supervised learning, unrolled architectures, plug-and-play methods with pretrained denoisers and uncertainty quantification!
* Test new deep learning-based methods on various standard inverse problems (MRI, CT, deblurring, super-resolution, inpainting, colorization, etc.) and compare with existing state-of-the-art methods.
* Create and share datasets, which can be seamlessly evaluated by other users.


.. raw:: html

   <div align="center">
     <img src="docs/source/figures/deepinv_schematic.png" width="1000"/>
    </div>

Read the documentation and examples at `https://deepinv.github.io <https://deepinv.github.io>`__.

Install
----------

(To be updated, the first stable release will come soon)


Contributing
-------

The preferred way to contribute to ``deepinv`` is to fork the `main
repository <https://github.com/deepinv/deepinv/>`__ on GitHub,
then submit a "Pull Request" (PR).


Finding help
-------------

(To be updated)


Citing DeepInv
---------------

If you use ``deepinv`` in a scientific publication, please cite the following paper

(To be updated)


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
