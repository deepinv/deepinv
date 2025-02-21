.. _reconstructors:

Introduction
------------
Reconstruction algorithms define an inversion function :math:`\hat{x}=\inversef{y}{A}`
which attempts to recover a signal :math:`x` from measurements :math:`y`, (possibly) given an operator :math:`A`.
All reconstruction algorithms in the library inherit from the
:class:`deepinv.models.Reconstructor` base class.

Below we provide a summary of existing reconstruction methods, and a qualitative
description of their reconstruction performance and speed.

.. tip::

      Some methods do not require any training and can be quickly deployed in your problem.

.. tip::

      If you need to train your model and don't have ground truth data,
      the library provides a :ref:`large set of self-supervised losses <self-supervised-losses>`
      which can learn from measurement data alone.


.. list-table:: Reconstruction methods
   :header-rows: 1

   * - **Family of methods**
     - **Description**
     - **Requires Training**
     - **Iterative**
     - **Sampling**
   * - :ref:`Artifact Removal <artifact>`
     - Applies a neural network to a non-learned pseudo-inverse
     - Yes
     - No
     - No
   * - :ref:`Plug-and-Play (PnP) <iterative>`
     - Leverages :ref:`pretrained denoisers <denoisers>` as priors within an optimisation algorithm.
     - No
     - Yes
     - No
   * - :ref:`Unfolded Networks <unfolded>`
     - Constructs a trainable architecture by unrolling a PnP algorithm.
     - Yes
     - Only :class:`DEQ <deepinv.unfolded.DEQ_builder>`
     - No
   * - :ref:`Diffusion <diffusion>`
     - Leverages :ref:`pretrained denoisers <denoisers>` within a ODE/SDE.
     - No
     - Yes
     - Yes
   * - :ref:`Non-learned priors <iterative>`
     - Solves an optimization problem with hand-crafted priors.
     - No
     - Yes
     - No
   * - :ref:`Markov Chain Monte Carlo <mcmc>`
     - Leverages :ref:`pretrained denoisers <denoisers>` as priors within an optimisation algorithm.
     - No
     - Yes
     - Yes
   * - :ref:`Generative Adversarial Networks and Deep Image Prior  <adversarial>`
     - Uses a generator network to model the set of possible images.
     - No
     - Yes
     - Depends
   * - :ref:`Specific network architectures <specific>`
     - Off-the-shelf architectures for specific inverse problems.
     - Yes
     - No
     - No


.. note::

        Some algorithms might be better at reconstructing images with good perceptual quality (e.g. diffusion methods)
        whereas other methods are better at reconstructing images with low distortion (close to the ground truth).

