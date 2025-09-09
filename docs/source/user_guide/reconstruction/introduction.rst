.. _reconstructors:

Introduction
------------
Reconstruction algorithms define an inversion function :math:`\hat{x}=\inversef{y}{A}`
which recovers a signal :math:`x` from measurements :math:`y` given an operator :math:`A`.

.. code-block::

  x_hat = model(y, physics)

.. seealso::
  
  See :ref:`pretrained reconstructors <pretrained-models>` for ready-to-use pretrained reconstruction algorithms
  that you can use to reconstruct images in one line.

Defining your own reconstructor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All reconstruction algorithms inherit from the
:class:`deepinv.models.Reconstructor` base class, take as input measurements `y`
and forward operator `physics`, and output a reconstruction `x_hat`.

To use your own reconstructor with DeepInverse, simply define the `forward` method to follow this pattern.

Summary
~~~~~~~

Below we provide a summary of existing reconstruction methods, and a qualitative
description of their reconstruction performance and speed.

For the models that require training, you can do this using the :ref:`trainer <trainer>` and :ref:`loss functions <loss>`.

.. list-table:: Reconstruction methods
   :header-rows: 1

   * - **Family of methods**
     - **Description**
     - **Requires Training**
     - **Iterative**
     - **Sampling**
   * - :ref:`Deep Reconstruction Models <deep-reconstructors>`
     - Deep model architectures for reconstruction.
     - No if pretrained, yes otherwise
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
   * - :ref:`Multi-physics models <general-reconstructors>`
     - Models trained on multiple various physics and datasets for robustness to different problems.
     - No
     - No
     - No

.. note::

        Some algorithms might be better at reconstructing images with good perceptual quality (e.g. diffusion methods)
        whereas other methods are better at reconstructing images with low distortion (close to the ground truth).

Using models in the cloud
^^^^^^^^^^^^^^^^^^^^^^^^^

The client model :class:`deepinv.models.Client` allows users to perform inference on models hosted in the cloud directly from DeepInverse.

The client allows contributors to disseminate their reconstruction models, without requiring the user to have high GPU resources
or to accurately define their physics. As a contributor, all you have to do is:

  * Define your model to take tensors as input and output tensors (like :class:`deepinv.models.Reconstructor`)
  * Create a simple API
  * Deploy it to the cloud, and distribute the endpoint URL and API keys to anyone who might want to use it!

The user then only needs to define this client, specify the endpoint URL and API key, and pass in an image as a tensor.
The client then performs checks and passes the deserialized tensor to the server for processing.