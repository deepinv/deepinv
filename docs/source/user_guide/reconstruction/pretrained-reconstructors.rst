.. _pretrained-reconstructors:

Pretrained Reconstructors
~~~~~~~~~~~~~~~~~~~~~~~~~

Some methods do not require any training can be quickly deployed to your problem.

.. seealso::

  See the example :ref:`sphx_glr_auto_examples_basics_demo_pretrained_model.py` for how to get started with these models on various problems.

These models can be defined in one line and inferenced in one line:

.. doctest::

  >>> import deepinv as dinv
  >>> x = dinv.utils.load_example("butterfly.png")
  >>> physics = dinv.physics.Downsampling()
  >>> y = physics(x)
  >>> model = dinv.models.ArtifactRemoval(dinv.models.MedianFilter()) # TODO dinv.models.RAM(pretrained=True) # or any of the models listed below
  >>> x_hat = model(y, physics) # Model inference
  >>> dinv.metric.PSNR()(x_hat, x) # TODO CHANGE MEDIANFILTER TO RAM BELOW TOO
  tensor([5.5290])

.. list-table:: Pretrained reconstructors
   :header-rows: 1

   * - **Name**
     - **Family**
     - **Modality**
     - **Speed**
   * - :class:`Reconstruct Anything Model <deepinv.models.MedianFilter>`
     - Feedforward
     - General
     - Fast
   * - :ref:`Plug-and-play <iterative>` with a pretrained denoiser
     - Iterative
     - General
     - Medium
   * - :ref:`Diffusion model <diffusion>` with a pretrained denoiser
     - Sampling
     - General
     - Slow

.. tip::

  If you want to fine-tune these models on your own measurements (without or with ground truth) or physics, check out :ref:`sphx_glr_auto_examples_models_demo_finetuning.py`.

.. seealso::

  See :ref:`pretrained denoisers <pretrained-weights>` for a full list of denoisers that can be plugged into iterative/sampling algorithms.