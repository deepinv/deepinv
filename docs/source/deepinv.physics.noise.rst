.. _noise_distributions:

Noise distributions
--------------------------------
Noise mappings :math:`N:\yset\mapsto \yset` are simple :class:`torch.nn.Module`.
The noise of a forward operator can be set in its construction
or simply as

.. doctest::

    >>> import torch
    >>> import deepinv as dinv
    >>> # load a CS operator with 300 measurements, acting on 28 x 28 grayscale images.
    >>> physics = dinv.physics.CompressedSensing(m=300, img_shape=(1, 28, 28))
    >>> physics.noise_model = dinv.physics.GaussianNoise(sigma=.05) # set up the noise


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.GaussianNoise
   deepinv.physics.LogPoissonNoise
   deepinv.physics.PoissonNoise
   deepinv.physics.PoissonGaussianNoise
   deepinv.physics.UniformNoise
   deepinv.physics.UniformGaussianNoise
   deepinv.physics.GammaNoise


The parameters of noise distributions can also be created from a :meth:`deepinv.physics.generator.PhysicsGenerator`,
which is useful for training and evaluating methods under various noise conditions.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.generator.SigmaGenerator

