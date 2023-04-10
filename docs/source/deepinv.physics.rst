Physics
===============================

This package contains a large collection of forward operators appearing in imaging applications.

The operators and are of the form

.. math::

    y = \noise{\forw{x}}

where :math:`x\in\xset` is an image of :math:`n` pixels, :math:`y\in\yset` are the measurements of size :math:`m`,
:math:`A:\xset\mapsto \yset` is a deterministic (linear or non-linear) mapping capturing the physics of the acquisition
and :math:`N:\yset\mapsto \yset` is a stochastic mapping which characterizes the noise affecting the measurements.


All forward operators inherit the structure of the ``Physics`` class.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.Physics

Linear operators
-------------------------------------
Operators where :math:`A:\xset\mapsto \yset` is a linear mapping.
All linear operators inherit the structure of the ``LinearPhysics`` class.
Linear operators with a closed-form singular value decomposition are defined via ``DecomposablePhysics``, which enables
the efficient computation of their pseudo-inverse and proximal operators.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.LinearPhysics
   deepinv.physics.DecomposablePhysics
   deepinv.physics.Inpainting
   deepinv.physics.Denoising
   deepinv.physics.Blur
   deepinv.physics.BlurFFT
   deepinv.physics.Downsampling
   deepinv.physics.CompressedSensing
   deepinv.physics.Decolorize


Non-linear operators
-------------------------------------
Operators where :math:`A:\xset\mapsto \yset` is a non-linear mapping (e.g., bilinear).

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.BlindBlur
   deepinv.physics.Haze

Noise distributions
-------------------------------------
Noise mappings :math:`N:\yset\mapsto \yset` are simple ``torch.nn.Module`` s.
The noise of a forward operator can be set in its construction
or simply as

::

    import deepinv as dinv

    # load a CS operator with 300 measurements, acting on 28 x 28 grayscale images.
    physics = dinv.physics.CompressedSensing(m=300, img_shape=(1, 28, 28))
    physics.noise_model = dinv.physics.GaussianNoise(sigma=.05) # set up the noise


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.GaussianNoise
   deepinv.physics.PoissonNoise
   deepinv.physics.PoissonGaussianNoise
   deepinv.physics.UniformNoise


Defining a new (linear) operator only requires a forward function and its transpose operation,
inheriting the remaining structure of the ``LinearPhysics`` class:

::

    import deepinv.physics.Physics as LinearPhysics

    # define an operator that converts color images into grayscale ones.
    class Decolorize(LinearPhysics):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def A(self, x):
            y = x[:, 0, :, :] * 0.2989 + x[:, 1, :, :] * 0.5870 + x[:, 2, :, :] * 0.1140
            return y.unsqueeze(1)

        def A_adjoint(self, y):
            return torch.cat([y*0.2989, y*0.5870, y*0.1140], dim=1)

.. note::

    If the operator is linear, it is recommended to verify that the transpose well defined using
    :meth:`deepinv.physics.LinearPhysics.adjointness_test()`,
    and that it has a unit norm using :meth:`deepinv.physics.LinearPhysics.compute_norm()`

    ::

        my_operator = Decolorize()
        norm = my_operator.compute_norm()
        if my_operator.adjointness_test()<1e-5 and .5 < norm < 1.5
            print('the operator has a well defined transpose and is well normalized!')

