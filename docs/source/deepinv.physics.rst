.. _physics:

Physics
=======

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


Operators can be called with the ``forward`` method, for example

.. exec_code::

    import torch
    import deepinv as dinv

    # load a CS operator with 300 measurements, acting on 28 x 28 grayscale images.
    physics = dinv.physics.CompressedSensing(m=300, img_shape=(1, 28, 28))
    x = torch.rand(1, 1, 28, 28) # create a random image
    y = physics(x) # compute noisy measurements

Linear operators
----------------
Operators where :math:`A:\xset\mapsto \yset` is a linear mapping.
All linear operators inherit the structure of the :class:`LinearPhysics` class.
Linear operators with a closed-form singular value decomposition are defined via :class:`DecomposablePhysics`,
which enables the efficient computation of their pseudo-inverse and proximal operators.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.LinearPhysics
   deepinv.physics.DecomposablePhysics
   deepinv.physics.Blur
   deepinv.physics.BlurFFT
   deepinv.physics.CompressedSensing
   deepinv.physics.Decolorize
   deepinv.physics.Denoising
   deepinv.physics.Downsampling
   deepinv.physics.MRI
   deepinv.physics.Inpainting
   deepinv.physics.SinglePixelCamera
   deepinv.physics.Tomography
   deepinv.physics.Pansharpen

All linear operators have adjoint, pseudo-inverse and prox functions (and more) which can be called as

.. exec_code::

    import torch
    import deepinv as dinv

    # load a CS operator with 300 measurements, acting on 28 x 28 grayscale images.
    physics = dinv.physics.CompressedSensing(m=300, img_shape=(1, 28, 28))
    x = torch.rand(1, 1, 28, 28) # create a random image
    y = physics(x) # compute noisy measurements
    y2 = physics.A(x) # compute the linear operator (no noise)
    x_adj = physics.A_adjoint(y) # compute the adjoint operator
    x_dagger = physics.A_dagger(y) # compute the pseudo-inverse operator
    x_prox = physics.prox_l2(x, y, .1) # compute the prox operator

Some operators have singular value decompositions (see :class:`deepinv.physics.DecomposablePhysics`) which
have additional methods.

Non-linear operators
--------------------
Operators where :math:`A:\xset\mapsto \yset` is a non-linear mapping (e.g., bilinear).

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.BlindBlur
   deepinv.physics.Haze
   deepinv.physics.SinglePhotonLidar

Noise distributions
-------------------
Noise mappings :math:`N:\yset\mapsto \yset` are simple :class:`torch.nn.Module`.
The noise of a forward operator can be set in its construction
or simply as

.. exec_code::

    import torch
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
   deepinv.physics.UniformGaussianNoise

Utils
-------------
This module also contains some utilities for physics operators.

.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

   deepinv.physics.blur.gaussian_blur