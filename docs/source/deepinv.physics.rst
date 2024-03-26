.. _physics:

Physics
=========

This package contains a large collection of forward operators appearing in imaging applications.

The operators and are of the form

.. math::

    y = \noise{\forw{x}}

where :math:`x\in\xset` is an image of :math:`n` pixels, :math:`y\in\yset` are the measurements of size :math:`m`,
:math:`A:\xset\mapsto \yset` is a deterministic (linear or non-linear) mapping capturing the physics of the acquisition
and :math:`N:\yset\mapsto \yset` is a mapping which characterizes the noise affecting the measurements.

Operators can be called with the ``forward`` method, for example

.. exec_code::

    import torch
    import deepinv as dinv

    # load a CS operator with 300 measurements, acting on 28 x 28 grayscale images.
    physics = dinv.physics.CompressedSensing(m=300, img_shape=(1, 28, 28))
    x = torch.rand(1, 1, 28, 28) # create a random image
    y = physics(x) # compute noisy measurements


Introduction
------------

All forward operators inherit the structure of the ``Physics`` class.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.Physics

Operators where :math:`A:\xset\mapsto \yset` is a linear mapping.
All linear operators inherit the structure of the :class:`LinearPhysics` class.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.LinearPhysics

Linear operators with a closed-form singular value decomposition are defined via :class:`DecomposablePhysics`,
which enables the efficient computation of their pseudo-inverse and proximal operators.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.DecomposablePhysics



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



Generators
^^^^^^^^^^^
The generators are used to

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.PhysicsGenerator
   deepinv.physics.GeneratorMixture

.. exec_code::

    import torch
    import deepinv as dinv

    x = torch.rand((1, 1, 32, 32))
    physics = dinv.physics.Blur(filter=dinv.physics.blur.gaussian_blur(1))
    y = physics(x) # compute with Gaussian blur
    generator = dinv.physics.generator.MotionBlurGenerator((1, 5, 5))
    kernel = generator.step() # generate new motion blur kernel
    y1 = physics(x, kernel) # compute with motion blur
    y2 = physics(x) # motion kernel is stored in the physics object as default kernel
    assert torch.all_close(y1, y2) # same result

Applications
------------

Various popular forward operators are provided with state-of-the-art implementations.

Diagonal operators
^^^^^^^^^^^^^^^^^^
Diagonal operators operate in the pixel domain and are used for denoising, inpainting, decolorization, etc.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.Denoising
   deepinv.physics.Inpainting
   deepinv.physics.Decolorize

Blur & Super-Resolution
^^^^^^^^^^^^^^^^^^^^^^^^
Different types of blur operators are available.
They can be stationary (convolutions) or space-varying. Also, we integrated super-resolution applications by composing blurs with downsampling.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.Blur
   deepinv.physics.BlurFFT
   deepinv.physics.SpaceVaryingBlur
   deepinv.physics.Downsampling

We provide the implementation of typical blur kernels such as Gaussian, bilinear, bicubic, etc.

.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

   deepinv.physics.blur.gaussian_blur
   deepinv.physics.blur.bilinear_filter
   deepinv.physics.blur.bicubic_filter


We also provide a set of generators to simulate various types of blur, which can be used to train blind or semi-blind
deblurring networks.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.generator.MotionBlurGenerator
   deepinv.physics.generator.DiffractionBlurGenerator

Magnetic Resonance Imaging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In MRI, the Fourier transform is sampled on a grid (FFT) or off-the grid, with a single coil or multiple coils.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.MRI


We provide generators for sampling acceleration masks:

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.generator.AccelerationMaskGenerator

Tomography 
^^^^^^^^^^

Tomography is based on the Radon-transform which computes line-integrals. 

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.Tomography



Remote Sensing
^^^^^^^^^^^^^^^^
Remote sensing operators are used to simulate the acquisition of satellite data.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.Pansharpen


Compressive operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The field of compressed sensing initially suggested to use white Gaussian or Bernoulli random vectors.
These operators are implemented in the following functions.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.CompressedSensing
   deepinv.physics.SinglePixelCamera


Single-photon lidar
^^^^^^^^^^^^^^^^^^^^^^^
Single-photon lidar is a popular technique for depth ranging and imaging.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.SinglePhotonLidar


Dehazing
^^^^^^^^^^^^^
Haze operators are used to capture the physics of light scattering in the atmosphere.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.Haze


Noise distributions
===================
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
   deepinv.physics.LogPoissonNoise
   deepinv.physics.PoissonNoise
   deepinv.physics.PoissonGaussianNoise
   deepinv.physics.UniformNoise
   deepinv.physics.UniformGaussianNoise




Defining new operators
--------------------------------

When defining a new linear operator, you can define the adjoint automatically using autograd with

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.physics.adjoint_function