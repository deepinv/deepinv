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

Operators are :meth:`torch.nn.Module` which can be called with the ``forward`` method, for example

.. doctest::

    >>> import torch
    >>> import deepinv as dinv
    >>>
    >>> # load a CS operator with 300 measurements, acting on 28 x 28 grayscale images.
    >>> physics = dinv.physics.CompressedSensing(m=300, img_shape=(1, 28, 28),
    >>>                                          noise_model=dinv.physics.GaussianNoise(sigma=.05))
    >>> x = torch.rand(1, 1, 28, 28) # create a random image
    >>> y = physics(x) # compute noisy measurements


Introduction
------------

All forward operators inherit the structure of the :class:`Physics` class.

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

.. doctest::

    >>> import torch
    >>> import deepinv as dinv
    >>>
    >>> # load a CS operator with 300 measurements, acting on 28 x 28 grayscale images.
    >>> physics = dinv.physics.CompressedSensing(m=300, img_shape=(1, 28, 28))
    >>> x = torch.rand(1, 1, 28, 28) # create a random image
    >>> y = physics(x) # compute noisy measurements
    >>> y2 = physics.A(x) # compute the linear operator (no noise)
    >>> x_adj = physics.A_adjoint(y) # compute the adjoint operator
    >>> x_dagger = physics.A_dagger(y) # compute the pseudo-inverse operator
    >>> x_prox = physics.prox_l2(x, y, .1) # compute the prox operator

Some operators have singular value decompositions (see :class:`deepinv.physics.DecomposablePhysics`) which
have additional methods.



Generators
^^^^^^^^^^^
Physics generators are used to sample new parameters for the forward operator (:math:`N(\cdot)` or :math:`A(\cdot)`). They are useful in inverse problems
where the forward operator is unknown or partially known, eg as in blind deconvolution, or in problems where the forward
operator might change from image to image in a known way, eg MRI with varying acceleration masks.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.generator.PhysicsGenerator
   deepinv.physics.generator.MotionBlurGenerator
   deepinv.physics.generator.DiffractionBlurGenerator
   deepinv.physics.generator.AccelerationMaskGenerator
   deepinv.physics.generator.SigmaGenerator

   
.. doctest::

    >>> import torch
    >>> import deepinv as dinv
    >>>
    >>> x = torch.rand((1, 1, 8, 8))
    >>> physics = dinv.physics.Blur(filter=dinv.physics.blur.gaussian_blur(1))
    >>> y = physics(x) # compute with Gaussian blur
    >>> generator = dinv.physics.generator.MotionBlurGenerator((1, 3, 3))
    >>> kernel = generator.step(x.size(0)) # generate a motion blur kernel
    >>> y1 = physics(x, **kernel) # compute with motion blur
    >>> assert not torch.allclose(y, y1)
    >>> y2 = physics(x) # motion kernel is stored in the physics object as default kernel
    >>> assert torch.allclose(y1, y2)


If at each iteration ones wants to generate both a new physics parameter and noise parameters,
one can add the physics and noise generators as follows to sample new parameters for 
the full forward operator :math:`N(A(x))`
    
.. doctest::  

    >>> mask_generator = dinv.physics.generator.SigmaGenerator() \
    >>>    + dinv.physics.generator.AccelerationMaskGenerator((32, 32))
    >>> params = mask_generator.step(4)
    >>> print(params)

    

For blind inverse problems where it can be useful to train on multiple families of operators, generators can be
mixed through the GeneratorMixture class that samples randomly from one of the mixed :class:`deepinv.physics.generator.PhysicsGenerator`
object passed as input with probabilities probs

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.generator.GeneratorMixture

.. doctest::

    >>> from deepinv.physics.generator import MotionBlurGenerator, DiffractionBlurGenerator
    >>> g1 = MotionBlurGenerator((1, 1, 3, 3))
    >>> g2 = DiffractionBlurGenerator((1, 1, 3, 3))
    >>> generator = GeneratorMixture([g1, g2], [0.5, 0.5])
    >>> params_dict = generator.step(batch_size=1)
        
    
    
Functionals
^^^^^^^^^^^
For each :class:`deepinv.physics.Physics` object, the forward operation is performed under the hood 
using the functions from :module: `deepinv.physics.functional`

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.functional.conv2d
   deepinv.physics.functional.conv_transpose2d
   deepinv.physics.functional.Radon
   deepinv.physics.functional.IRadon
   deepinv.physics.functional.histogramdd
   deepinv.physics.functional.ThinPlateSpline

.. doctest::

    >>> import torch
    >>> import deepinv as dinv
    
    >>> x = torch.zeros((1, 1, 16, 16)) # Define black image of size 16x16
    >>> x[:, :, 8, 8] = 1 # Define one white pixel in the middle
    >>> filter = torch.ones((1, 1, 3, 3)) / 4
    >>>
    >>> padding = "circular"
    >>> Ax = dinv.physics.functional.conv2d(x, filter, padding)
    >>> print(Ax[:, :, 7:10, 7:10])
    tensor([[[[0.2500, 0.2500, 0.0000],
          [0.2500, 0.2500, 0.0000],
          [0.0000, 0.0000, 0.0000]]]])
    >>>      
    >>> torch.manual_seed(0)
    >>> y = torch.randn_like(Ax)
    >>> z = dinv.physics.functional.conv_transpose2d(y, filter, padding)
    >>> print((Ax * y).sum(dim=(1, 2, 3)) - (x * z).sum(dim=(1, 2, 3)))
    tensor([5.9605e-08])

    

Forward operators
--------------------

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

   deepinv.physics.generator.PSFGenerator
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


The parameters of noise distributions can also be created from a :meth:`deepinv.physics.PhysicsGenerator`,
which is useful for training and evaluating methods under various noise conditions.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.generator.SigmaGenerator


Defining new operators
--------------------------------

Defining a new forward operator is relatively simple. You need to create a new class that inherits from the right
physics class, that is :meth:`deepinv.physics.Physics` for non-linear operators,
:meth:`deepinv.physics.LinearPhysics` for linear operators and :meth:`deepinv.physics.DecomposablePhysics`
for linear operators with a closed-form singular value decomposition. The only requirement is to define
a :class:`deepinv.physics.Physics.A` method that computes the forward operator. See the
example :ref:`sphx_glr_auto_examples_basics_demo_physics.py` for more details.

Defining a new linear operator requires the definition of :class:`deepinv.physics.LinearPhysics.A_adjoint`,
you can define the adjoint automatically using autograd with

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.physics.adjoint_function

Note however that coding a closed form adjoint is generally more efficient.