.. _physics:

Physics
=======

Introduction
---------------

This package contains a large collection of forward operators appearing in imaging applications.
The acquisition models are of the form

.. math::

    y = \noise{\forw{x}}

where :math:`x\in\xset` is an image, :math:`y\in\yset` are the measurements, :math:`A:\xset\mapsto \yset` is a
deterministic (linear or non-linear) operator capturing the physics of the acquisition and
:math:`N:\yset\mapsto \yset` is a mapping which characterizes the noise affecting the measurements.


All forward operators inherit the structure of the :meth:`Physics` class:

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.Physics

They are :class:`torch.nn.Module` which can be called with the ``forward`` method.


.. doctest::

    >>> import torch
    >>> import deepinv as dinv
    >>> # load an inpainting operator that masks 50% of the pixels and adds Gaussian noise
    >>> physics = dinv.physics.Inpainting(mask=.5, tensor_size=(1, 28, 28),
    ...                    noise_model=dinv.physics.GaussianNoise(sigma=.05))
    >>> x = torch.rand(1, 1, 28, 28) # create a random image
    >>> y = physics(x) # compute noisy measurements
    >>> y2 = physics.A(x) # compute the A operator (no noise)

Linear operators
^^^^^^^^^^^^^^^^^^^^^^^^^

Linear operators :math:`A:\xset\mapsto \yset` inherit the structure of the :meth:`deepinv.physics.LinearPhysics` class.
They have important specific properties such as the existence of an adjoint :math:`A^*:\yset\mapsto \xset`. 
Linear operators with a closed-form singular value decomposition are defined via :meth:`deepinv.physics.DecomposablePhysics`,
which enables the efficient computation of their pseudo-inverse and regularized inverse.
Composition and linear combinations of linear operators is still a linear operator.

.. doctest::

    >>> import torch
    >>> import deepinv as dinv
    >>> # load a CS operator with 300 measurements, acting on 28 x 28 grayscale images.
    >>> physics = dinv.physics.CompressedSensing(m=300, img_shape=(1, 28, 28))
    >>> x = torch.rand(1, 1, 28, 28) # create a random image
    >>> y = physics(x) # compute noisy measurements
    >>> y2 = physics.A(x) # compute the linear operator (no noise)
    >>> x_adj = physics.A_adjoint(y) # compute the adjoint operator
    >>> x_dagger = physics.A_dagger(y) # compute the pseudo-inverse operator
    >>> x_prox = physics.prox_l2(x, y, .1) # compute a regularized inverse

More details can be found in the doc of each class:

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.LinearPhysics
   deepinv.physics.DecomposablePhysics


Parameter-dependent operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many (linear or non-linear) operators depend on (optional) parameters :math:`\theta` that describe the imaging system, ie
:math:`y = \noise{\forw{x, \theta}}` where
the ``forward`` method can be called with a dictionary of parameters as an extra input. The explicit dependency on
:math:`\theta` is often useful for blind inverse problems, model identification, imaging system optimization, etc.
The following example shows how operators and their parameter can be instantiated and called as:

.. doctest::

   >>> import torch
   >>> from deepinv.physics import Blur
   >>> x = torch.rand((1, 1, 16, 16))
   >>> theta = torch.ones((1, 1, 2, 2)) / 4 # a basic 2x2 averaging filter
   >>> # default usage
   >>> physics = Blur(filter=theta) # we instantiate a blur operator with its convolution filter
   >>> y = physics(x)
   >>>
   >>> # A second possibility
   >>> physics = Blur() # a blur operator without convolution filter
   >>> y = physics(x, filter=theta) # we define the blur by specifying its filter
   >>> y = physics(x) # now, the filter is well-defined and this line does the same as above
   >>>
   >>> # The same can be done by passing in a dictionary including 'filter' as a key
   >>> physics = Blur() # a blur operator without convolution filter
   >>> dict_params = {'filter': theta, 'dummy': None}
   >>> y = physics(x, **dict_params) # # we define the blur by passing in the dictionary



Physics Generators
^^^^^^^^^^^^^^^^^^^
We provide some parameters generation methods to sample random parameters' :math:`\theta`.
Physics generators inherit from the :meth:`PhysicsGenerator` class:

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.generator.PhysicsGenerator

.. doctest::

    >>> import torch
    >>> import deepinv as dinv
    >>>
    >>> x = torch.rand((1, 1, 8, 8))
    >>> physics = dinv.physics.Blur(filter=dinv.physics.blur.gaussian_blur(.2))
    >>> y = physics(x) # compute with Gaussian blur
    >>> generator = dinv.physics.generator.MotionBlurGenerator(psf_size=(3, 3))
    >>> params = generator.step(x.size(0)) # params = {'filter': torch.tensor(...)}
    >>> y1 = physics(x, **params) # compute with motion blur
    >>> assert not torch.allclose(y, y1) # different blurs, different outputs
    >>> y2 = physics(x) # motion kernel is stored in the physics object as default kernel
    >>> assert torch.allclose(y1, y2) # same blur, same output

If we want to generate both a new physics and noise parameters,
it is possible to sum generators as follows:

.. doctest::

    >>> mask_generator = dinv.physics.generator.SigmaGenerator() \
    ...    + dinv.physics.generator.AccelerationMaskGenerator((32, 32))
    >>> params = mask_generator.step(batch_size=4)
    >>> print(sorted(params.keys()))
    ['mask', 'sigma']

It is also possible to mix generators of physics parameters through the :meth:`GeneratorMixture` class:

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.generator.GeneratorMixture


Forward operators
--------------------

Various popular forward operators are provided with efficient implementations.

Pixelwise operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Pixelwise operators operate in the pixel domain and are used for denoising, inpainting, decolorization, etc.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.Denoising
   deepinv.physics.Inpainting
   deepinv.physics.Decolorize

Blur & Super-Resolution
^^^^^^^^^^^^^^^^^^^^^^^^
Different types of blur operators are available, from simple stationary kernels to space-varying ones.

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

Compressive operators are implemented in the following classes:

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

Phase retrieval
^^^^^^^^^^^^^^^^^^^^^^^^^
Operators where :math:`A:\xset\mapsto \yset` is of the form :math:`A(x) = |Bx|^2` with :math:`B` a linear operator.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.PhaseRetrieval
   deepinv.physics.RandomPhaseRetrieval

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


The parameters of noise distributions can also be created from a :meth:`deepinv.physics.generator.PhysicsGenerator`,
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


Functional
--------------------

The toolbox is based on efficient PyTorch implementations of basic operations such as diagonal multipliers, Fourier transforms, convolutions, product-convolutions, Radon transform, interpolation mappings.
Similar to the PyTorch structure, they are available within :py:mod:`deepinv.physics.functional`.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.functional.conv2d
   deepinv.physics.functional.conv_transpose2d
   deepinv.physics.functional.conv2d_fft
   deepinv.physics.functional.conv_transpose2d_fft
   deepinv.physics.functional.conv3d
   deepinv.physics.functional.conv_transpose3d
   deepinv.physics.functional.product_convolution2d
   deepinv.physics.functional.multiplier
   deepinv.physics.functional.multiplier_adjoint
   deepinv.physics.functional.Radon
   deepinv.physics.functional.IRadon
   deepinv.physics.functional.histogramdd
   deepinv.physics.functional.histogram

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
    tensor([[[[0.2500, 0.2500, 0.2500],
              [0.2500, 0.2500, 0.2500],
              [0.2500, 0.2500, 0.2500]]]])
    >>>
    >>> _ = torch.manual_seed(0)
    >>> y = torch.randn_like(Ax)
    >>> z = dinv.physics.functional.conv_transpose2d(y, filter, padding)
    >>> print((Ax * y).sum(dim=(1, 2, 3)) - (x * z).sum(dim=(1, 2, 3)))
    tensor([5.9605e-08])