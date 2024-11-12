deepinv.physics
================


Base Classes
------------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.Physics
   deepinv.physics.LinearPhysics
   deepinv.physics.DecomposablePhysics

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.generator.PhysicsGenerator
   deepinv.physics.generator.GeneratorMixture


Operators
---------------

Pixelwise operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.Denoising
   deepinv.physics.Inpainting
   deepinv.physics.Decolorize
   deepinv.physics.Demosaicing

For random inpainting we also provide masks generators:

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.generator.BernoulliSplittingMaskGenerator
   deepinv.physics.generator.GaussianSplittingMaskGenerator
   deepinv.physics.generator.Phase2PhaseSplittingMaskGenerator
   deepinv.physics.generator.Artifact2ArtifactSplittingMaskGenerator

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
   deepinv.physics.blur.sinc_filter


We also provide a set of generators to simulate various types of blur:

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.generator.MotionBlurGenerator
   deepinv.physics.generator.DiffractionBlurGenerator
   deepinv.physics.generator.DiffractionBlurGenerator3D
   deepinv.physics.generator.ProductConvolutionBlurGenerator

Magnetic Resonance Imaging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In MRI, the Fourier transform is sampled on a grid (FFT) or off-the grid, with a single coil or multiple coils.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.MRI
   deepinv.physics.DynamicMRI
   deepinv.physics.SequentialMRI


We provide generators for creating random and non-random acceleration masks using Cartesian sampling, for both static (k) and dynamic (k-t) accelerated MRI:

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.generator.BaseMaskGenerator
   deepinv.physics.generator.GaussianMaskGenerator
   deepinv.physics.generator.RandomMaskGenerator
   deepinv.physics.generator.EquispacedMaskGenerator

Tomography
^^^^^^^^^^

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.Tomography



Remote Sensing
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.Pansharpen


Compressive operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.CompressedSensing
   deepinv.physics.SinglePixelCamera


Radio interferometric imaging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.RadioInterferometry


Single-photon lidar
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.SinglePhotonLidar


Dehazing
^^^^^^^^^^^^^

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
   deepinv.physics.generator.SigmaGenerator


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.physics.TimeMixin
    deepinv.physics.adjoint_function


Functional
--------------------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.functional.conv2d
   deepinv.physics.functional.conv_transpose2d
   deepinv.physics.functional.conv2d_fft
   deepinv.physics.functional.conv_transpose2d_fft
   deepinv.physics.functional.conv3d_fft
   deepinv.physics.functional.conv_transpose3d_fft
   deepinv.physics.functional.product_convolution2d
   deepinv.physics.functional.multiplier
   deepinv.physics.functional.multiplier_adjoint
   deepinv.physics.functional.Radon
   deepinv.physics.functional.IRadon
   deepinv.physics.functional.histogramdd
   deepinv.physics.functional.histogram
