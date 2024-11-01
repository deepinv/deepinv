.. _forward_operators:

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
   deepinv.physics.Demosaicing

For random inpainting we also provide generators to create random masks on-the-fly. These can also be used as splitting masks for :class:`deepinv.loss.SplittingLoss` and its variations.

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


We also provide a set of generators to simulate various types of blur, which can be used to train blind or semi-blind
deblurring networks.

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
In MRI, the Fourier transform is sampled on a grid (FFT) or off-the grid, with a single coil or multiple coils. We provide 2D and 2D+t dynamic MRI physics.

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


Radio interferometric imaging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The radio interferometric imaging operator is implemented in the following class:

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.RadioInterferometry


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
