deepinv.physics
===============

This module provides a set of forward operators for various imaging modalities.
Please refer to the :ref:`physics` section for more details.

Base Classes
------------
.. userguide:: physics_intro

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.Physics
   deepinv.physics.LinearPhysics
   deepinv.physics.DecomposablePhysics
   deepinv.physics.StackedPhysics
   deepinv.physics.StackedLinearPhysics

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.generator.PhysicsGenerator
   deepinv.physics.generator.GeneratorMixture


Operators
---------
.. userguide:: physics_list

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.Denoising
   deepinv.physics.Inpainting
   deepinv.physics.Decolorize
   deepinv.physics.Demosaicing
   deepinv.physics.Blur
   deepinv.physics.BlurFFT
   deepinv.physics.SpaceVaryingBlur
   deepinv.physics.Downsampling
   deepinv.physics.MRI
   deepinv.physics.MRIMixin
   deepinv.physics.DynamicMRI
   deepinv.physics.MultiCoilMRI
   deepinv.physics.SequentialMRI
   deepinv.physics.Tomography
   deepinv.physics.Pansharpen
   deepinv.physics.HyperSpectralUnmixing
   deepinv.physics.CompressedSensing
   deepinv.physics.StructuredRandom
   deepinv.physics.SinglePixelCamera
   deepinv.physics.RadioInterferometry
   deepinv.physics.SinglePhotonLidar
   deepinv.physics.Haze
   deepinv.physics.PhaseRetrieval
   deepinv.physics.RandomPhaseRetrieval
   deepinv.physics.StructuredRandomPhaseRetrieval
   deepinv.physics.Ptychography
   deepinv.physics.PtychographyLinearOperator


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.generator.BernoulliSplittingMaskGenerator
   deepinv.physics.generator.GaussianSplittingMaskGenerator
   deepinv.physics.generator.Phase2PhaseSplittingMaskGenerator
   deepinv.physics.generator.Artifact2ArtifactSplittingMaskGenerator
   deepinv.physics.generator.MotionBlurGenerator
   deepinv.physics.generator.DiffractionBlurGenerator
   deepinv.physics.generator.DiffractionBlurGenerator3D
   deepinv.physics.generator.ProductConvolutionBlurGenerator
   deepinv.physics.generator.BaseMaskGenerator
   deepinv.physics.generator.GaussianMaskGenerator
   deepinv.physics.generator.RandomMaskGenerator
   deepinv.physics.generator.EquispacedMaskGenerator


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.blur.gaussian_blur
   deepinv.physics.blur.bilinear_filter
   deepinv.physics.blur.bicubic_filter
   deepinv.physics.blur.sinc_filter
   deepinv.physics.phase_retrieval.build_probe
   deepinv.physics.phase_retrieval.generate_shifts

Noise distributions
-------------------
.. userguide:: noise_list

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
    deepinv.physics.stack


Functional
----------
.. userguide:: physics_functional

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
