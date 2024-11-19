.. _physics:

Operators & Noise
=================


.. _physics_list:

Operators
~~~~~~~~~
Operators describe the forward model :math:`z = A(x,\theta)`.

.. list-table:: Operators, Definitions, and Generators
   :header-rows: 1

   * - **Family**
     - **Operators**
     - **Generators**

   * - Pixelwise
     - | :class:`deepinv.physics.Denoising`
       | :class:`deepinv.physics.Inpainting`
       | :class:`deepinv.physics.Demosaicing`
       | :class:`deepinv.physics.Decolorize`
     - | :class:`BernoulliSplittingMaskGenerator <deepinv.physics.generator.BernoulliSplittingMaskGenerator>`
       | :class:`GaussianSplittingMaskGenerator <deepinv.physics.generator.GaussianSplittingMaskGenerator>`
       | :class:`Phase2PhaseSplittingMaskGenerator <deepinv.physics.generator.Phase2PhaseSplittingMaskGenerator>`
       | :class:`Artifact2ArtifactSplittingMaskGenerator <deepinv.physics.generator.Artifact2ArtifactSplittingMaskGenerator>`

   * - Blur & Super-Resolution
     - | :class:`deepinv.physics.Blur`
       | :class:`deepinv.physics.BlurFFT`
       | :class:`deepinv.physics.SpaceVaryingBlur`
       | :class:`deepinv.physics.Downsampling`
     - | :class:`MotionBlurGenerator <deepinv.physics.generator.MotionBlurGenerator>`
       | :class:`DiffractionBlurGenerator <deepinv.physics.generator.DiffractionBlurGenerator>`
       | :class:`ProductConvolutionBlurGenerator <deepinv.physics.generator.ProductConvolutionBlurGenerator>`


   * - Magnetic Resonance Imaging (MRI)
     - | :class:`deepinv.physics.MRI`
       | :class:`deepinv.physics.DynamicMRI`
       | :class:`deepinv.physics.SequentialMRI`
     - | :class:`GaussianMaskGenerator <deepinv.physics.generator.GaussianMaskGenerator>`
       | :class:`RandomMaskGenerator <deepinv.physics.generator.RandomMaskGenerator>`
       | :class:`EquispacedMaskGenerator <deepinv.physics.generator.EquispacedMaskGenerator>`

   * - Tomography
     - :class:`deepinv.physics.Tomography`
     -

   * - Remote Sensing
     - :class:`deepinv.physics.Pansharpen`
     -

   * - Compressive
     - | :class:`deepinv.physics.CompressedSensing`
       | :class:`deepinv.physics.SinglePixelCamera`
     -

   * - Radio Interferometric Imaging
     - :class:`deepinv.physics.RadioInterferometry`
     -

   * - Single-Photon Lidar
     - :class:`deepinv.physics.SinglePhotonLidar`
     -

   * - Dehazing
     - :class:`deepinv.physics.Haze`
     -

   * - Phase Retrieval
     - | :class:`deepinv.physics.PhaseRetrieval`
       | :class:`deepinv.physics.RandomPhaseRetrieval`
     -


.. _noise_list:

Noise distributions
~~~~~~~~~~~~~~~~~~~
Noise distributions describe the noise model :math:`N`,
where :math:`y = N(z)` with :math:`z=A(x)`. The noise models can be assigned
to **any** operator in the list above, by setting the ``noise_model`` attribute at initialization.

.. list-table:: Noise Distributions and Their Probability Distributions
   :header-rows: 1

   * - **Noise**
     - :math:`p(y|z)`

   * - :class:`deepinv.physics.GaussianNoise`
     - :math:`\mathcal{N}(z, I\sigma^2)`

   * - :class:`deepinv.physics.PoissonNoise`
     - :math:`\mathcal{P}(y/\lambda)`

   * - :class:`deepinv.physics.PoissonGaussianNoise`
     - :math:`y = \log \bar{z} + \epsilon`, :math:`\epsilon \sim \mathcal{N}(0, I\sigma^2)`, :math:`\bar{z} \sim \mathcal{P}(z/\lambda)`

   * - :class:`deepinv.physics.LogPoissonNoise`
     - :math:`y = \gamma \bar{z}`, :math:`\bar{z} \sim \mathcal{P}(z/\lambda)`

   * - :class:`deepinv.physics.UniformNoise`
     - :math:`\mathcal{U}(z-a, z+b)`


