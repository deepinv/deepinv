.. _physics:

Operators & Noise
=================


.. _physics_list:

Operators
~~~~~~~~~
Operators describe the forward model :math:`z = A(x,\theta)`, where
:math:`x` is the input image and :math:`\theta` are the parameters of the operator.
The parameters :math:`\theta` can be sampled using random generators, which are available for some specific classes.
Using automatic differentiation, we can compute derivatives w.r.t to both the input :math:`x` or the parameters :math:`\theta`. 
This is particular useful when dealing with blind inverse problems or parameter estimation.

.. tip::
  The operator you are looking for is not on this list?
  **See** :ref:`sphx_glr_auto_examples_basics_demo_custom_physics.py` for how to implement your own physics operator.

.. list-table:: Operators, Definitions, and Generators
   :header-rows: 1

   * - **Family**
     - **Operators**
     - **Generators**

   * - Pixelwise
     -
       | :class:`deepinv.physics.Denoising`
       | :class:`deepinv.physics.Inpainting`
       | :class:`deepinv.physics.Demosaicing`
       | :class:`deepinv.physics.Decolorize`
     -
       | :class:`BernoulliSplittingMaskGenerator <deepinv.physics.generator.BernoulliSplittingMaskGenerator>`
       | :class:`GaussianSplittingMaskGenerator <deepinv.physics.generator.GaussianSplittingMaskGenerator>`
       | :class:`MultiplicativeSplittingMaskGenerator <deepinv.physics.generator.MultiplicativeSplittingMaskGenerator>`
       | :class:`Phase2PhaseSplittingMaskGenerator <deepinv.physics.generator.Phase2PhaseSplittingMaskGenerator>`
       | :class:`Artifact2ArtifactSplittingMaskGenerator <deepinv.physics.generator.Artifact2ArtifactSplittingMaskGenerator>`

   * - Blur & Super-Resolution
     -
       | :class:`deepinv.physics.Blur`
       | :class:`deepinv.physics.BlurFFT`
       | :class:`deepinv.physics.SpaceVaryingBlur`
       | :class:`deepinv.physics.Downsampling`
       | :class:`deepinv.physics.Upsampling`
       | :class:`deepinv.physics.DownsamplingMatlab`
     -
       | :class:`MotionBlurGenerator <deepinv.physics.generator.MotionBlurGenerator>`
       | :class:`DownsamplingGenerator <deepinv.physics.generator.DownsamplingGenerator>`
       | :class:`DiffractionBlurGenerator <deepinv.physics.generator.DiffractionBlurGenerator>`
       | :class:`ProductConvolutionBlurGenerator <deepinv.physics.generator.ProductConvolutionBlurGenerator>`
       | :class:`ConfocalBlurGenerator3D <deepinv.physics.generator.ConfocalBlurGenerator3D>`
       | :class:`gaussian_blur <deepinv.physics.blur.gaussian_blur>`, :class:`sinc_filter <deepinv.physics.blur.sinc_filter>`
       | :class:`bilinear_filter <deepinv.physics.blur.bilinear_filter>`, :class:`bicubic_filter <deepinv.physics.blur.bicubic_filter>`

   * - Magnetic Resonance Imaging (MRI)
     -
       | :class:`deepinv.physics.MRI`
       | :class:`deepinv.physics.MultiCoilMRI`
       | :class:`deepinv.physics.DynamicMRI`
       | :class:`deepinv.physics.SequentialMRI`
       | The above all also natively support 3D MRI.
     -
       | :class:`GaussianMaskGenerator <deepinv.physics.generator.GaussianMaskGenerator>`
       | :class:`RandomMaskGenerator <deepinv.physics.generator.RandomMaskGenerator>`
       | :class:`EquispacedMaskGenerator <deepinv.physics.generator.EquispacedMaskGenerator>`
       | :class:`PolyOrderMaskGenerator <deepinv.physics.generator.PolyOrderMaskGenerator>`
       | The above all also support k+t dynamic sampling.

   * - Tomography
     -
       | :class:`deepinv.physics.Tomography`
       | :class:`deepinv.physics.TomographyWithAstra`
     -

   * - Remote Sensing & Multispectral
     -
       | :class:`deepinv.physics.Pansharpen`
       | :class:`deepinv.physics.HyperSpectralUnmixing`
       | :class:`deepinv.physics.CompressiveSpectralImaging`
     -

   * - Compressive
     -
       | :class:`deepinv.physics.CompressedSensing`
       | :class:`deepinv.physics.StructuredRandom`
       | :class:`deepinv.physics.SinglePixelCamera`
     -

   * - Radio Interferometric Imaging
     -
       | :class:`deepinv.physics.RadioInterferometry`
     -

   * - Single-Photon Lidar
     -
       | :class:`deepinv.physics.SinglePhotonLidar`
     -

   * - Dehazing
     -
       | :class:`deepinv.physics.Haze`
     -

   * - Phase Retrieval
     -
       | :class:`deepinv.physics.PhaseRetrieval`
       | :class:`RandomPhaseRetrieval <deepinv.physics.RandomPhaseRetrieval>`
       | :class:`StructuredRandomPhaseRetrieval <deepinv.physics.StructuredRandomPhaseRetrieval>`
       | :class:`Ptychography <deepinv.physics.Ptychography>`
       | :class:`PtychographyLinearOperator <deepinv.physics.PtychographyLinearOperator>`
     - | :func:`build_probe <deepinv.physics.phase_retrieval.build_probe>`
       | :func:`generate_shifts <deepinv.physics.phase_retrieval.generate_shifts>`


.. _wrapper_list:

Wrappers
~~~~~~~~~
Wrappers are operators that can be used to adapt existing operators to a new problem.

.. list-table:: Wrappers
    :header-rows: 1

    * - **Family**
      - **Operators**

    * - Multiscale
      -
         | :class:`deepinv.physics.PhysicsMultiScaler`
         | :class:`deepinv.physics.LinearPhysicsMultiScaler`

    * - Padding/Cropping
      -
         | :class:`deepinv.physics.PhysicsCropper`


.. _noise_list:

Noise distributions
~~~~~~~~~~~~~~~~~~~
Noise distributions describe the noise model :math:`N`,
where :math:`y = N(z)` with :math:`z=A(x)`. The noise models can be assigned
to **any** operator in the list above, by setting the
:func:`set_noise_model <deepinv.physics.Physics.set_noise_model>` attribute at initialization.
By default, the noise model is set to :class:`ZeroNoise <deepinv.physics.ZeroNoise>`.

.. list-table:: Noise Distributions and Their Probability Distributions
   :header-rows: 1

   * - **Noise**
     - :math:`y|z`

   * - :class:`deepinv.physics.ZeroNoise`
     - :math:`y=z`

   * - :class:`deepinv.physics.GaussianNoise`
     - :math:`y\sim \mathcal{N}(z, I\sigma^2)`

   * - :class:`deepinv.physics.PoissonNoise`
     - :math:`y \sim \mathcal{P}(z/\gamma)`

   * - :class:`deepinv.physics.PoissonGaussianNoise`
     - :math:`y = \gamma z + \epsilon`, :math:`z\sim\mathcal{P}(\frac{z}{\gamma})`, :math:`\epsilon\sim\mathcal{N}(0, I \sigma^2)`

   * - :class:`deepinv.physics.LogPoissonNoise`
     - :math:`y = \frac{1}{\mu} \log(\frac{\mathcal{P}(\exp(-\mu z) N_0)}{N_0})`

   * - :class:`deepinv.physics.UniformNoise`
     - :math:`y\sim \mathcal{U}(z-a, z+b)`

   * - :class:`deepinv.physics.SaltPepperNoise`
     - :math:`y = \begin{cases} 0 & \text{if } z < p\\ x & \text{if } z \in [p, 1-s]\\ 1 & \text{if } z > 1 - s\end{cases}` with :math:`z\sim\mathcal{U}(0,1)`

   * - :class:`deepinv.physics.ZeroNoise`
     - :math:`y = z`