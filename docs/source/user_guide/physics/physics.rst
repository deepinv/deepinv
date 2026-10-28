.. _physics:

Operators & Noise
=================


.. _physics_list:

Operators
~~~~~~~~~
Operators describe the forward model :math:`z = A(x,\theta)`, where
:math:`x` is the input image and :math:`\theta` are the parameters of the operator.
The parameters :math:`\theta` can be sampled using random generators, which are available for some specific classes.
Using automatic differentiation, we can compute derivatives w.r.t both the input :math:`x` or the parameters :math:`\theta`. 
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
       | :class:`deepinv.physics.SpatialUnwrapping`
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
       | :class:`deepinv.physics.TiledSpaceVaryingBlur`
       | :class:`deepinv.physics.Downsampling`
       | :class:`deepinv.physics.Upsampling`
       | :class:`deepinv.physics.DownsamplingMatlab`
     -
       | :class:`GaussianBlurGenerator <deepinv.physics.generator.GaussianBlurGenerator>`
       | :class:`MotionBlurGenerator <deepinv.physics.generator.MotionBlurGenerator>`
       | :class:`DownsamplingGenerator <deepinv.physics.generator.DownsamplingGenerator>`
       | :class:`DiffractionBlurGenerator <deepinv.physics.generator.DiffractionBlurGenerator>`
       | :class:`ProductConvolutionBlurGenerator <deepinv.physics.generator.ProductConvolutionBlurGenerator>`
       | :class:`TiledBlurGenerator <deepinv.physics.generator.TiledBlurGenerator>`
       | :class:`ConfocalBlurGenerator3D <deepinv.physics.generator.ConfocalBlurGenerator3D>`
       | :class:`gaussian_blur <deepinv.physics.functional.gaussian_blur>`, :class:`sinc_filter <deepinv.physics.functional.sinc_filter>`
       | :class:`bilinear_filter <deepinv.physics.functional.bilinear_filter>`, :class:`bicubic_filter <deepinv.physics.functional.bicubic_filter>`
       | :class:`Zernike <deepinv.physics.generator.Zernike>` 

   * - Magnetic Resonance Imaging (MRI)
     -
       | :class:`deepinv.physics.MRI`
       | :class:`deepinv.physics.MultiCoilMRI`
       | :class:`deepinv.physics.DynamicMRI`
       | :class:`deepinv.physics.SequentialMRI`
       | :class:`deepinv.physics.DynamicMultiCoilMRI`
       | :class:`deepinv.physics.SequentialMultiCoilMRI`
       | The MRI operators above also natively support 3D MRI.
       | :class:`deepinv.physics.NonCartesianMRI`
     -
       | :class:`GaussianMaskGenerator <deepinv.physics.generator.GaussianMaskGenerator>`
       | :class:`RandomMaskGenerator <deepinv.physics.generator.RandomMaskGenerator>`
       | :class:`EquispacedMaskGenerator <deepinv.physics.generator.EquispacedMaskGenerator>`
       | :class:`PolyOrderMaskGenerator <deepinv.physics.generator.PolyOrderMaskGenerator>`
       | :class:`SequentialMaskGenerator <deepinv.physics.generator.SequentialMaskGenerator>`
       | Cartesian mask generators support k+t dynamic sampling.
     -
       | :class:`BrownianGenerator <deepinv.physics.generator.BrownianGenerator>`
       | Trajectory generators allow modelling motion during acquisition.


   * - Tomography
     -
       | :class:`deepinv.physics.Tomography`
       | :class:`deepinv.physics.TomographyWithAstra`
       | :class:`deepinv.physics.PET`
     -

   * - Ultrasound
     -
       | :class:`deepinv.physics.UltrasoundPlaneWave`
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
       | :class:`PtychographyGeometry <deepinv.physics.PtychographyGeometry>`
       | :class:`FarFieldPtychographyGeometry <deepinv.physics.FarFieldPtychographyGeometry>`
       | :class:`NearFieldPtychographyGeometry <deepinv.physics.NearFieldPtychographyGeometry>`
     - | :func:`build_probe <deepinv.physics.phase_retrieval.build_probe>`
       | :func:`generate_shifts <deepinv.physics.phase_retrieval.generate_shifts>`

   * - Inverse Scattering
     -
       | :class:`deepinv.physics.Scattering`
     -

.. _dynamic_imaging:

Dynamic & time-resolved imaging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`deepinv` supports a range of dynamic and time-resolved operators. 

- **Dynamic imaging**: `x` is a video of shape `B,C,T,...` and `y` is a time-series of shape `B,C,T,...`. Dynamic operators allow to model video inverse problems where we try to recover videos, not just static images.
- **"Time-resolved"/"Sequential" imaging**: `x` is an image of shape `B,C,...` and `y` is a time-series of shape `B,C,T,...`. Time-resolved operators allow to model sequential-based acquisition, where we try to recover static images from a series of **shots** `y`. Paired with **per-shot transforms**, this allows us to model motion-corruption, shot-varying blur etc.

Many imaging modalities have natural "dynamic" variants, for example:

.. list-table:: Dynamic imaging modalities
   :header-rows: 1

   * - **Modality**
     - **Recover `x`...**
     - **... from `y`**
     - **Class**

   * - Cardiac MRI
     - Videos of cardiac phases
     - Dynamic kspace
     -
       | :class:`deepinv.physics.DynamicMRI`
       | :class:`deepinv.physics.DynamicMultiCoilMRI`

   * - Video deblurring
     - Sharp videos
     - Blurry videos
     -
       | :class:`deepinv.physics.Blur`
       | :class:`deepinv.physics.BlurFFT`

Many static imaging modalities are actually more accurately modelled as "time-resolved" or "sequential". For example in MRI and CT, kspace lines or projections
are not actually acquired all at the same time, but acquired sequentially in "shots" as the scanner moves or scans.
Time between projections is non-negligible, and therefore subject motion may arise during scanning.

.. note::
  Each physics below will implement the sequential sampling differently. For example, MRI's FFT can be parallelised across time, so sequential physics can be
  efficiently described as a summed dynamic physics. However, CT projectors are almost always written sequentially already.

.. list-table:: Time-resolved/sequential imaging modalities
   :header-rows: 1

   * - **Modality**
     - **Recover `x`...**
     - **... from `y`**
     - **Class**

   * - Most of MRI
     - (Static) image
     - Dynamic kspace acquired sequentially in shots (lines or spokes).
     - :class:`deepinv.physics.SequentialMultiCoilMRI` with :class:`deepinv.physics.generator.SequentialMaskGenerator`

   * - CT or CBCT
     - (Static) image
     - Projections acquired sequentially as the scanner rotates.
     - TBC

   * - Motion deblurring
     - Sharp image
     - Sum of multiple images acquired with spatial displacement
     -
       | :class:`deepinv.physics.Blur`
       | :class:`deepinv.physics.BlurFFT`

   * - Multi-image super-resolution
     - Sharp high-resolution image
     - Multiple low-resolution images acquired with spatial displacement
     - TBC

The time-resolved/sequential physics above allow us to model **per-shot transforms**, for example motion-corruption/compensation,
which is an important problem in many modalities. First, define a :class:`transform <deepinv.transform.Transform>`, which is an
object with deterministic `.forward` and `.inverse` methods, given specific transform parameters (e.g. rotation angle).
Then, define transform parameters of shape `B,T` where `T` is the number of time-steps or "shots". 
Finally, define the sequential physics, which performs a transform per-shot inside `A`, and the inverse transform inside `A_adjoint`.
In the example below, we simulate sequential MRI measurements where the patient rotates through 15 degrees across a 4-shot acquisition:

.. doctest::

    >>> import deepinv as dinv
    >>> import torch
    >>> from deepinv.physics.generator import RandomMaskGenerator, SequentialMaskGenerator

    >>> transform = dinv.transform.Rotate(interpolation_mode="bilinear", index_params_into_batch=True)

    >>> theta = torch.tensor([[0., 5., 10., 15.]]) # shape B,T

    >>> mask = SequentialMaskGenerator(RandomMaskGenerator((2, 320, 320), acceleration=1), T=4).step()["mask"] # B,C,T,H,W

    >>> physics = dinv.physics.SequentialMultiCoilMRI(
    ...     mask=mask,
    ...     transform=transform,
    ...     transform_params={"theta": theta},
    ... )

    >>> x = dinv.utils.phantoms.generate_shepp_logan(320, channels=2).unsqueeze(0) # 1,2,320,320

    >>> y = physics(x) # simulate motion-corrupted measurement
    >>> print(y.shape) # B,C,N,T,H,W
    torch.Size([1, 2, 1, 4, 320, 320])

    >>> x_adj = physics.A_adjoint(y) # motion-compensated recon
    >>> print(x_adj.shape)  # B,C,H,W
    torch.Size([1, 2, 320, 320])

    >>> x_mc = physics.A_adjoint(y, blind=True) # motion-corrupted ("blind") recon
    >>> print(x_mc.shape) # B,C,H,W
    torch.Size([1, 2, 320, 320])


Since transform params are params of the physics, you can use :class:`trajectory generators <deepinv.physics.generator.TrajectoryGenerator>`
to simulate random transform trajectories on the fly, e.g. for training, and pass them to ``physics.update`` along with other physics params:

.. doctest::

    >>> from deepinv.physics.generator import BrownianGenerator
    >>> physics_generator = BrownianGenerator("theta", T=4, sigma=6, bound=15) + SequentialMaskGenerator(RandomMaskGenerator((2, 320, 320), acceleration=2), T=4)
    >>> params = physics_generator.step(batch_size=1)
    >>> print(sorted(params.keys())) # theta of shape B,T and mask of shape B,C,T,H,W
    ['mask', 'theta']
    >>> physics.update(**params)
    >>> y = physics(x) # motion-corrupted with Brownian generator

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

   * - :class:`deepinv.physics.LaplaceNoise`
     - :math:`y\sim\text{Laplace}(z, b)`

   * - :class:`deepinv.physics.LogPoissonNoise`
     - :math:`y = -\frac{1}{\mu} \log(\frac{\mathcal{P}(\exp(-\mu z) N_0)}{N_0})`

   * - :class:`deepinv.physics.UniformNoise`
     - :math:`y\sim \mathcal{U}(z-a, z+b)`

   * - :class:`deepinv.physics.SaltPepperNoise`
     - :math:`y = \begin{cases} 0 & \text{if } z < p\\ x & \text{if } z \in [p, 1-s]\\ 1 & \text{if } z > 1 - s\end{cases}` with :math:`z\sim\mathcal{U}(0,1)`

   * - :class:`deepinv.physics.GammaNoise`
     - :math:`y = \mathcal{G}(\ell, x/\ell)`
     
   * - :class:`deepinv.physics.FisherTippettNoise`
     - :math:`p(y\vert x) = \frac{\ell^{\ell}}{\Gamma(\ell)}\mathrm{e}^{\ell(y-x)}\mathrm{e}^{-\ell\mathrm{e}^{(y-x)}}`

   * - :class:`deepinv.physics.RicianNoise`
     - :math:`y = \sqrt{(x + \sigma \epsilon_1)^2 + (\sigma \epsilon_2)^2}`, :math:`\epsilon_1\sim\mathcal{N}(0,I)`, :math:`\epsilon_2\sim\mathcal{N}(0,I)`
