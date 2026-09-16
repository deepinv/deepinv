<a id="blind"></a>

# Blind Inverse Problems

Following the [notation of the library](https://deepinv.org/user_guide/physics/intro.html.md#parameter-dependent-operators), here we consider measurements of the form
$y = \noise{\forw{x, \theta}}$, where $\theta$ represents unknown physics parameters.
Noise parameters associated to $\noise{\cdot}$ may also be unknown. In this section, we consider two classes of problems:

- **Calibration problems**: Estimate the unknown parameters $\theta$ given paired signal and measurement data $(x,y)$
- **Blind inverse problems**: Jointly estimate the signal $x$ and $\theta$ parameters (and other noise parameters) from the measurements $y$. Some methods directly estimate the signal without explicitly estimating the parameters.

## Calibration problems

If paired measurement and signal data is available at inference time, physics parameters can be estimated using optimization methods.
See the example [Calibrating physics operators](https://deepinv.org/auto_examples/blind-inverse-problems/demo_optimizing_physics_parameter.html.md#sphx-glr-auto-examples-blind-inverse-problems-demo-optimizing-physics-parameter-py) for more details.

## Physics parameters estimation

If only measurement data is available $\theta$ at inference time, we can estimate the parameters from the observed data,
and then use any non-blind reconstructor to recover the image.
The library provides the following parameter estimation models/algorithms:

#### Identification models

| Model/Algorithm                                                                                                         | Tensor Size (C, H, W)   | Pretrained Weights   | Physics                                                                                            | Parameters estimated     | Examples                                                                                                                                     |
|-------------------------------------------------------------------------------------------------------------------------|-------------------------|----------------------|----------------------------------------------------------------------------------------------------|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| [`KernelIdentificationNetwork`](https://deepinv.org/api/stubs/deepinv.models.KernelIdentificationNetwork.html.md#deepinv.models.KernelIdentificationNetwork) | C=3; H,W>8              | RGB                  | [`SpaceVaryingBlur`](https://deepinv.org/api/stubs/deepinv.physics.SpaceVaryingBlur.html.md#deepinv.physics.SpaceVaryingBlur) | `filters`, `multipliers` | [blind deblurring](https://deepinv.org/auto_examples/blind-inverse-problems/demo_blind_deblurring.html.md#sphx-glr-auto-examples-blind-inverse-problems-demo-blind-deblurring-py).      |
| [`ESPIRiT`](https://deepinv.org/api/stubs/deepinv.physics.MultiCoilMRI.html.md#deepinv.physics.MultiCoilMRI)                                   | C=2; H,W>64             | (non-learned)        | [`MultiCoilMRI`](https://deepinv.org/api/stubs/deepinv.physics.MultiCoilMRI.html.md#deepinv.physics.MultiCoilMRI)         | `coil_maps`              | [MRI coil map estimation](https://deepinv.org/auto_examples/physics/demo_mri_tour.html.md#sphx-glr-auto-examples-physics-demo-mri-tour-py).                      |
| [`WaveletNoiseEstimator`](https://deepinv.org/api/stubs/deepinv.models.WaveletNoiseEstimator.html.md#deepinv.models.WaveletNoiseEstimator)             | C=1, 2, 3               | (non-learned)        | [`GaussianNoise`](https://deepinv.org/api/stubs/deepinv.physics.GaussianNoise.html.md#deepinv.physics.GaussianNoise)       | `sigma`                  | [noise level estimation](https://deepinv.org/auto_examples/blind-inverse-problems/demo_blind_denoising.html.md#sphx-glr-auto-examples-blind-inverse-problems-demo-blind-denoising-py). |
| [`Patch`](https://deepinv.org/api/stubs/deepinv.models.PatchCovarianceNoiseEstimator.html.md#deepinv.models.PatchCovarianceNoiseEstimator)                     | C=1, 2, 3               | (non-learned)        | [`GaussianNoise`](https://deepinv.org/api/stubs/deepinv.physics.GaussianNoise.html.md#deepinv.physics.GaussianNoise)       | `sigma`                  | [noise level estimation](https://deepinv.org/auto_examples/blind-inverse-problems/demo_blind_denoising.html.md#sphx-glr-auto-examples-blind-inverse-problems-demo-blind-denoising-py). |
