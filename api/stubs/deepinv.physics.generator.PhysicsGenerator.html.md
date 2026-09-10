# PhysicsGenerator

### *class* deepinv.physics.generator.PhysicsGenerator(step=lambda \*\*kwargs: ..., rng=None, device='cpu', dtype=torch.float32, \*\*kwargs)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Base class for parameter generation of physics parameters.

Physics generators are used to generate the parameters $\theta$ of (parameter-dependent) forward operators.

Generators can be summed to create larger generators via [`deepinv.physics.generator.PhysicsGenerator.__add__()`](#deepinv.physics.generator.PhysicsGenerator.__add__),
or mixed to create a generator that randomly selects them via [`deepinv.physics.generator.GeneratorMixture`](https://deepinv.org/api/stubs/deepinv.physics.generator.GeneratorMixture.html.md#deepinv.physics.generator.GeneratorMixture).

* **Parameters:**
  * **step** (*Callable*) – a function that generates the parameters of the physics, e.g.,
    the filter of the [`deepinv.physics.Blur`](https://deepinv.org/api/stubs/deepinv.physics.Blur.html.md#deepinv.physics.Blur). This function should return the parameters in a dictionary with
    the corresponding key and value pairs.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – (optional) a pseudorandom random number generator for the parameter generation.
    If `None`, the default Generator of PyTorch will be used.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – cpu or cuda
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – the data type of the generated parameters

<hr />

* **Examples:**
  Generating blur and noise levels:
  ```pycon
  >>> import torch
  >>> from deepinv.physics.generator import MotionBlurGenerator, SigmaGenerator
  >>> # combine a PhysicsGenerator for blur and noise level parameters
  >>> generator = MotionBlurGenerator(psf_size = (3, 3), num_channels = 1) + SigmaGenerator()
  >>> params_dict = generator.step(batch_size=1, seed=0) # dict_keys(['filter', 'sigma'])
  >>> print(params_dict['filter'])
  tensor([[[[0.0000, 0.1006, 0.0000],
            [0.0000, 0.8994, 0.0000],
            [0.0000, 0.0000, 0.0000]]]])
  >>> print(params_dict['sigma'])
  tensor([0.2532])
  ```

#### \_\_add_\_(other)

Creates a new generator from the sum of two generators.

* **Parameters:**
  **other** (*Generator*) – the other generator to be added.
* **Returns:**
  A new generator that generates a larger dictionary with parameters of the two generators.

#### average(n=2000, batch_size=1, \*\*kwargs)

Calculate average of physics generator.
:param int n: number of samples to average over, defaults to 2000
:param int n: number of samples to compute in parallel, higher means faster but more costly memory-wise, defaults to 1
:param kwargs: kwargs to pass to `step` method.
:returns: A dictionary with the new parameters, that is `{param_name: param_value}`.

#### reset_rng()

Reset the random number generator to its initial state.

#### rng_manual_seed(seed=None)

Sets the seed for the random number generator.

* **Parameters:**
  **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – the seed to set for the random number generator. If string passed,
  generate seed from the hash of the string.
  If not provided, the current state of the random number generator is used.
  Note: The `torch.manual_seed` is triggered when a the random number generator is not initialized.

#### step(batch_size=1, seed=None, \*\*kwargs)

Generates a batch of parameters for the forward operator.

* **Parameters:**
  * **batch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the number of samples to generate.
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the seed for the random number generator.
* **Returns:**
  A dictionary with the new parameters, that is `{param_name: param_value}`.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)

<a id="sphx-glr-backref-deepinv-physics-generator-physicsgenerator"></a>

## Examples using `PhysicsGenerator`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train various networks using adversarial training for deblurring problems. We demonstrate running training and inference using a conditional GAN (i.e. DeblurGAN), CSGM, AmbientGAN and UAIR implemented in the library, and how to simply train your own GAN by using deepinv.training.AdversarialTrainer. These examples can also be easily extended to train more complicated GANs such as CycleGAN.">  <div class="sphx-glr-thumbnail-title">Imaging inverse problems with adversarial networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to get started with DeepInverse in under 5 minutes.">  <div class="sphx-glr-thumbnail-title">5 minute quickstart tutorial</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.physics.Physics together with automatic differentiation to estimate unknown parameters of your forward operator.">  <div class="sphx-glr-thumbnail-title">Calibrating physics operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model :footciteterris2025reconstruct to solve inverse problems.">  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of 2D blur operators in DeepInverse. In particular, we show how to use DiffractionBlurs (Fresnel diffraction), motion blurs and space varying blurs.">  <div class="sphx-glr-thumbnail-title">Tour of blur operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of 3D blur operators in the library. In particular, we show how to use Diffraction Blurs (Fresnel diffraction) to simulate fluorescence microscopes.">  <div class="sphx-glr-thumbnail-title">3D diffraction PSF</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in physics.">  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the self-supervised Artifact2Artifact loss for solving an undersampled sequential MRI reconstruction problem without ground truth.">  <div class="sphx-glr-thumbnail-title">Self-supervised MRI reconstruction with Artifact2Artifact</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Equivariant splitting consists in minimizing a self-supervised loss to train a reconstruction model using measurement data only :footcitesechaud26Equivariant.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate scan-specific self-supervised learning, that is, learning to reconstruct MRI scans from a single accelerated sample without ground truth.">  <div class="sphx-glr-thumbnail-title">Scan-specific zero-shot SSDU for MRI</div>
</div>
<!-- thumbnail-parent-div-close --></div>
