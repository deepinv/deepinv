# PSFGenerator

### *class* deepinv.physics.generator.PSFGenerator(psf_size=(31, 31), \*\*kwargs)

Bases: [`PhysicsGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.PhysicsGenerator.md#deepinv.physics.generator.PhysicsGenerator)

Base class for generating Point Spread Functions (PSFs).

* **Parameters:**
  **psf_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – the shape of the generated PSF in 2D
  `(kernel_size, kernel_size)`. If an `int` is given, it will be used for both dimensions.

<a id="sphx-glr-backref-deepinv-physics-generator-psfgenerator"></a>

## Examples using `PSFGenerator`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train various networks using adversarial training for deblurring problems. We demonstrate running training and inference using a conditional GAN (i.e. DeblurGAN), CSGM, AmbientGAN and UAIR implemented in the library, and how to simply train your own GAN by using deepinv.training.AdversarialTrainer. These examples can also be easily extended to train more complicated GANs such as CycleGAN.">![](auto_examples/adversarial-learning/images/thumb/sphx_glr_demo_gan_imaging_thumb.png)

[Imaging inverse problems with adversarial networks](https://deepinv.org/auto_examples/adversarial-learning/demo_gan_imaging.md)

  <div class="sphx-glr-thumbnail-title">Imaging inverse problems with adversarial networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to get started with DeepInverse in under 5 minutes.">![](auto_examples/basics/images/thumb/sphx_glr_demo_quickstart_thumb.png)

[5 minute quickstart tutorial](https://deepinv.org/auto_examples/basics/demo_quickstart.md)

  <div class="sphx-glr-thumbnail-title">5 minute quickstart tutorial</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.physics.Physics together with automatic differentiation to estimate unknown parameters of your forward operator.">![](auto_examples/blind-inverse-problems/images/thumb/sphx_glr_demo_optimizing_physics_parameter_thumb.png)

[Calibrating physics operators](https://deepinv.org/auto_examples/blind-inverse-problems/demo_optimizing_physics_parameter.md)

  <div class="sphx-glr-thumbnail-title">Calibrating physics operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model terris2025reconstruct to solve inverse problems.">![](auto_examples/models/images/thumb/sphx_glr_demo_foundation_model_thumb.png)

[Inference and fine-tune a foundation model](https://deepinv.org/auto_examples/models/demo_foundation_model.md)

  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of 2D blur operators in DeepInverse. In particular, we show how to use DiffractionBlurs (Fresnel diffraction), motion blurs and space varying blurs.">![](auto_examples/physics/images/thumb/sphx_glr_demo_blur_tour_thumb.png)

[Tour of blur operators](https://deepinv.org/auto_examples/physics/demo_blur_tour.md)

  <div class="sphx-glr-thumbnail-title">Tour of blur operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of 3D blur operators in the library. In particular, we show how to use Diffraction Blurs (Fresnel diffraction) to simulate fluorescence microscopes.">![](auto_examples/physics/images/thumb/sphx_glr_demo_microscopy_3d_thumb.png)

[3D diffraction PSF](https://deepinv.org/auto_examples/physics/demo_microscopy_3d.md)

  <div class="sphx-glr-thumbnail-title">3D diffraction PSF</div>
</div>
<!-- thumbnail-parent-div-close --></div>
