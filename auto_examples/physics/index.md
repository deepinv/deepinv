<a id="sphx-glr-auto-examples-physics"></a>

# Physics

<div id='sg-tag-list' class='sphx-glr-tag-list'></div><div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" data-sgtags='["Tomography", "Denoising", "Single-pixel", "Demosaicing"]' tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in here.">![](auto_examples/physics/images/thumb/sphx_glr_demo_physics_tour_thumb.png)

[Tour of forward sensing operators](https://deepinv.org/auto_examples/physics/demo_physics_tour.md)

  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div><div class="sphx-glr-thumbcontainer" data-sgtags='["Deblurring", "Microscopy"]' tooltip="This example provides a tour of 2D blur operators in DeepInverse. In particular, we show how to use DiffractionBlurs (Fresnel diffraction), motion blurs and space varying blurs.">![](auto_examples/physics/images/thumb/sphx_glr_demo_blur_tour_thumb.png)

[Tour of blur operators](https://deepinv.org/auto_examples/physics/demo_blur_tour.md)

  <div class="sphx-glr-thumbnail-title">Tour of blur operators</div>
</div><div class="sphx-glr-thumbcontainer" data-sgtags='["MRI"]' tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">![](auto_examples/physics/images/thumb/sphx_glr_demo_mri_tour_thumb.png)

[Tour of MRI functionality in DeepInverse](https://deepinv.org/auto_examples/physics/demo_mri_tour.md)

  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" data-sgtags='["Ultrasound"]' tooltip="This example presents the plane-wave ultrafast ultrasound forward physics (deepinv.physics.UltrasoundPlaneWave) available in DeepInverse for pulse-echo imaging problems.">![](auto_examples/physics/images/thumb/sphx_glr_demo_ultrasound_tour_thumb.png)

[Tour of ultrafast ultrasound in DeepInverse](https://deepinv.org/auto_examples/physics/demo_ultrasound_tour.md)

  <div class="sphx-glr-thumbnail-title">Tour of ultrafast ultrasound in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" data-sgtags='["Microscopy", "Deblurring"]' tooltip="This example provides a tour of 3D blur operators in the library. In particular, we show how to use Diffraction Blurs (Fresnel diffraction) to simulate fluorescence microscopes.">![](auto_examples/physics/images/thumb/sphx_glr_demo_microscopy_3d_thumb.png)

[3D diffraction PSF](https://deepinv.org/auto_examples/physics/demo_microscopy_3d.md)

  <div class="sphx-glr-thumbnail-title">3D diffraction PSF</div>
</div><div class="sphx-glr-thumbcontainer" data-sgtags='["Plug-and-play"]' tooltip="This example shows how to create a random phase retrieval operator and generate phaseless measurements from a given image. The example showcases 4 different reconstruction methods to recover the image from the phaseless measurements:">![](auto_examples/physics/images/thumb/sphx_glr_demo_phase_retrieval_thumb.png)

[Random phase retrieval and reconstruction methods.](https://deepinv.org/auto_examples/physics/demo_phase_retrieval.md)

  <div class="sphx-glr-thumbnail-title">Random phase retrieval and reconstruction methods.</div>
</div><div class="sphx-glr-thumbcontainer" data-sgtags='["Deblurring"]' tooltip="Real-world blurry images have decorrelated opposite boundaries, unlike images synthetically blurred using circular filters. This makes the use of spectral deconvolution methods (inverse filtering, Wiener filtering) impractical and prone to ringing artifacts. Liu-Jia padding liu2008reducing is a pre-processing step that pads the input image to make it have smooth circular boundaries, while preserving the original spectral content as much as possible.">![](auto_examples/physics/images/thumb/sphx_glr_demo_liu_jia_padding_thumb.png)

[Spectral Methods for Non-Circular Deblurring with Liu-Jia Padding](https://deepinv.org/auto_examples/physics/demo_liu_jia_padding.md)

  <div class="sphx-glr-thumbnail-title">Spectral Methods for Non-Circular Deblurring with Liu-Jia Padding</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to create a Ptychography phase retrieval operator and generate phaseless measurements from a given image.">![](auto_examples/physics/images/thumb/sphx_glr_demo_ptychography_thumb.png)

[Ptychography phase retrieval](https://deepinv.org/auto_examples/physics/demo_ptychography.md)

  <div class="sphx-glr-thumbnail-title">Ptychography phase retrieval</div>
</div><div class="sphx-glr-thumbcontainer" data-sgtags='["Tomography"]' tooltip="In this example we show how to use the deepinv.physics.Scattering forward model.">![](auto_examples/physics/images/thumb/sphx_glr_demo_scattering_thumb.png)

[Inverse scattering problem](https://deepinv.org/auto_examples/physics/demo_scattering.md)

  <div class="sphx-glr-thumbnail-title">Inverse scattering problem</div>
</div><div class="sphx-glr-thumbcontainer" data-sgtags='["Denoising"]' tooltip="This example demonstrates how to denoise images corrupted by Poisson-Gaussian noise using the Generalized Anscombe Transform (GAT), which converts any Gaussian denoiser into a Poisson-Gaussian denoiser makitalo2012optimal.">![](auto_examples/physics/images/thumb/sphx_glr_demo_anscombe_thumb.png)

[Poisson-Gaussian Denoising with the Generalized Anscombe Transform](https://deepinv.org/auto_examples/physics/demo_anscombe.md)

  <div class="sphx-glr-thumbnail-title">Poisson-Gaussian Denoising with the Generalized Anscombe Transform</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo illustrates the impact of different Hadamard pattern ordering algorithms in the Single Pixel Camera (SPC), a computational imaging system that uses a single photodetector to capture images by projecting the scene onto a series of patterns. The SPC is implemented in the DeepInverse library.">![](auto_examples/physics/images/thumb/sphx_glr_demo_spc_thumb.png)

[Pattern Ordering in a Compressive Single Pixel Camera](https://deepinv.org/auto_examples/physics/demo_spc.md)

  <div class="sphx-glr-thumbnail-title">Pattern Ordering in a Compressive Single Pixel Camera</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows the use of the deepinv.physics.SpatialUnwrapping forward model and the deepinv.optim.ItohFidelity for unwrapping problems, which occur in modulo imaging, interferometry SAR and other imaging applications. It shows how to generate a wrapped phase image, apply blur and noise, and reconstruct the original phase using both DCT inversion and ADMM optimization.">![](auto_examples/physics/images/thumb/sphx_glr_demo_spatial_unwrapping_thumb.png)

[Spatial unwrapping and modulo imaging](https://deepinv.org/auto_examples/physics/demo_spatial_unwrapping.md)

  <div class="sphx-glr-thumbnail-title">Spatial unwrapping and modulo imaging</div>
</div><div class="sphx-glr-thumbcontainer" data-sgtags='["Tomography", "PET/SPECT"]' tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct a volume from them.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet3d_thumb.png)

[Positron emission tomography (PET) in 3D](https://deepinv.org/auto_examples/physics/demo_pet3d.md)

  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 3D</div>
</div><div class="sphx-glr-thumbcontainer" data-sgtags='["Tomography", "PET/SPECT"]' tooltip="This example reconstructs a 2D slice from the BrainWeb \`&lt;https://github.com/casperdcl/brainweb&gt;\`_ positron emission tomography (PET) dataset. The slice contains five hot lesions, we simulate a sinogram with deepinv.physics.PET and we compare standard PET reconstruction algorithms with methods that support penalized objective functions.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet_brainweb_2d_thumb.png)

[2D PET reconstruction with the Brainweb dataset](https://deepinv.org/auto_examples/physics/demo_pet_brainweb_2d.md)

  <div class="sphx-glr-thumbnail-title">2D PET reconstruction with the Brainweb dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we show how to use the deepinv.physics.SinglePhotonLidar forward model.">![](auto_examples/physics/images/thumb/sphx_glr_demo_lidar_thumb.png)

[Single photon lidar operator for depth ranging.](https://deepinv.org/auto_examples/physics/demo_lidar.md)

  <div class="sphx-glr-thumbnail-title">Single photon lidar operator for depth ranging.</div>
</div><div class="sphx-glr-thumbcontainer" data-sgtags='["Tomography", "PET/SPECT"]' tooltip="This example reconstructs a volume from the BrainWeb \`&lt;https://github.com/casperdcl/brainweb&gt;\`_ positron emission tomography (PET) dataset. We compare standard PET reconstruction algorithms with methods that support penalized objective functions.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet_brainweb_3d_thumb.png)

[3D PET reconstruction with the Brainweb dataset](https://deepinv.org/auto_examples/physics/demo_pet_brainweb_3d.md)

  <div class="sphx-glr-thumbnail-title">3D PET reconstruction with the Brainweb dataset</div>
</div><div class="sphx-glr-thumbcontainer" data-sgtags='["Tomography", "PET/SPECT"]' tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct an image from them.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet2d_thumb.png)

[Positron emission tomography (PET) in 2D](https://deepinv.org/auto_examples/physics/demo_pet2d.md)

  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 2D</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we demonstrate remote sensing inverse problems for multispectral satellite imaging.">![](auto_examples/physics/images/thumb/sphx_glr_demo_remote_sensing_thumb.png)

[Remote sensing with satellite images](https://deepinv.org/auto_examples/physics/demo_remote_sensing.md)

  <div class="sphx-glr-thumbnail-title">Remote sensing with satellite images</div>
</div>
<!-- thumbnail-parent-div-close --></div>
