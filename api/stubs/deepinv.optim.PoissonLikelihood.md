# PoissonLikelihood

### *class* deepinv.optim.PoissonLikelihood(gain=1.0, bkg=0, denormalize=True)

Bases: [`DataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.md#deepinv.optim.DataFidelity)

Poisson negative log-likelihood.

$$
\datafid{z}{y} =  -y^{\top} \log(z+\beta)+1^{\top}z
$$

where $y$ are the measurements, $z$ is the estimated (positive) density and $\beta\geq 0$ is
an optional background level.

#### NOTE
The function is not Lipschitz smooth w.r.t. $z$ in the absence of background ($\beta=0$).

* **Parameters:**
  * **gain** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – gain of the measurement $y$. Default: 1.0.
  * **bkg** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – background level $\beta$. Default: 0.
  * **denormalize** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if True, the measurement is multiplied by the gain. Default: True.

<a id="sphx-glr-backref-deepinv-optim-poissonlikelihood"></a>

## Examples using `PoissonLikelihood`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example introduces the Richardson-Lucy algorithm for deconvolution in the blind setting, where both the underlying clean image and the blur kernel are unknown.">![](auto_examples/blind-inverse-problems/images/thumb/sphx_glr_demo_blind_richardsonlucy_thumb.png)

[Blind Richardson-Lucy deconvolution](https://deepinv.org/auto_examples/blind-inverse-problems/demo_blind_richardsonlucy.md)

  <div class="sphx-glr-thumbnail-title">Blind Richardson-Lucy deconvolution</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to solve Poisson inverse problems using the Maximum-Likelihood Expectation-Maximization (MLEM) algorithm sheppMaximumLikelihoodReconstruction1982, also known as the Richardson-Lucy algorithm in the deconvolution setting richardsonBayesianBasedIterativeMethod1972,lucyIterativeTechniqueRectification1974.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_poisson_mlem_thumb.png)

[Poisson Inverse Problems with Maximum-Likelihood Expectation-Maximization (MLEM)](https://deepinv.org/auto_examples/optimization/demo_poisson_mlem.md)

  <div class="sphx-glr-thumbnail-title">Poisson Inverse Problems with Maximum-Likelihood Expectation-Maximization (MLEM)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct an image from them.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet2d_thumb.png)

[Positron emission tomography (PET) in 2D](https://deepinv.org/auto_examples/physics/demo_pet2d.md)

  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 2D</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct a volume from them.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet3d_thumb.png)

[Positron emission tomography (PET) in 3D](https://deepinv.org/auto_examples/physics/demo_pet3d.md)

  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 3D</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a 2D slice from the BrainWeb \`&lt;https://github.com/casperdcl/brainweb&gt;\`_ positron emission tomography (PET) dataset. The slice contains five hot lesions, we simulate a sinogram with deepinv.physics.PET and we compare standard PET reconstruction algorithms with methods that support penalized objective functions.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet_brainweb_2d_thumb.png)

[2D PET reconstruction with the Brainweb dataset](https://deepinv.org/auto_examples/physics/demo_pet_brainweb_2d.md)

  <div class="sphx-glr-thumbnail-title">2D PET reconstruction with the Brainweb dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a volume from the BrainWeb \`&lt;https://github.com/casperdcl/brainweb&gt;\`_ positron emission tomography (PET) dataset. We compare standard PET reconstruction algorithms with methods that support penalized objective functions.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet_brainweb_3d_thumb.png)

[3D PET reconstruction with the Brainweb dataset](https://deepinv.org/auto_examples/physics/demo_pet_brainweb_3d.md)

  <div class="sphx-glr-thumbnail-title">3D PET reconstruction with the Brainweb dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use a mirror descent algorithm for solving an inverse problem with Poisson noise. See hurault2023convergent for more details.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_PnP_mirror_descent_thumb.png)

[Plug-and-Play algorithm with Mirror Descent for Poisson noise inverse problems.](https://deepinv.org/auto_examples/plug-and-play/demo_PnP_mirror_descent.md)

  <div class="sphx-glr-thumbnail-title">Plug-and-Play algorithm with Mirror Descent for Poisson noise inverse problems.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we revisit the implementation of the DiffPIR diffusion algorithm for image reconstruction from zhu2023denoising. The full algorithm is implemented in deepinv.sampling.DiffPIR.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_diffpir_thumb.png)

[Implementing DiffPIR](https://deepinv.org/auto_examples/sampling/demo_diffpir.md)

  <div class="sphx-glr-thumbnail-title">Implementing DiffPIR</div>
</div>
<!-- thumbnail-parent-div-close --></div>
