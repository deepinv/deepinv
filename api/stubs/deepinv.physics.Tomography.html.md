# Tomography

### *class* deepinv.physics.Tomography(angles, img_width, circle=False, parallel_computation=True, adjoint_via_backprop=True, fbp_interpolate_boundary=False, normalize=None, fan_beam=False, fan_parameters=None, device=torch.device('cpu'), dtype=torch.float, \*\*kwargs)

Bases: [`LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)

(Computed) Tomography operator.

The Radon transform is the integral transform which takes a square image $x$ defined on the plane to a function
$y=\forw{x}$ defined on the (two-dimensional) space of lines in the plane, whose value at a particular line is equal
to the line integral of the function over that line.

#### NOTE
The pseudo-inverse is computed using the filtered back-projection algorithm with a Ramp filter.
This is not the exact linear pseudo-inverse of the Radon transform, but it is a good approximation which is
robust to noise.

#### NOTE
The measurements are not normalized by the image size, thus the norm of the operator depends on the image size.

#### NOTE
This operator only handles 2D images. For more advanced use-cases, see the [`deepinv.physics.TomographyWithAstra`](https://deepinv.org/api/stubs/deepinv.physics.TomographyWithAstra.html.md#deepinv.physics.TomographyWithAstra) operator which handles 2D and 3D geometries.

#### WARNING
The adjoint operator has small numerical errors due to interpolation. Set `adjoint_via_backprop=True` if you want to use the exact adjoint (computed via autograd).

#### WARNING
By default, `normalize` is set to `True` if not specified. Initializing the operator without specifying the normalization behavior will issue a warning. Note that normalizing the operator affects the reconstruction dynamics, which may not always be suitable for real-world applications.

* **Parameters:**
  * **angles** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – These are the tomography angles. If the type is `int`, the angles are sampled uniformly between 0 and 360 degrees.
    If the type is [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), the angles are the ones provided (e.g., `torch.linspace(0, 180, steps=10)`).
  * **img_width** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – width/height of the square image input.
  * **circle** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True` both forward and backward projection will be restricted to pixels inside a circle
    inscribed in the square image.
  * **parallel_computation** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, all projections are performed in parallel. Requires more memory but is faster on GPUs.
  * **adjoint_via_backprop** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, the adjoint will be computed via [`deepinv.physics.adjoint_function()`](https://deepinv.org/api/stubs/deepinv.physics.adjoint_function.html.md#deepinv.physics.adjoint_function). Otherwise the inverse Radon transform is used.
    The inverse Radon transform is computationally cheaper (particularly in memory), but has a small adjoint mismatch.
    The backprop adjoint is the exact adjoint, but might break random seeds since it backpropagates through [`torch.nn.functional.grid_sample()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample), see the note [here](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html).
  * **fbp_interpolate_boundary** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – the [`filtered back-projection`](#deepinv.physics.Tomography.A_dagger) usually contains streaking artifacts on the boundary due to padding. For `fbp_interpolate_boundary=True`
    these artifacts are corrected by cutting off the outer two pixels of the FBP and recovering them by interpolating the remaining image. This option
    only makes sense if `circle` is set to `False`. Hence it will be ignored if `circle` is True.
  * **normalize** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True` [`A`](#deepinv.physics.Tomography.A) and [`A_adjoint`](#deepinv.physics.Tomography.A_adjoint) are normalized so that the operator has unit norm. (default: `True`)
  * **fan_beam** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, use fan beam geometry, if `False` use parallel beam
  * **fan_parameters** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *|* [*float*](https://docs.python.org/3.9/library/functions.html#float) *]*) – 

    Only used if fan_beam is `True`. Contains the parameters defining the scanning geometry. The dict should contain the keys:
    - ”pixel_spacing” defining the distance between two pixels in the image, default: 0.5 / (in_size)
    - ”source_radius” distance between the x-ray source and the rotation axis (middle of the image), default: 57.5
    - ”detector_radius” distance between the x-ray detector and the rotation axis (middle of the image), default: 57.5
    - ”n_detector_pixels” number of pixels of the detector, default: 258
    - ”detector_spacing” distance between two pixels on the detector, default: 0.077

    The default values are adapted from the geometry in Khalil *et al.*<sup>[1](#footcite-khalil2023hyperspectral)</sup>.
    where pixel spacing, source and detector radius and detector spacing are given in cm.
    Note that a to small value of n_detector_pixels\*detector_spacing can lead to severe circular artifacts in any reconstruction.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – gpu or cpu.

<hr />

* **Examples:**
  Tomography operator with defined angles for 3x3 image:
  ```pycon
  >>> from deepinv.physics import Tomography
  >>> seed = torch.manual_seed(0)  # Random seed for reproducibility
  >>> x = torch.randn(1, 1, 4, 4)  # Define random 4x4 image
  >>> angles = torch.linspace(0, 45, steps=3)
  >>> physics = Tomography(angles=angles, img_width=4, circle=True, normalize=False)
  >>> physics(x)
  tensor([[[[ 0.0000, -0.1791, -0.1719],
            [-0.5713, -0.4521, -0.5177],
            [ 0.0340,  0.1448,  0.2334],
            [ 0.0000, -0.0448, -0.0430]]]])
  ```

  Tomography operator with 3 uniformly sampled angles in [0, 360] for 3x3 image:
  ```pycon
  >>> from deepinv.physics import Tomography
  >>> seed = torch.manual_seed(0)  # Random seed for reproducibility
  >>> x = torch.randn(1, 1, 4, 4)  # Define random 4x4 image
  >>> physics = Tomography(angles=3, img_width=4, circle=True, normalize=False)
  >>> physics(x)
  tensor([[[[ 0.0000, -0.1806,  0.0500],
            [-0.5713, -0.6076, -0.6815],
            [ 0.0340,  0.3175,  0.0167],
            [ 0.0000, -0.0452,  0.0989]]]])
  ```

<hr />

* **References:**

* <a id='footcite-khalil2023hyperspectral'>**[1]**</a> Mohamad Khalil, Jan Kehres, and Wail Mustafa. Hyperspectral 2d fan-beam x-ray ct dataset of 5 materials. September 2023. Dataset. URL: [https://doi.org/10.5281/zenodo.8307932](https://doi.org/10.5281/zenodo.8307932), [doi:10.5281/zenodo.8307932](https://doi.org/10.5281/zenodo.8307932).

#### A(x, \*\*kwargs)

Forward projection.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input of shape [B,C,H,W]
* **Returns:**
  measurement of shape [B,C,A,N], with A the number of angular positions, and N the number of detector cells.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### A_adjoint(y, \*\*kwargs)

Computes adjoint of the tomography operator.

#### WARNING
The default adjoint operator has small numerical errors due to interpolation. Set `adjoint_via_backprop=True` if you want to use the exact adjoint (computed via autograd).

* **Parameters:**
  **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements of shape [B,C,A,N]
* **Returns:**
  scaled back-projection of shape [B,C,H,W]
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### A_dagger(y, fbp=False, \*\*kwargs)

Computes the solution in $x$ to $y = Ax$ using a least squares solver. A faster approximation can be obtained by setting `fbp=True`, which computes the filtered back-projection of the measurements.

#### WARNING
The filtered back-projection algorithm is not the exact linear pseudo-inverse of the Radon transform, but it is a good approximation that is robust to noise.

* **Parameters:**
  **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements of shape [B,C,A,N], with A the number of angular positions, and N the number of detector cells
* **Returns:**
  filtered back-projection of shape [B,C,H,W]
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### fbp(y, \*\*kwargs)

Computes the filtered back-projection (FBP) of the measurements.

#### TIP
By default, the FBP reconstruction can display artifacts at the borders. Set `fbp_interpolate_boundary=True` to remove them with padding.

* **Parameters:**
  **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements of shape [B,C,A,N], with A the number of angular positions, and N the number of detector cells
* **Returns:**
  filtered back-projection of shape [B,C,H,W]
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-physics-tomography"></a>

## Examples using `Tomography`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model :footciteterris2025reconstruct to solve inverse problems.">  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we use patch priors for limited angle computed tomography. More precisely, we consider the inverse problem y = \\mathrm{noisy}(Ax), where A is the discretized Radon transform with 100 equispace angles between 20 and 160 degrees. For the reconstruction, we minimize the variational problem">  <div class="sphx-glr-thumbnail-title">Patch priors for limited-angle computed tomography</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to solve Poisson inverse problems using the Maximum-Likelihood Expectation-Maximization (MLEM) algorithm :footcitesheppMaximumLikelihoodReconstruction1982, also known as the Richardson-Lucy algorithm in the deconvolution setting :footciterichardsonBayesianBasedIterativeMethod1972,lucyIterativeTechniqueRectification1974.">  <div class="sphx-glr-thumbnail-title">Poisson Inverse Problems with Maximum-Likelihood Expectation-Maximization (MLEM)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in physics.">  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard PnP algorithm with DnCNN denoiser for computed tomography.">  <div class="sphx-glr-thumbnail-title">Vanilla PnP for computed tomography (CT).</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised learning with measurement splitting, to train a denoiser network on the MNIST dataset. The physics here is noisy computed tomography, as is the case in Noise2Inverse :footcitehendriksen2020noise2inverse. Note this example can also be easily applied to undersampled multicoil MRI as is the case in SSDU :footciteyaman2020self.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with measurement splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="where both the data fidelity and the prior are learned modules, distinct for each iterations.">  <div class="sphx-glr-thumbnail-title">Learned Primal-Dual algorithm for CT scan.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
