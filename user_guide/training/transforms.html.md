<a id="transform"></a>

# Transforms

This module contains different transforms which can be used for data augmentation or together with the equivariant losses.

We implement various geometric transforms, ranging from Euclidean to homography and diffeomorphisms, some of which offer group-theoretic properties.

**See** [Image transforms for equivariance & augmentations](https://deepinv.org/auto_examples/transforms-equivariance/demo_transforms.html.md#sphx-glr-auto-examples-transforms-equivariance-demo-transforms-py) **for example usage and visualisations.**

Transforms inherit from [`deepinv.transform.Transform`](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform). Transforms can also be stacked by summing them, chained by multiplying them (i.e. product group), or joined via `|` to randomly select.
There are numerous other parameters e.g to randomly transform multiple times at once, to constrain the parameters to a range etc.

Common usages of transforms:

- Make a denoiser equivariant using [`deepinv.models.EquivariantDenoiser`](https://deepinv.org/api/stubs/deepinv.models.EquivariantDenoiser.html.md#deepinv.models.EquivariantDenoiser)
  <br/>
  by performing Reynolds averaging using `symmetrize()`. See [Image transformations for Equivariant Imaging](https://deepinv.org/auto_examples/self-supervised-learning/demo_ei_transforms.html.md#sphx-glr-auto-examples-self-supervised-learning-demo-ei-transforms-py).
  <br/>
- Make a reconstructor equivariant using [`deepinv.models.EquivariantReconstructor`](https://deepinv.org/api/stubs/deepinv.models.EquivariantReconstructor.html.md#deepinv.models.EquivariantReconstructor)
  <br/>
  by performing Reynolds averaging. See [Image transforms for equivariance & augmentations](https://deepinv.org/auto_examples/transforms-equivariance/demo_transforms.html.md#sphx-glr-auto-examples-transforms-equivariance-demo-transforms-py).
  <br/>
- Equivariant imaging (EI) using the [`deepinv.loss.EILoss`](https://deepinv.org/api/stubs/deepinv.loss.EILoss.html.md#deepinv.loss.EILoss) loss.
  <br/>
  See [Self-supervised learning with Equivariant Imaging for MRI.](https://deepinv.org/auto_examples/self-supervised-learning/demo_equivariant_imaging.html.md#sphx-glr-auto-examples-self-supervised-learning-demo-equivariant-imaging-py).
  <br/>

If needed, transforms can also be made deterministic by passing in specified parameters to the forward method.
This allows every transform to have its own deterministic inverse using `transform.inverse()`.
Transforms can also be seamlessly integrated with existing `torchvision` transforms and can also accept video (5D) input.

For example, random transforms can be used as follows:

```pycon
>>> import torch
>>> from deepinv.transform import Shift, Rotate
>>> from torchvision.transforms import InterpolationMode
>>> x = torch.rand((1, 1, 2, 2)) # Define random image (B,C,H,W)
>>> transform = Shift() # Define random shift transform
>>> transform(x).shape
torch.Size([1, 1, 2, 2])
>>> y = transform(transform(x, x_shift=[1]), x_shift=[-1]) # Deterministic transform
>>> torch.all(x == y)
tensor(True)
>>> transform(torch.rand((1, 1, 3, 2, 2))).shape # Accepts video input of shape (B,C,T,H,W)
torch.Size([1, 1, 3, 2, 2])
>>> transform = Rotate(
...         interpolation_mode=InterpolationMode.BILINEAR
... ) + Shift() # Stack rotate and shift transforms
>>> transform(x).shape
torch.Size([2, 1, 2, 2])
>>> rotoshift = Rotate(
...         interpolation_mode=InterpolationMode.BILINEAR
... ) * Shift() # Chain rotate and shift transforms
>>> rotoshift(x).shape
torch.Size([1, 1, 2, 2])
>>> transform = Rotate(
...         interpolation_mode=InterpolationMode.BILINEAR
... ) | Shift() # Randomly select rotate or shift transforms
>>> transform(x).shape
torch.Size([1, 1, 2, 2])
>>> f = lambda x: x[..., [0]] * x # Function to be symmetrized
>>> f_s = rotoshift.symmetrize(f)
>>> f_s(x).shape
torch.Size([1, 1, 2, 2])
```

## Simple transforms

We provide the following simple geometric transforms.

#### Simple Transformations

| **Transform**                                                                                          | **Uses Interpolation**   | **Exact Inversion**   |
|--------------------------------------------------------------------------------------------------------|--------------------------|-----------------------|
| [`deepinv.transform.Rotate`](https://deepinv.org/api/stubs/deepinv.transform.Rotate.html.md#deepinv.transform.Rotate)     | Yes                      | No                    |
| [`deepinv.transform.Shift`](https://deepinv.org/api/stubs/deepinv.transform.Shift.html.md#deepinv.transform.Shift)       | No                       | Yes                   |
| [`deepinv.transform.Scale`](https://deepinv.org/api/stubs/deepinv.transform.Scale.html.md#deepinv.transform.Scale)       | Yes                      | No                    |
| [`deepinv.transform.Reflect`](https://deepinv.org/api/stubs/deepinv.transform.Reflect.html.md#deepinv.transform.Reflect)   | No                       | Yes                   |
| [`deepinv.transform.Identity`](https://deepinv.org/api/stubs/deepinv.transform.Identity.html.md#deepinv.transform.Identity) | No                       | Yes                   |

## Advanced transforms

We implement the following further geometric transforms.
The projective transformations formulate the image transformations using the pinhole camera model,
from which various transformation subgroups can be derived.
See [Image transformations for Equivariant Imaging](https://deepinv.org/auto_examples/self-supervised-learning/demo_ei_transforms.html.md#sphx-glr-auto-examples-self-supervised-learning-demo-ei-transforms-py) for a demonstration.
Note these require installing the library `kornia`.

#### Advanced Transformations

| **Transform**                                                                                                                          | **Description**                                                                                                  |
|----------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| [`deepinv.transform.Homography`](https://deepinv.org/api/stubs/deepinv.transform.Homography.html.md#deepinv.transform.Homography)                             | A general projective transformation allowing perspective distortion and transformation between different planes. |
| [`deepinv.transform.projective.Euclidean`](https://deepinv.org/api/stubs/deepinv.transform.projective.Euclidean.html.md#deepinv.transform.projective.Euclidean)         | A rigid transformation that preserves angles and distances, allowing only rotation and translation.              |
| [`deepinv.transform.projective.Similarity`](https://deepinv.org/api/stubs/deepinv.transform.projective.Similarity.html.md#deepinv.transform.projective.Similarity)       | A transformation that preserves shapes through scaling, rotation, and translation, maintaining proportions.      |
| [`deepinv.transform.projective.Affine`](https://deepinv.org/api/stubs/deepinv.transform.projective.Affine.html.md#deepinv.transform.projective.Affine)               | A transformation preserving parallel lines, allowing scaling, rotation, translation, and shearing.               |
| [`deepinv.transform.projective.PanTiltRotate`](https://deepinv.org/api/stubs/deepinv.transform.projective.PanTiltRotate.html.md#deepinv.transform.projective.PanTiltRotate) | A specialized transformation that simulates pan, tilt, and rotation effects in imaging.                          |
| [`deepinv.transform.CPABDiffeomorphism`](https://deepinv.org/api/stubs/deepinv.transform.CPABDiffeomorphism.html.md#deepinv.transform.CPABDiffeomorphism)             | A continuous piecewise affine transformation allowing for smooth and invertible deformations across an image.    |
| [`deepinv.transform.rotate_via_shear()`](https://deepinv.org/api/stubs/deepinv.transform.rotate_via_shear.html.md#deepinv.transform.rotate_via_shear)               | A rotation implemented via shear operations for reduced interpolation artifacts.                                 |

## Video transforms

While all geometric transforms accept video input, the following transforms work specifically in the time dimension.
These can be easily compounded with geometric transformations using the `*` operation.

#### Time Transforms

| **Transform**                                                                                            | **Description**                         |
|----------------------------------------------------------------------------------------------------------|-----------------------------------------|
| [`deepinv.transform.ShiftTime`](https://deepinv.org/api/stubs/deepinv.transform.ShiftTime.html.md#deepinv.transform.ShiftTime) | A temporal shift in the time dimension. |

## Non-geometric transforms

Non-geometric transforms are often used for data augmentation.
Note that not all of these are necessarily invertible or form groups.

#### Non-geometric Transforms

| **Transform**                                                                                                          | **Description**                            |
|------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|
| [`deepinv.transform.RandomNoise`](https://deepinv.org/api/stubs/deepinv.transform.RandomNoise.html.md#deepinv.transform.RandomNoise)           | Add random noise to data (non-invertible). |
| [`deepinv.transform.RandomPhaseError`](https://deepinv.org/api/stubs/deepinv.transform.RandomPhaseError.html.md#deepinv.transform.RandomPhaseError) | Add random phase error to frequency data.  |
