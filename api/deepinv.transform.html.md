# deepinv.transform

This module contains different transforms which can be used for data augmentation or together with the equivariant losses.
Please refer to the [user guide](https://deepinv.org/user_guide/training/transforms.html.md#transform) for more information.

## Base class

**User Guide:** refer to [Transforms](https://deepinv.org/user_guide/training/transforms.html.md#transform) for more information.

| [`deepinv.transform.Transform`](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform)   | Base class for image transforms.   |
|------------------------------------------------------------------------------------------------------------|------------------------------------|

## Simple transforms

| [`deepinv.transform.Rotate`](https://deepinv.org/api/stubs/deepinv.transform.Rotate.html.md#deepinv.transform.Rotate)     | 2D Rotations.                           |
|--------------------------------------------------------------------------------------------------------|-----------------------------------------|
| [`deepinv.transform.Shift`](https://deepinv.org/api/stubs/deepinv.transform.Shift.html.md#deepinv.transform.Shift)       | Fast integer 2D translations.           |
| [`deepinv.transform.Scale`](https://deepinv.org/api/stubs/deepinv.transform.Scale.html.md#deepinv.transform.Scale)       | 2D Scaling.                             |
| [`deepinv.transform.Reflect`](https://deepinv.org/api/stubs/deepinv.transform.Reflect.html.md#deepinv.transform.Reflect)   | Reflect (flip) in random multiple axes. |
| [`deepinv.transform.Identity`](https://deepinv.org/api/stubs/deepinv.transform.Identity.html.md#deepinv.transform.Identity) | Identity transform i.e. trivial group.  |

## Advanced transforms

| [`deepinv.transform.Homography`](https://deepinv.org/api/stubs/deepinv.transform.Homography.html.md#deepinv.transform.Homography)                             | Random projective transformations (homographies).                                          |
|----------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| [`deepinv.transform.projective.Euclidean`](https://deepinv.org/api/stubs/deepinv.transform.projective.Euclidean.html.md#deepinv.transform.projective.Euclidean)         | Random Euclidean image transformations using projective transformation framework.          |
| [`deepinv.transform.projective.Similarity`](https://deepinv.org/api/stubs/deepinv.transform.projective.Similarity.html.md#deepinv.transform.projective.Similarity)       | Random 2D similarity image transformations using projective transformation framework.      |
| [`deepinv.transform.projective.Affine`](https://deepinv.org/api/stubs/deepinv.transform.projective.Affine.html.md#deepinv.transform.projective.Affine)               | Random affine image transformations using projective transformation framework.             |
| [`deepinv.transform.projective.PanTiltRotate`](https://deepinv.org/api/stubs/deepinv.transform.projective.PanTiltRotate.html.md#deepinv.transform.projective.PanTiltRotate) | Random 3D camera rotation image transformations using projective transformation framework. |
| [`deepinv.transform.CPABDiffeomorphism`](https://deepinv.org/api/stubs/deepinv.transform.CPABDiffeomorphism.html.md#deepinv.transform.CPABDiffeomorphism)             | Continuous Piecewise-Affine-based Diffeomorphism.                                          |
| [`deepinv.transform.rotate_via_shear`](https://deepinv.org/api/stubs/deepinv.transform.rotate_via_shear.html.md#deepinv.transform.rotate_via_shear)                 | 2D rotation of image by angle via shear composition through FFT.                           |

## Video transforms

While all geometric transforms accept video input, the following transforms work specifically in the time dimension.
These can be easily compounded with geometric transformations using the `*` operation.

| [`deepinv.transform.ShiftTime`](https://deepinv.org/api/stubs/deepinv.transform.ShiftTime.html.md#deepinv.transform.ShiftTime)   | Shift a video in time with reflective padding.   |
|------------------------------------------------------------------------------------------------------------|--------------------------------------------------|

## Non-geometric transforms

Non-geometric transforms are often used for data augmentation.
Note that not all of these are necessarily invertible or form groups.

| [`deepinv.transform.RandomNoise`](https://deepinv.org/api/stubs/deepinv.transform.RandomNoise.html.md#deepinv.transform.RandomNoise)           | Random noise transform.       |
|------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| [`deepinv.transform.RandomPhaseError`](https://deepinv.org/api/stubs/deepinv.transform.RandomPhaseError.html.md#deepinv.transform.RandomPhaseError) | Random phase error transform. |
