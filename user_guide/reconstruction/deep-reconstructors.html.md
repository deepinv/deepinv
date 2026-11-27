<a id="deep-reconstructors"></a>

# Deep Reconstruction Models

The simplest method for reconstructing an image from measurements is to pass it through a feedforward
model architecture that is conditioned on the acquisition physics, that is $\inversef{y}{A}$. We offer a range of architectures for general and specific problems.

<a id="artifact"></a>

## Artifact Removal

The simplest reconstruction architecture first maps the measurements
to the image domain via a non-learned mapping, and then applies a denoiser network to the obtain the final reconstruction.

The [`deepinv.models.ArtifactRemoval`](https://deepinv.org/api/stubs/deepinv.models.ArtifactRemoval.html.md#deepinv.models.ArtifactRemoval) class converts a denoiser [`deepinv.models.Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser) or other image-to-image network $\phi$ into a
reconstruction network [`deepinv.models.Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor) $R$ by doing

- Adjoint: $\inversef{y}{A}=\phi(A^{\top}y)$ with `mode='adjoint'`.
  <br/>
  This option is generally to linear operators $A$.
  <br/>
- Pseudoinverse: $\inversef{y}{A}=\phi(A^{\dagger}y)$ with `mode='pinv'`.
- Direct: $\inversef{y}{A}=\phi(y)$ with `mode='direct'`.
  <br/>
  This option serves as a wrapper to obtain a [`Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor), and can be used to adapt a generic denoiser or image-to-image network into one that is specific to an inverse problem.
  <br/>

<a id="general-reconstructors"></a>

## General reconstruction models

We provide the following list of reconstruction models trained on multiple various physics and datasets
to provide robustness to different problems.

See [Description of weights](https://deepinv.org/user_guide/reconstruction/pretrained-models.html.md#pretrained-weights) for more information on pretrained denoisers.

#### Multiphysics reconstruction models

| Model                                                                                  | Type     | Tensor Size (C, H, W)   | Pretrained Weights   | Noise level aware   |
|----------------------------------------------------------------------------------------|----------|-------------------------|----------------------|---------------------|
| [`deepinv.models.RAM`](https://deepinv.org/api/stubs/deepinv.models.RAM.html.md#deepinv.models.RAM) | CNN-UNet | C=1, 2, 3; H,W>8        | C=1, 2, 3            | Yes                 |

<a id="specific-reconstructors"></a>

## Specific reconstruction models

We also provide some architectures for specific inverse problems.

#### Specific architectures

| Model                                                                                            | Description                          |
|--------------------------------------------------------------------------------------------------|--------------------------------------|
| [`deepinv.models.PanNet`](https://deepinv.org/api/stubs/deepinv.models.PanNet.html.md#deepinv.models.PanNet)     | PanNet model for pansharpening.      |
| [`deepinv.models.SRResNet`](https://deepinv.org/api/stubs/deepinv.models.SRResNet.html.md#deepinv.models.SRResNet) | SRResNet model for super-resolution. |
