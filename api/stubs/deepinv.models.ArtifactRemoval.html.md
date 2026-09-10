# ArtifactRemoval

### *class* deepinv.models.ArtifactRemoval(backbone_net, mode='adjoint', ckpt_path=None, device=None)

Bases: [`Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor)

Artifact removal architecture.

Transforms a denoiser $\phi$ into a reconstruction network $R$ by doing

- Adjoint: $\inversef{y}{A}=\phi(A^{\top}y)$ with `mode='adjoint'`.
- Pseudoinverse: $\inversef{y}{A}=\phi(A^{\dagger}y)$ with `mode='pinv'`.
- Direct: $\inversef{y}{A}=\phi(y)$ with `mode='direct'`.

#### NOTE
In the case of `mode='pinv'`, the architecture is inspired by the FBPConvNet
approach Jin *et al.*<sup>[1](#footcite-jin2017deep)</sup> where a deep network $\phi$
is used to improve the filtered back projection $A^{\dagger}y$.

* **Parameters:**
  * **backbone_net** ([*deepinv.models.Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser) *,* [*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Base denoiser network $\phi$
    (see [Denoisers](https://deepinv.org/user_guide/reconstruction/denoisers.html.md#denoisers) for available architectures).
  * **mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Reconstruction mode. Options are ‘direct’, ‘adjoint’ or ‘pinv’.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – cpu or gpu.

<hr />

* **References:**

* <a id='footcite-jin2017deep'>**[1]**</a> Kyong Hwan Jin, Michael T McCann, Emmanuel Froustey, and Michael Unser. Deep convolutional neural network for inverse problems in imaging. *IEEE transactions on image processing*, 26(9):4509–4522, 2017.

#### backbone_inference(tensor_in, physics, y, \*\*kwargs)

Perform inference on the backbone network.

By default, treats backbone network as a denoiser.
Override for different inference e.g. for an unrolled network.

* **Parameters:**
  * **tensor_in** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor as dictated by ArtifactRemoval mode
  * **physics** ([*Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – forward physics
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input measurements y
* **Returns:**
  reconstructed image
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### forward(y, physics, \*\*kwargs)

Reconstructs a signal estimate from measurements y

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – forward operator
  * **kwargs** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – additional keyword arguments for the backbone network.
* **Returns:**
  reconstructed image
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-models-artifactremoval"></a>

## Examples using `ArtifactRemoval`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example provides a very simple quick start introduction to training reconstruction networks with DeepInverse for solving imaging inverse problems.">  <div class="sphx-glr-thumbnail-title">Training a reconstruction model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a reconstruction network for an inpainting inverse problem on a fully self-supervised way, i.e., using measurement data only.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning from incomplete measurements of multiple operators.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images only via the Neighbor2Neighbor loss, which exploits the local correlation of natural images.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the Neighbor2Neighbor loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, using noisy images only via the Generalized Recorrupted2Recorrupted (GR2R) loss :footcitemonroy2025generalized, which exploits knowledge about the noise distribution. You can change the noise distribution by selecting from predefined noise models such as Gaussian, Poisson, and Gamma noise.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the Generalized R2R loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised learning with measurement splitting, to train a denoiser network on the MNIST dataset. The physics here is noisy computed tomography, as is the case in Noise2Inverse :footcitehendriksen2020noise2inverse. Note this example can also be easily applied to undersampled multicoil MRI as is the case in SSDU :footciteyaman2020self.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with measurement splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images only via the SURE loss, which exploits knowledge about the noise distribution.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the SURE loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images with unknown noise level only via the UNSURE loss, which is introduced by :footcitetachella2024unsure.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the UNSURE loss.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
