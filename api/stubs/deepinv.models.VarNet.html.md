# VarNet

### *class* deepinv.models.VarNet(denoiser=None, sensitivity_model=None, num_cascades=12, mode='varnet')

Bases: [`ArtifactRemoval`](https://deepinv.org/api/stubs/deepinv.models.ArtifactRemoval.html.md#deepinv.models.ArtifactRemoval), [`MRIMixin`](https://deepinv.org/api/stubs/deepinv.utils.MRIMixin.html.md#deepinv.utils.MRIMixin)

VarNet or E2E-VarNet model.

These models are from the papers Sriram *et al.*<sup>[1](#footcite-sriram2020end)</sup> and Hammernik *et al.*<sup>[2](#footcite-hammernik2018learning)</sup>.
This performs unrolled iterations on the image estimate x (as per the original VarNet paper)
or the kspace y (as per E2E-VarNet).

#### NOTE
For singlecoil MRI, either mode is valid.
For multicoil MRI, the VarNet mode will simply sum over the coils (not preferred). Using E2E-VarNet is therefore preferred.
For sensitivity-map estimation for multicoil MRI, pass in `sensitivity_model`.

Code loosely adapted from E2E-VarNet implementation from [https://github.com/facebookresearch/fastMRI/blob/main/fastmri/models/varnet.py](https://github.com/facebookresearch/fastMRI/blob/main/fastmri/models/varnet.py).

* **Parameters:**
  * **denoiser** ([*Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser) *,* [*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – backbone network that parametrizes the grad of the regulariser.
    If `None`, a small DnCNN is used.
  * **sensitivity_model** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – network to jointly estimate coil sensitivity maps for multi-coil MRI. If `None`, do not perform any map estimation. For single-coil MRI, unused.
  * **num_cascades** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of unrolled iterations (‘cascades’).
  * **mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – if ‘varnet’, perform iterates on the images x as in original VarNet.
    If ‘e2e-varnet’, perform iterates on the kspace y as in the E2E-VarNet.

<hr />

* **References:**

* <a id='footcite-sriram2020end'>**[1]**</a> Anuroop Sriram, Jure Zbontar, Tullie Murrell, Aaron Defazio, C Lawrence Zitnick, Nafissa Yakubova, Florian Knoll, and Patricia Johnson. End-to-end variational networks for accelerated mri reconstruction. In *Medical image computing and computer assisted intervention–MICCAI 2020: 23rd international conference, Lima, Peru, October 4–8, 2020, proceedings, part II 23*, 64–73. Springer, 2020.
* <a id='footcite-hammernik2018learning'>**[2]**</a> Kerstin Hammernik, Teresa Klatzer, Erich Kobler, Michael P Recht, Daniel K Sodickson, Thomas Pock, and Florian Knoll. Learning a variational network for reconstruction of accelerated mri data. *Magnetic resonance in medicine*, 79(6):3055–3071, 2018.

#### backbone_inference(tensor_in, physics, y)

Perform inference on input tensor.

Uses physics and y for data consistency.
If necessary, perform fully-sampled MRI IFFT on model output.

* **Parameters:**
  * **tensor_in** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor as dictated by VarNet mode (either k-space or image)
  * **physics** ([*Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – forward physics for data consistency
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input measurements y for data consistency
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) reconstructed image
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-models-varnet"></a>

## Examples using `VarNet`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div>
<!-- thumbnail-parent-div-close --></div>
