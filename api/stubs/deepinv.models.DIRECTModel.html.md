# DIRECTModel

### *class* deepinv.models.DIRECTModel(model_name='jointicnet_5x', pretrained=True, device='cpu')

Bases: [`Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor), [`MRIMixin`](https://deepinv.org/api/stubs/deepinv.utils.MRIMixin.html.md#deepinv.utils.MRIMixin)

Pretrained MRI reconstruction models from DIRECT library.

Runs a pretrained model from [DIRECT](https://github.com/NKI-AI/direct) for (multi-coil) MRI reconstruction from kspace to images.

Available models:

**Models trained on Calgary-Campinas 12-coil brain** (downloaded from [here](https://huggingface.co/NKI-AI/direct-calgary-campinas))

- `jointicnet_5x` (or `_10x`) <sup>[1](#footcite-jun2021joint)</sup>,
- `recurrentvarnet_5x` (or `_10x`) <sup>[2](#footcite-yiasemis2021recurrent)</sup>,
- `varnet_5x` (or `_10x`) <sup>[3](#footcite-sriram2020end)</sup>,
- `conjgradnet_5x` (or `_10x`) <sup>[4](#footcite-shewchuk1994conjugate)</sup>,
- `iterdualnet_5x` (or `_10x`) <sup>[5](#footcite-moriakov2026conditional)</sup>,
- `kikinet_5x` (or `_10x`) <sup>[6](#footcite-eo2018kiki)</sup>,
- `lpdnet_5x` (or `_10x`) <sup>[7](#footcite-adler2018learned)</sup>,
- `unet_5x` (or `_10x`) <sup>[8](#footcite-ronneberger2015unet)</sup>,
- `xpdnet_5x` (or `_10x`) <sup>[9](#footcite-ramzi2021xpdnet)</sup>,
- `multidomainnet` <sup>[10](#footcite-muckley2021results)</sup> (downloaded from [DIRECT](https://files.aiforoncology.nl/direct-project), repaired locally, uploaded to [HF](https://huggingface.co/Andrewwango/direct)).

**Models trained on a mix of MRI datasets (including brain, cardiac, knee and prostate)** (downloaded from [here](https://huggingface.co/NKI-AI/direct-uniform))

- `vsharp_brain` <sup>[11](#footcite-yiasemis2024vsharp)</sup><sup>[12](#footcite-yiasemis2025uniform)</sup>,
- `vsharp_cardiac`,
- `vsharp_knee`,
- `vsharp_prostate`

The wrapped models handle the MRI physics and estimate coil maps themselves.

#### NOTE
deepinv uses centered FFTs but DIRECT uses uncentered, so we pre-shift `y` (a checkerboard modulation) into
DIRECT’s convention.

Also, the output scale is not preserved, so its intensity is proportional to but not equal to `y`.

#### NOTE
This model requires DIRECT >=2.2.0 and Python >=3.12. Install it with `pip install deepinv[direct]`.
The model should be used on non-CPU devices.
CPU support is experimental and depends on your platform due to DIRECT requirement constraints.

#### WARNING
Currently, this model can only be used for evaluation, not training/fine-tuning. If you want to use the model in training mode, please open a feature request issue on GitHub.

* **Parameters:**
  * **model_name** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – model name, see list above.
  * **pretrained** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – If `True`, the model will be initialized with pretrained weights from DIRECT. If `str`, load from file.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – device.

<hr />

* **Example:**
  ```pycon
  >>> import deepinv as dinv
  >>> model = dinv.models.DIRECTModel("vsharp_brain")
  >>> x_hat = model(y, physics)  # y: multicoil k-space, physics: dinv.physics.MultiCoilMRI
  ```

<hr />

* **References:**

* <a id='footcite-jun2021joint'>**[1]**</a> Yohan Jun, Hyungseob Shin, Taejoon Eo, and Dosik Hwang. Joint deep model-based mr image and coil sensitivity reconstruction network (joint-icnet) for fast mri. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 5270–5279. 2021.
* <a id='footcite-yiasemis2021recurrent'>**[2]**</a> George Yiasemis, Jan-Jakob Sonke, Clara Sánchez, and Jonas Teuwen. Recurrent variational network: a deep learning inverse problem solver applied to the task of accelerated mri reconstruction. In *Proceedings of the 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2021.
* <a id='footcite-sriram2020end'>**[3]**</a> Anuroop Sriram, Jure Zbontar, Tullie Murrell, Aaron Defazio, C Lawrence Zitnick, Nafissa Yakubova, Florian Knoll, and Patricia Johnson. End-to-end variational networks for accelerated mri reconstruction. In *Medical image computing and computer assisted intervention–MICCAI 2020: 23rd international conference, Lima, Peru, October 4–8, 2020, proceedings, part II 23*, 64–73. Springer, 2020.
* <a id='footcite-shewchuk1994conjugate'>**[4]**</a> Jonathan Richard Shewchuk. An introduction to the conjugate gradient method without the agonizing pain. Technical Report, Carnegie Mellon University, 1994.
* <a id='footcite-moriakov2026conditional'>**[5]**</a> Nikita Moriakov, George Yiasemis, Jan-Jakob Sonke, and Jonas Teuwen. Conditional learned reconstruction for accelerated mri. In *Proceedings of Machine Learning Research*, volume 315, 754–780. 2026.
* <a id='footcite-eo2018kiki'>**[6]**</a> Taejoon Eo, Yohan Jun, Taeseong Kim, Jinseong Jang, Ho-Joon Lee, and Dosik Hwang. Kiki-net: cross-domain convolutional neural networks for reconstructing undersampled magnetic resonance images. *Magnetic Resonance in Medicine*, 80(5):2188–2201, 2018.
* <a id='footcite-adler2018learned'>**[7]**</a> Jonas Adler and Ozan Öktem. Learned primal-dual reconstruction. *IEEE transactions on medical imaging*, 37(6):1322–1332, 2018.
* <a id='footcite-ronneberger2015unet'>**[8]**</a> Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: convolutional networks for biomedical image segmentation. In *Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015*, 234–241. Springer, 2015.
* <a id='footcite-ramzi2021xpdnet'>**[9]**</a> Zaccharie Ramzi, Philippe Ciuciu, and Jean-Luc Starck. Xpdnet for mri reconstruction: an application to the 2020 fastmri challenge. In *ISMRM 2021*. 2021.
* <a id='footcite-muckley2021results'>**[10]**</a> Matthew J Muckley, Bruno Riemenschneider, Alireza Radmanesh, Sunwoo Kim, Geunu Jeong, Jingyu Ko, Yohan Jun, Hyungseob Shin, Dosik Hwang, Mahmoud Mostapha, and others. Results of the 2020 fastmri challenge for machine learning mr image reconstruction. *IEEE Transactions on Medical Imaging*, 40(9):2306–2317, 2021.
* <a id='footcite-yiasemis2024vsharp'>**[11]**</a> George Yiasemis, Nikita Moriakov, Jan-Jakob Sonke, and Jonas Teuwen. Vsharp: variable splitting half-quadratic admm algorithm for reconstruction of inverse-problems. *Magnetic Resonance Imaging*, 2024.
* <a id='footcite-yiasemis2025uniform'>**[12]**</a> George Yiasemis, Jonatan Ferm, Nikita Moriakov, Ritse Mann, Jan-Jakob Sonke, and Jonas Teuwen. Uniform: a unified deep learning framework for multi-organ and multi-contrast mri reconstruction. In *The 8th International Conference on Medical Imaging with Deep Learning*. 2025.

#### forward(y, physics, \*\*kwargs)

Reconstruct image from k-space `y` and `physics`.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – k-space of shape `(B,2,N,H,W)` for multicoil or `(B,2,H,W)` for singlecoil MRI.
  * **physics** ([*deepinv.physics.MultiCoilMRI*](https://deepinv.org/api/stubs/deepinv.physics.MultiCoilMRI.html.md#deepinv.physics.MultiCoilMRI) *,* [*deepinv.physics.MRI*](https://deepinv.org/api/stubs/deepinv.physics.MRI.html.md#deepinv.physics.MRI)) – MRI physics with mask (coil maps ignored).

<a id="sphx-glr-backref-deepinv-models-directmodel"></a>

## Examples using `DIRECTModel`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo reconstructs undersampled k-space data for 2D cardiac and brain MRI on:">![](auto_examples/models/images/thumb/sphx_glr_demo_mri_pretrained_thumb.png)

[Reconstruct undersampled k-space for cardiac and brain MRI](https://deepinv.org/auto_examples/models/demo_mri_pretrained.html.md)

  <div class="sphx-glr-thumbnail-title">Reconstruct undersampled k-space for cardiac and brain MRI</div>
</div>
<!-- thumbnail-parent-div-close --></div>
