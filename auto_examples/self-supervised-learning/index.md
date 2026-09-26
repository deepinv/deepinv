<a id="sphx-glr-auto-examples-self-supervised-learning"></a>

# Self-Supervised Learning

<div id='sg-tag-list' class='sphx-glr-tag-list'></div><div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates various geometric image transformations implemented in deepinv that can be used in Equivariant Imaging (EI) for self-supervised learning:">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_ei_transforms_thumb.png)

[Image transformations for Equivariant Imaging](https://deepinv.org/auto_examples/self-supervised-learning/demo_ei_transforms.md)

  <div class="sphx-glr-thumbnail-title">Image transformations for Equivariant Imaging</div>
</div><div class="sphx-glr-thumbcontainer" data-sgtags='["MRI", "Denoising", "Foundation model"]' tooltip="We demonstrate self-supervised (blind) denoising of a low-field MRI scan without ground truth data.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_lowfieldmri_thumb.png)

[Low-field MRI denoising without ground truth](https://deepinv.org/auto_examples/self-supervised-learning/demo_lowfieldmri.md)

  <div class="sphx-glr-thumbnail-title">Low-field MRI denoising without ground truth</div>
</div><div class="sphx-glr-thumbcontainer" data-sgtags='["Denoising"]' tooltip="This code shows how to restore a single image corrupted by Poisson noise using Poisson2Sparse, without requiring external training or knowledge of the noise level.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_poisson2sparse_thumb.png)

[Poisson denoising using Poisson2Sparse](https://deepinv.org/auto_examples/self-supervised-learning/demo_poisson2sparse.md)

  <div class="sphx-glr-thumbnail-title">Poisson denoising using Poisson2Sparse</div>
</div><div class="sphx-glr-thumbcontainer" data-sgtags='["MRI", "Foundation model"]' tooltip="We demonstrate scan-specific self-supervised learning, that is, learning to reconstruct MRI scans from a single accelerated sample without ground truth.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_scan_specific_thumb.png)

[Scan-specific zero-shot SSDU for MRI](https://deepinv.org/auto_examples/self-supervised-learning/demo_scan_specific.md)

  <div class="sphx-glr-thumbnail-title">Scan-specific zero-shot SSDU for MRI</div>
</div><div class="sphx-glr-thumbcontainer" data-sgtags='["MRI"]' tooltip="We demonstrate the self-supervised Artifact2Artifact loss for solving an undersampled sequential MRI reconstruction problem without ground truth.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_artifact2artifact_thumb.png)

[Self-supervised MRI reconstruction with Artifact2Artifact](https://deepinv.org/auto_examples/self-supervised-learning/demo_artifact2artifact.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised MRI reconstruction with Artifact2Artifact</div>
</div><div class="sphx-glr-thumbcontainer" data-sgtags='["Denoising"]' tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, using noisy images only via the Generalized Recorrupted2Recorrupted (GR2R) loss monroy2025generalized, which exploits knowledge about the noise distribution. You can change the noise distribution by selecting from predefined noise models such as Gaussian, Poisson, and Gamma noise.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_r2r_denoising_thumb.png)

[Self-supervised denoising with the Generalized R2R loss.](https://deepinv.org/auto_examples/self-supervised-learning/demo_r2r_denoising.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the Generalized R2R loss.</div>
</div><div class="sphx-glr-thumbcontainer" data-sgtags='["Denoising"]' tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images only via the Neighbor2Neighbor loss, which exploits the local correlation of natural images.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_n2n_denoising_thumb.png)

[Self-supervised denoising with the Neighbor2Neighbor loss.](https://deepinv.org/auto_examples/self-supervised-learning/demo_n2n_denoising.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the Neighbor2Neighbor loss.</div>
</div><div class="sphx-glr-thumbcontainer" data-sgtags='["Denoising"]' tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images only via the SURE loss, which exploits knowledge about the noise distribution.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_sure_denoising_thumb.png)

[Self-supervised denoising with the SURE loss.](https://deepinv.org/auto_examples/self-supervised-learning/demo_sure_denoising.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the SURE loss.</div>
</div><div class="sphx-glr-thumbcontainer" data-sgtags='["Denoising"]' tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images with unknown noise level only via the UNSURE loss, which is introduced by tachella2024unsure.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_unsure_thumb.png)

[Self-supervised denoising with the UNSURE loss.](https://deepinv.org/auto_examples/self-supervised-learning/demo_unsure.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the UNSURE loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a reconstruction network for an inpainting inverse problem on a fully self-supervised way, i.e., using measurement data only.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_multioperator_imaging_thumb.png)

[Self-supervised learning from incomplete measurements of multiple operators.](https://deepinv.org/auto_examples/self-supervised-learning/demo_multioperator_imaging.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised learning from incomplete measurements of multiple operators.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a reconstruction network for an MRI inverse problem on a fully self-supervised way, i.e., using measurement data only.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_equivariant_imaging_thumb.png)

[Self-supervised learning with Equivariant Imaging for MRI.](https://deepinv.org/auto_examples/self-supervised-learning/demo_equivariant_imaging.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Imaging for MRI.</div>
</div><div class="sphx-glr-thumbcontainer" data-sgtags='["Foundation model"]' tooltip="Equivariant splitting consists in minimizing a self-supervised loss to train a reconstruction model using measurement data only sechaud26Equivariant.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_equivariant_splitting_thumb.png)

[Self-supervised learning with Equivariant Splitting](https://deepinv.org/auto_examples/self-supervised-learning/demo_equivariant_splitting.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Splitting</div>
</div><div class="sphx-glr-thumbcontainer" data-sgtags='["Denoising"]' tooltip="We demonstrate self-supervised learning with measurement splitting, to train a denoiser network on the MNIST dataset. The physics here is noisy computed tomography, as is the case in Noise2Inverse hendriksen2020noise2inverse. Note this example can also be easily applied to undersampled multicoil MRI as is the case in SSDU yaman2020self.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_splitting_loss_thumb.png)

[Self-supervised learning with measurement splitting](https://deepinv.org/auto_examples/self-supervised-learning/demo_splitting_loss.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised learning with measurement splitting</div>
</div><div class="sphx-glr-thumbcontainer" data-sgtags='["Denoising", "Ultrasound"]' tooltip="This example despeckles real clinical ultrasound B-mode images with Speckle2Self li2025speckle2self, a model pretrained without clean reference images.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_ultrasound_despeckling_thumb.png)

[Ultrasound despeckling from B-mode images](https://deepinv.org/auto_examples/self-supervised-learning/demo_ultrasound_despeckling.md)

  <div class="sphx-glr-thumbnail-title">Ultrasound despeckling from B-mode images</div>
</div>
<!-- thumbnail-parent-div-close --></div>
