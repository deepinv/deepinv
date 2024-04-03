r"""
Imaging inverse problems with adversarial networks
==================================================

This example shows you how to train various networks using adversarial
training for deblurring problems. We demonstrate running training and
inference using DeblurGAN, AmbientGAN and UAIR implemented in the
``deepinv`` library, and how to simply train your own GAN by using
``deepinv.training.AdversarialTrainer``. These examples can also be
easily extended to train more complicated GANs such as CycleGAN.

-  Kupyn et al., `DeblurGAN: Blind Motion Deblurring Using Conditional
   Adversarial
   Networks <https://openaccess.thecvf.com/content_cvpr_2018/papers/Kupyn_DeblurGAN_Blind_Motion_CVPR_2018_paper.pdf>`__
-  Bora et al., `AmbientGAN: Generative models from lossy
   measurements <https://openreview.net/forum?id=Hy7fDog0b>`__
-  Pajot et al., `Unsupervised Adversarial Image
   Reconstruction <https://openreview.net/forum?id=BJg4Z3RqF7>`__

Adversarial networks are characterised by the addition of an adversarial
loss :math:`\mathcal{L}_\text{adv}` to the standard reconstruction loss:

.. math:: \mathcal{L}_\text{adv}(x,\hat x;D)=\mathbb{E}_{x\sim p_x}\left[q(D(x))\right]+\mathbb{E}_{\hat x\sim p_{\hat x}}\left[q(1-D(\hat x))\right]

where :math:`D(\cdot)` is the discriminator model, :math:`x` is the
reference image, :math:`\hat x` is the estimated reconstruction,
:math:`q(\cdot)` is a quality function (e.g :math:`q(x)=x` for WGAN).
Training alternates between generator :math:`f` and discriminator
:math:`D` in a minimax game. When there are no ground truths (i.e
unsupervised), this may be defined on the measurements :math:`y`
instead.

**DeblurGAN** forward pass:

.. math:: \hat x = f(y)

**DeblurGAN** loss:

.. math:: \mathcal{L}=\mathcal{L}_\text{sup}(\hat x, x)+\mathcal{L}_\text{adv}(\hat x, x;D)

where :math:`\mathcal{L}_\text{sup}` is a supervised loss such as
pixel-wise MSE or VGG Perceptual Loss.

**AmbientGAN** forward pass:

.. math:: \hat x = f(z),\quad z\sim \mathcal{N}(\mathbf{0},\mathbf{I}_k)

**AmbientGAN** loss (where :math:`A(\cdot)` is the physics):

.. math:: \mathcal{L}=\mathcal{L}_\text{adv}(A(\hat x), y;D)

Forward pass at eval time:

.. math:: \hat x = f(\hat z)\quad\text{s.t.}\quad\hat z=\operatorname*{argmin}_z D(A(f(z)),y)

**UAIR** forward pass:

.. math:: \hat x = f(y)

**UAIR** loss:

.. math:: \mathcal{L}=\mathcal{L}_\text{adv}(\hat y, y;D)+\lVert A(f(\hat y))- \hat y\rVert^2_2,\quad\hat y=A(\hat x)

"""

import deepinv as dinv
import torch


# %%
# Test
# 

a = torch.Tensor([1])