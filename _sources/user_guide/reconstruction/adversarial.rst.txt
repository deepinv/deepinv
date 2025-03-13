.. _adversarial:

Adversarial Networks
====================
There are two types of adversarial networks for imaging: conditional and unconditional.
See :ref:`sphx_glr_auto_examples_adversarial-learning_demo_gan_imaging.py` for examples.
Adversarial training can be done using the :class:`deepinv.training.AdversarialTrainer` class,
which is a subclass of :class:`deepinv.Trainer`.

Conditional GAN
---------------
Conditional generative adversarial networks (cGANs) aim to learn a reconstruction
network :math:`\hat{x}=R(y,A,z)`, which maps the measurements :math:`y` to the signal :math:`x`,
possibly conditioned on a random variable :math:`z` and the forward operator :math:`A`,
which strikes a good trade-off between distortion :math:`\|x-\hat{x}\|^2` and perception (how close
are the distribution of reconstructed and clean images :math:`p_{\hat{x}}` and :math:`p_x`).

They are trained by adding an adversarial
loss :math:`\mathcal{L}_\text{adv}` to the standard reconstruction loss:

.. math:: \mathcal{L}_\text{total}=\mathcal{L}_\text{rec}+\lambda\mathcal{L}_\text{adv}

where :math:`\lambda` is a hyperparameter that balances the two losses. The reconstruction loss
is often a mean squared error (MSE) :math:`\mathcal{L}_\text{rec}(x,\hat{x})=\|x-\hat{x}\|^2` (or a self-supervised alternative),
while the adversarial loss is

.. math:: \mathcal{L}_\text{adv}(x,\hat x;D)=\mathbb{E}_{x\sim p_x}\left[q(D(x))\right]+\mathbb{E}_{\hat x\sim p_{\hat x}}\left[q(1-D(\hat x))\right]

where :math:`D(\cdot)` is the discriminator model, :math:`x` is the reference image, :math:`\hat{x}` is the
estimated reconstruction, :math:`q(\cdot)` is a quality function (e.g :math:`q(x)=x` for WGAN).
Training alternates between generator :math:`G` and discriminator :math:`D` in a minimax game.
When there are no ground truths (i.e. self-supervised), this may be defined on the measurements :math:`y` instead.
See the list of available adversarial losses in :ref:`adversarial-losses`.

The reconstruction network (i.e. the "generator") :math:`R` can be any architecture that maps the measurements :math:`y` to the signal :math:`x`,
including :ref:`artifact removal <artifact>` or :ref:`unfolded <unfolded>` networks.

The discriminator network :math:`D` can be implemented with one of the following architectures:

.. list-table:: Discriminator Networks
   :header-rows: 1

   * - Discriminator
     - Description
   * - :class:`DCGANDiscriminator <deepinv.models.DCGANDiscriminator>`
     - Deep Convolution GAN discriminator model
   * - :class:`ESRGANDiscriminator <deepinv.models.ESRGANDiscriminator>`
     - Enhanced Super-Resolution GAN discriminator model
   * - :class:`PatchGANDiscriminator <deepinv.models.PatchGANDiscriminator>`
     - PatchGAN discriminator model


Unconditional GAN
-----------------
Unconditional generative adversarial networks train a generator network :math:`\hat{x}=G(z)` to map
a simple distribution :math:`p_z` (e.g., Gaussian) to the signal distribution :math:`p_x`.
The generator is trained with an adversarial loss:

.. math:: \mathcal{L}_\text{total}=\mathcal{L}_\text{adv}(\hat x, x;D)

See the list of available adversarial losses in :ref:`adversarial-losses`, including CSGM and AmbientGAN training.

Once the generator is trained, we can solve inverse problems by looking for a latent :math:`z` that
matches the observed measurements :math:`\forw{R(z)}\approx y`:

.. math:: \hat x = \inverse{\hat z}\quad\text{s.t.}\quad\hat z=\operatorname*{argmin}_z \lVert \forw{\inverse{z}}-y\rVert _2^2

We can adapt any latent generator model to train an unconditional GAN and perform conditional inference:

.. list-table:: Unconditional GANs
   :header-rows: 1

   * - Generator
     - Description
   * - :class:`DCGANGenerator <deepinv.models.DCGANGenerator>`
     - DCGAN unconditional generator model
   * - :class:`CSGMGenerator <deepinv.models.CSGMGenerator>`
     - Adapts an unconditional generator model for CSGM or AmbientGAN training.


.. _deep-image-prior:

Deep Image Prior
~~~~~~~~~~~~~~~~

The :class:`deep image prior <deepinv.models.DeepImagePrior>` uses an untrained convolutional decoder network as :math:`R` applied to a random input :math:`z`.
The choice of the architecture of :math:`R` is crucial for the success of the method: we provide the
:class:`deepinv.models.ConvDecoder` architecture, which is based on a convolutional decoder network,
and has shown good inductive bias for image reconstruction tasks.
