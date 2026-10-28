<a id="adversarial"></a>

# Adversarial Networks

There are two types of adversarial networks for imaging: conditional and unconditional.
See [Imaging inverse problems with adversarial networks](https://deepinv.org/auto_examples/adversarial-learning/demo_gan_imaging.html.md#sphx-glr-auto-examples-adversarial-learning-demo-gan-imaging-py) for examples.
Adversarial training can be done using the [`deepinv.training.AdversarialTrainer`](https://deepinv.org/api/stubs/deepinv.training.AdversarialTrainer.html.md#deepinv.training.AdversarialTrainer) class,
which is a subclass of [`deepinv.Trainer`](https://deepinv.org/api/stubs/deepinv.Trainer.html.md#deepinv.Trainer).

## Conditional GAN

Conditional generative adversarial networks (cGANs) aim to learn a reconstruction
network $\hat{x}=R(y,A,z)$, which maps the measurements $y$ to the signal $x$,
possibly conditioned on a random variable $z$ and the forward operator $A$,
which strikes a good trade-off between distortion $\|x-\hat{x}\|^2$ and perception (how close
are the distribution of reconstructed and clean images $p_{\hat{x}}$ and $p_x$).

They are trained by adding an adversarial
loss $\mathcal{L}_\text{adv}$ to the standard reconstruction loss:

$$
\mathcal{L}_\text{total}=\mathcal{L}_\text{rec}+\lambda\mathcal{L}_\text{adv}

$$

where $\lambda$ is a hyperparameter that balances the two losses. The reconstruction loss
is often a mean squared error (MSE) $\mathcal{L}_\text{rec}(x,\hat{x})=\|x-\hat{x}\|^2$ (or a self-supervised alternative),
while the adversarial loss is

$$
\mathcal{L}_\text{adv}(x,\hat x;D)=\mathbb{E}_{x\sim p_x}\left[q(D(x))\right]+\mathbb{E}_{\hat x\sim p_{\hat x}}\left[q(1-D(\hat x))\right]

$$

where $D(\cdot)$ is the discriminator model, $x$ is the reference image, $\hat{x}$ is the
estimated reconstruction, $q(\cdot)$ is a quality function (e.g $q(x)=x$ for WGAN).
Training alternates between generator $G$ and discriminator $D$ in a minimax game.
When there are no ground truths (i.e. self-supervised), this may be defined on the measurements $y$ instead.
See the list of available adversarial losses in [Adversarial Learning](https://deepinv.org/user_guide/training/loss.html.md#adversarial-losses).

The reconstruction network (i.e. the “generator”) $R$ can be any architecture that maps the measurements $y$ to the signal $x$,
including [artifact removal](https://deepinv.org/user_guide/reconstruction/deep-reconstructors.html.md#artifact) or [unfolded](https://deepinv.org/user_guide/reconstruction/unfolded.html.md#unfolded) networks.

The discriminator network $D$ can be implemented with one of the following architectures:

#### Discriminator Networks

| Discriminator                                                                                               | Description                                       |
|-------------------------------------------------------------------------------------------------------------|---------------------------------------------------|
| [`DCGANDiscriminator`](https://deepinv.org/api/stubs/deepinv.models.DCGANDiscriminator.html.md#deepinv.models.DCGANDiscriminator)       | Deep Convolution GAN discriminator model          |
| [`ESRGANDiscriminator`](https://deepinv.org/api/stubs/deepinv.models.ESRGANDiscriminator.html.md#deepinv.models.ESRGANDiscriminator)     | Enhanced Super-Resolution GAN discriminator model |
| [`PatchGANDiscriminator`](https://deepinv.org/api/stubs/deepinv.models.PatchGANDiscriminator.html.md#deepinv.models.PatchGANDiscriminator) | PatchGAN discriminator model                      |

## Unconditional GAN

Unconditional generative adversarial networks train a generator network $\hat{x}=G(z)$ to map
a simple distribution $p_z$ (e.g., Gaussian) to the signal distribution $p_x$.
The generator is trained with an adversarial loss:

$$
\mathcal{L}_\text{total}=\mathcal{L}_\text{adv}(\hat x, x;D)

$$

See the list of available adversarial losses in [Adversarial Learning](https://deepinv.org/user_guide/training/loss.html.md#adversarial-losses), including CSGM and AmbientGAN training.

Once the generator is trained, we can solve inverse problems by looking for a latent $z$ that
matches the observed measurements $\forw{R(z)}\approx y$:

$$
\hat x = \inverse{\hat z}\quad\text{s.t.}\quad\hat z=\operatorname*{argmin}_z \lVert \forw{\inverse{z}}-y\rVert _2^2

$$

We can adapt any latent generator model to train an unconditional GAN and perform conditional inference:

#### Unconditional GANs

| Generator                                                                                     | Description                                                              |
|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| [`DCGANGenerator`](https://deepinv.org/api/stubs/deepinv.models.DCGANGenerator.html.md#deepinv.models.DCGANGenerator) | DCGAN unconditional generator model                                      |
| [`CSGMGenerator`](https://deepinv.org/api/stubs/deepinv.models.CSGMGenerator.html.md#deepinv.models.CSGMGenerator)   | Adapts an unconditional generator model for CSGM or AmbientGAN training. |

<a id="deep-image-prior"></a>

### Deep Image Prior

The [`deep image prior`](https://deepinv.org/api/stubs/deepinv.models.DeepImagePrior.html.md#deepinv.models.DeepImagePrior) uses an untrained convolutional decoder network as $R$ applied to a random input $z$.
The choice of the architecture of $R$ is crucial for the success of the method: we provide the
[`deepinv.models.ConvDecoder`](https://deepinv.org/api/stubs/deepinv.models.ConvDecoder.html.md#deepinv.models.ConvDecoder) architecture, which is based on a convolutional decoder network,
and has shown good inductive bias for image reconstruction tasks.

For single-image Poisson denoising specifically, [`Poisson2Sparse`](https://deepinv.org/api/stubs/deepinv.models.Poisson2Sparse.html.md#deepinv.models.Poisson2Sparse) can be used in conjunction with [`ConvLista`](https://deepinv.org/api/stubs/deepinv.models.ConvLista.html.md#deepinv.models.ConvLista).
