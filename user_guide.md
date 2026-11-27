<a id="user-guide"></a>

# User Guide

Imaging inverse problems are described by the equation $y = \noise{\forw{x}}$ where
$x$ is an unknown signal (image, volume, etc.) we want to recover,
$y$ are the observed measurements, $A$ is a
deterministic (linear or non-linear) operator capturing the physics of the acquisition and
$N$ characterizes the noise affecting the measurements.

## Operators

The library provides a large variety of imaging operators `physics` modelling $\noise{\forw{\cdot}}$,
which can simulate the observation process:

```default
x = load_image()
y = physics(x) # simulate observation
```

 **Introduction**

Introduction to the physics module.

 **Operators**

Forward operators and noise distributions.

 **Functional**

Various utilities for forward
operators.

## Reconstruction

In order to recover an image from its measurements, the library provides many
reconstruction methods $\hat{x}=R(y, A)$, which often leverage knowledge of the acquisition `physics`.
Given a restoration model `model`, the reconstruction is therefore provided as

```default
x_hat = model(y, physics) # reconstruct signal
```

 **Introduction**

Introduction and summary of reconstruction algorithms.

 **Pseudoinverse**

Simplest reconstruction without priors or training.

 **Pretrained Models**

Ready-to-use pretrained models and denoisers.

 **Denoisers**

Classical and deep denoisers with
pretrained weights.

 **Deep Reconstruction Methods**

Deep model architectures for reconstruction.

 **Optimization**

Priors and data-fidelity functions,
and optimization algorithms.

 **Unfolded Algorithms**

Unfolded architectures.

 **Iterative Reconstruction**

Plug-and-play, RED, variational methods.

 **Sampling**

Diffusion and MCMC algorithms.

 **Adversarial Reconstruction**

Conditional, unconditional GANs and deep image prior.

 **Blind Inverse Problems**

Reconstruction with misspecified physics and/or noise level.

## Training, Testing and Utilities

All the tools from the library, from measurement operator to restoration methods,
are implemented as [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) and therefore natively support backpropagation.
Reconstruction networks `model` can be trained on datasets to improve their performance:

```default
trainer = Trainer(model, loss, optimizer, metric, train_dataset, ...)
trainer.train()
trainer.test(test_dataset)
```

 **Training**

Training and testing reconstruction
models.

 **Datasets**

Utilities to generate and load datasets
for training and testing.

 **Loss**

Supervised and self-supervised losses to train the models.

 **Metrics**

Distortion and perceptual metrics to evaluate reconstructions.

 **Transforms**

Transforms for data augmentation
and self-supervised learning.

 **Utils**

Plotting and other utilities.

## Distributed Computing

In large-scale image reconstruction problems, one GPU might not be enough to handle large images/volumes or large forward operators.
We provide multi-GPU tools specific to image reconstruction with `deepinv.distributed`:

```default
physics = deepinv.distributed.distribute(physics) # distribute physics
denoiser = deepinv.distributed.distribute(denoiser) # distribute denoiser
```

 **Distributed Reconstruction**

Distributed reconstruction for large-scale images and volumes.

 **Distributed Training**

Distributed training for large-scale images and volumes.
