<a id="reconstructors"></a>

# Introduction

Reconstruction algorithms define an inversion function $\hat{x}=\inversef{y}{A}$
which recovers a signal $x$ from measurements $y$ given an operator $A$.

```default
x_hat = model(y, physics)
```

#### SEE ALSO
See [pretrained reconstructors](https://deepinv.org/user_guide/reconstruction/pretrained-models.html.md#pretrained-models) for ready-to-use pretrained reconstruction algorithms
that you can use to reconstruct images in one line.

## Defining your own reconstructor

All reconstruction algorithms inherit from the
[`deepinv.models.Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor) base class, take as input measurements `y`
and forward operator `physics`, and output a reconstruction `x_hat`.

To use your own reconstructor with DeepInverse, simply define the `forward` method to follow this pattern.

## Summary

Below we provide a summary of existing reconstruction methods, and a qualitative
description of their reconstruction performance and speed.

For the models that require training, you can do this using the [trainer](https://deepinv.org/user_guide/training/trainer.html.md#trainer) and [loss functions](https://deepinv.org/user_guide/training/loss.html.md#loss).

#### Reconstruction methods

| **Family of methods**                                                                                           | **Description**                                                                                                                     | **Requires Training**           | **Iterative**                                                                          | **Sampling**   |
|-----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|---------------------------------|----------------------------------------------------------------------------------------|----------------|
| [Least squares pseudoinverse](https://deepinv.org/user_guide/reconstruction/least-squares.html.md#least-squares)                        | Least squares solution without priors or training.                                                                                  | No                              | Yes, via conjugate gradient if linear                                                  | No             |
| [Deep Reconstruction Models](https://deepinv.org/user_guide/reconstruction/deep-reconstructors.html.md#deep-reconstructors)                   | Deep model architectures for reconstruction.                                                                                        | No if pretrained, yes otherwise | No                                                                                     | No             |
| [Plug-and-Play (PnP)](https://deepinv.org/user_guide/reconstruction/iterative.html.md#iterative)                                    | Leverages [pretrained denoisers](https://deepinv.org/user_guide/reconstruction/denoisers.html.md#denoisers) as priors within an optimisation algorithm. | No                              | Yes                                                                                    | No             |
| [Unfolded Networks](https://deepinv.org/user_guide/reconstruction/unfolded.html.md#unfolded)                                       | Constructs a trainable architecture by unrolling a PnP algorithm.                                                                   | Yes                             | Only [`DEQ`](https://deepinv.org/api/stubs/deepinv.unfolded.DEQ_builder.html.md#deepinv.unfolded.DEQ_builder) | No             |
| [Diffusion](https://deepinv.org/user_guide/reconstruction/sampling.html.md#diffusion)                                              | Leverages [pretrained denoisers](https://deepinv.org/user_guide/reconstruction/denoisers.html.md#denoisers) within a ODE/SDE.                           | No                              | Yes                                                                                    | Yes            |
| [Non-learned priors](https://deepinv.org/user_guide/reconstruction/iterative.html.md#iterative)                                     | Solves an optimization problem with hand-crafted priors.                                                                            | No                              | Yes                                                                                    | No             |
| [Markov Chain Monte Carlo](https://deepinv.org/user_guide/reconstruction/sampling.html.md#mcmc)                                    | Leverages [pretrained denoisers](https://deepinv.org/user_guide/reconstruction/denoisers.html.md#denoisers) as priors within an optimisation algorithm. | No                              | Yes                                                                                    | Yes            |
| [Generative Adversarial Networks and Deep Image Prior](https://deepinv.org/user_guide/reconstruction/adversarial.html.md#adversarial) | Uses a generator network to model the set of possible images.                                                                       | No                              | Yes                                                                                    | Depends        |
| [Multi-physics models](https://deepinv.org/user_guide/reconstruction/deep-reconstructors.html.md#general-reconstructors)                      | Models trained on multiple various physics and datasets for robustness to different problems.                                       | No                              | No                                                                                     | No             |

#### NOTE
Some algorithms might be better at reconstructing images with good perceptual quality (e.g. diffusion methods)
whereas other methods are better at reconstructing images with low distortion (close to the ground truth).

### Using models in the cloud

The client model [`deepinv.models.Client`](https://deepinv.org/api/stubs/deepinv.models.Client.html.md#deepinv.models.Client) allows users to perform inference on models hosted in the cloud directly from DeepInverse.

The client allows contributors to disseminate their reconstruction models, without requiring the user to have high GPU resources
or to accurately define their physics. As a contributor, all you have to do is:

> * Define your model to take tensors as input and output tensors (like [`deepinv.models.Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor))
> * Create a simple API
> * Deploy it to the cloud, and distribute the endpoint URL and API keys to anyone who might want to use it!

The user then only needs to define this client, specify the endpoint URL and API key, and pass in an image as a tensor.
The client then performs checks and passes the deserialized tensor to the server for processing.
