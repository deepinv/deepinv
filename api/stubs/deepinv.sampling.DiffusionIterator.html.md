# DiffusionIterator

### *class* deepinv.sampling.DiffusionIterator(cur_params=None, clip=None)

Bases: [`SamplingIterator`](https://deepinv.org/api/stubs/deepinv.sampling.SamplingIterator.html.md#deepinv.sampling.SamplingIterator)

Helper class used by [`deepinv.sampling.DiffusionSampler`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSampler.html.md#deepinv.sampling.DiffusionSampler) to interface diffusion models with the
[`deepinv.sampling.BaseSampling`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSampling.html.md#deepinv.sampling.BaseSampling) framework.

#### NOTE
Users should typically interact with [`deepinv.sampling.DiffusionSampler`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSampler.html.md#deepinv.sampling.DiffusionSampler) rather than this class directly.
