<a id="benchmarks"></a>

# Benchmarks

This section provides benchmark results for various datasets and physics models.

#### NOTE
Benchmarks are defined in the [https://github.com/deepinv/benchmarks](https://github.com/deepinv/benchmarks) repository.
To contribute a new benchmark or add your solver to an existing benchmark, please refer to this repository.

## List of benchmarks

| Benchmark                                                                                          | Dataset                                                                         | Physics                                                                                    | Noise Model                                                                                  |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| [CBSD68 gaussian denoising](https://deepinv.org/auto_benchmarks/cbsd68_gaussian_denoising.html.md#cbsd68-gaussian-denoising) | [`CBSD68`](https://deepinv.org/api/stubs/deepinv.datasets.CBSD68.html.md#deepinv.datasets.CBSD68) | [`Denoising`](https://deepinv.org/api/stubs/deepinv.physics.Denoising.html.md#deepinv.physics.Denoising)       | [`GaussianNoise`](https://deepinv.org/api/stubs/deepinv.physics.GaussianNoise.html.md#deepinv.physics.GaussianNoise) |
| [DIV2K Gaussian Deblurring](https://deepinv.org/auto_benchmarks/div2k_gaussian_deblurring.html.md#div2k-gaussian-deblurring) | [`DIV2K`](https://deepinv.org/api/stubs/deepinv.datasets.DIV2K.html.md#deepinv.datasets.DIV2K)   | [`Blur`](https://deepinv.org/api/stubs/deepinv.physics.Blur.html.md#deepinv.physics.Blur)                 | [`GaussianNoise`](https://deepinv.org/api/stubs/deepinv.physics.GaussianNoise.html.md#deepinv.physics.GaussianNoise) |
| [DIV2K Inpainting easy](https://deepinv.org/auto_benchmarks/div2k_inpainting_easy.html.md#div2k-inpainting-easy)         | [`DIV2K`](https://deepinv.org/api/stubs/deepinv.datasets.DIV2K.html.md#deepinv.datasets.DIV2K)   | [`Inpainting`](https://deepinv.org/api/stubs/deepinv.physics.Inpainting.html.md#deepinv.physics.Inpainting)     | [`ZeroNoise`](https://deepinv.org/api/stubs/deepinv.physics.ZeroNoise.html.md#deepinv.physics.ZeroNoise)         |
| [DIV2K Super Resolution 2x](https://deepinv.org/auto_benchmarks/div2k_super_resolution_2x.html.md#div2k-super-resolution-2x) | [`DIV2K`](https://deepinv.org/api/stubs/deepinv.datasets.DIV2K.html.md#deepinv.datasets.DIV2K)   | [`Downsampling`](https://deepinv.org/api/stubs/deepinv.physics.Downsampling.html.md#deepinv.physics.Downsampling) | [`ZeroNoise`](https://deepinv.org/api/stubs/deepinv.physics.ZeroNoise.html.md#deepinv.physics.ZeroNoise)         |

## Testing your method on benchmarks

To evaluate your own reconstruction methods on these benchmarks, install `deepinv_bench`:

```bash
pip install git+https://github.com/deepinv/benchmarks.git#egg=deepinv_bench
```

If you have already installed benchmarks, you can update it with:

```bash
pip install --upgrade --force-reinstall --no-deps git+https://github.com/deepinv/benchmarks.git#egg=deepinv_bench
```

and then run on python:

```python
from deepinv_bench import run_benchmark
import deepinv as dinv
my_solver = ... # replace with your reconstruction method
results = run_benchmark(my_solver, "benchmark_name")
```

where  `benchmark_name` is the name of the benchmark and `my_solver` is your reconstruction method which receives `(y, physics, **kwargs)` where

- `y` is a [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) containing the measurements,
- `physics` is the [`forward operator`](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)

and outputs a [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) containing the reconstructed image.
