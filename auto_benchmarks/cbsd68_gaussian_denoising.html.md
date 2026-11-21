<a id="cbsd68-gaussian-denoising"></a>

# CBSD68 gaussian denoising

- *Dataset*: [`CBSD68`](https://deepinv.org/api/stubs/deepinv.datasets.CBSD68.html.md#deepinv.datasets.CBSD68)
- *Physics*: [`Denoising`](https://deepinv.org/api/stubs/deepinv.physics.Denoising.html.md#deepinv.physics.Denoising)
- *Noise model*: [`GaussianNoise`](https://deepinv.org/api/stubs/deepinv.physics.GaussianNoise.html.md#deepinv.physics.GaussianNoise)
- *sigma*: 0.1
- *img_size*: 256

Run this benchmark with

```python
from deepinv_bench import run_benchmark
my_solver = lambda y, physics: ...  # your solver here
results = run_benchmark(my_solver, "cbsd68_gaussian_denoising")
```

#### WARNING
Runtimes are only indicative and may vary depending on various factors which are not controlled in the benchmark.

| solver_name                                                                                                                                       | PSNR         | LPIPS       |
|---------------------------------------------------------------------------------------------------------------------------------------------------|--------------|-------------|
| [Restormer](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/cbsd68_gaussian_denoising/solvers/restormer.py)              | 32.25 ± 2.31 | 0.09 ± 0.04 |
| [SwinIR](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/cbsd68_gaussian_denoising/solvers/swinir.py)                    | 32.21 ± 2.28 | 0.09 ± 0.04 |
| [DRUNet](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/cbsd68_gaussian_denoising/solvers/drunet.py)                    | 32.17 ± 2.30 | 0.09 ± 0.04 |
| [GSDRUNet](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/cbsd68_gaussian_denoising/solvers/gsdrunet.py)                | 32.06 ± 2.25 | 0.09 ± 0.03 |
| [DiffUNet](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/cbsd68_gaussian_denoising/solvers/diffunet.py)                | 32.03 ± 2.28 | 0.10 ± 0.05 |
| [RAM](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/cbsd68_gaussian_denoising/solvers/ram.py)                          | 31.96 ± 2.25 | 0.10 ± 0.04 |
| [NCSNpp](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/cbsd68_gaussian_denoising/solvers/ncsnpp.py)                    | 31.91 ± 2.21 | 0.11 ± 0.05 |
| [DScCP](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/cbsd68_gaussian_denoising/solvers/dsccp.py)                      | 30.93 ± 1.98 | 0.12 ± 0.05 |
| [SCUNet](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/cbsd68_gaussian_denoising/solvers/scunet.py)                    | 30.86 ± 2.20 | 0.14 ± 0.07 |
| [DnCNN](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/cbsd68_gaussian_denoising/solvers/dncnn.py)                      | 30.39 ± 1.61 | 0.14 ± 0.06 |
| [TGVDenoiser](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/cbsd68_gaussian_denoising/solvers/tgv_denoiser.py)         | 28.07 ± 2.05 | 0.24 ± 0.06 |
| [TVDenoiser](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/cbsd68_gaussian_denoising/solvers/tv_denoiser.py)           | 27.63 ± 2.52 | 0.26 ± 0.08 |
| [BilateralFilter](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/cbsd68_gaussian_denoising/solvers/bilateral_filter.py) | 26.68 ± 1.47 | 0.44 ± 0.08 |
| [WaveletDenoiser](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/cbsd68_gaussian_denoising/solvers/wavelet_denoiser.py) | 25.18 ± 0.89 | 0.32 ± 0.10 |
