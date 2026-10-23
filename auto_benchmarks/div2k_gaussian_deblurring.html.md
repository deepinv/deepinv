<a id="div2k-gaussian-deblurring"></a>

# DIV2K Gaussian Deblurring

- *Dataset*: [`DIV2K`](https://deepinv.org/api/stubs/deepinv.datasets.DIV2K.html.md#deepinv.datasets.DIV2K)
- *Physics*: [`Blur`](https://deepinv.org/api/stubs/deepinv.physics.Blur.html.md#deepinv.physics.Blur)
- *Noise model*: [`GaussianNoise`](https://deepinv.org/api/stubs/deepinv.physics.GaussianNoise.html.md#deepinv.physics.GaussianNoise)
- *sigma*: 0.05
- *sigma_blur*: 2
- *img_size*: 256

Run this benchmark with

```python
from deepinv_bench import run_benchmark
my_solver = lambda y, physics: ...  # your solver here
results = run_benchmark(my_solver, "div2k_gaussian_deblurring")
```

#### WARNING
Runtimes are only indicative and may vary depending on various factors which are not controlled in the benchmark.

| solver_name                                                                                                                                                   | PSNR         | LPIPS       |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|-------------|
| [RAM](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/div2k_gaussian_deblurring/solvers/ram.py)                                      | 25.63 ± 2.56 | 0.27 ± 0.11 |
| [DPIR[sigma=0.1]](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/div2k_gaussian_deblurring/solvers/dpir.py)                         | 24.43 ± 2.49 | 0.42 ± 0.12 |
| [DiffPIR[denoiser=DRUNet,zeta=0.45]](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/div2k_gaussian_deblurring/solvers/diffpir.py)   | 24.16 ± 2.41 | 0.33 ± 0.11 |
| [DiffPIR[denoiser=DiffUNet,zeta=0.45]](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/div2k_gaussian_deblurring/solvers/diffpir.py) | 23.99 ± 2.39 | 0.28 ± 0.10 |
| [DPIR[sigma=0.2]](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/div2k_gaussian_deblurring/solvers/dpir.py)                         | 23.17 ± 2.32 | 0.50 ± 0.12 |
| [DPS[denoiser=DRUNet,max_iter=1000]](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/div2k_gaussian_deblurring/solvers/dps.py)       | 22.11 ± 2.25 | 0.51 ± 0.13 |
| [DPS[denoiser=DiffUNet,max_iter=1000]](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/div2k_gaussian_deblurring/solvers/dps.py)     | 20.72 ± 2.65 | 0.51 ± 0.15 |
