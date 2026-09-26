<a id="div2k-super-resolution-2x"></a>

# DIV2K Super Resolution 2x

- *Dataset*: [`DIV2K`](https://deepinv.org/api/stubs/deepinv.datasets.DIV2K.html.md#deepinv.datasets.DIV2K)
- *Physics*: [`Downsampling`](https://deepinv.org/api/stubs/deepinv.physics.Downsampling.html.md#deepinv.physics.Downsampling)
- *Noise model*: [`ZeroNoise`](https://deepinv.org/api/stubs/deepinv.physics.ZeroNoise.html.md#deepinv.physics.ZeroNoise)
- *factor*: 2
- *filter*: bicubic
- *img_size*: 256

Run this benchmark with

```python
from deepinv_bench import run_benchmark
my_solver = lambda y, physics: ...  # your solver here
results = run_benchmark(my_solver, "div2k_super_resolution_2x")
```

#### WARNING
Runtimes are only indicative and may vary depending on various factors which are not controlled in the benchmark.

| solver_name                                                                                                                                               | PSNR         | LPIPS       |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|-------------|
| [SwinIR[variant=medium]](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/div2k_super_resolution_2x/solvers/swinir.py)            | 32.98 ± 3.40 | 0.06 ± 0.04 |
| [SwinIR[variant=lightweight]](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/div2k_super_resolution_2x/solvers/swinir.py)       | 32.43 ± 3.32 | 0.07 ± 0.05 |
| [DPIR[sigma=0.1]](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/div2k_super_resolution_2x/solvers/dpir.py)                     | 26.86 ± 2.28 | 0.32 ± 0.12 |
| [DPIR[sigma=0.2]](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/div2k_super_resolution_2x/solvers/dpir.py)                     | 24.44 ± 2.20 | 0.43 ± 0.13 |
| [DPS[denoiser=DRUNet,max_iter=1000]](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/div2k_super_resolution_2x/solvers/dps.py)   | 21.94 ± 2.68 | 0.54 ± 0.15 |
| [DPS[denoiser=DiffUNet,max_iter=1000]](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/div2k_super_resolution_2x/solvers/dps.py) | 20.07 ± 2.90 | 0.61 ± 0.18 |
