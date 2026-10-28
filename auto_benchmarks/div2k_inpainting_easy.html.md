<a id="div2k-inpainting-easy"></a>

# DIV2K Inpainting easy

- *Dataset*: [`DIV2K`](https://deepinv.org/api/stubs/deepinv.datasets.DIV2K.html.md#deepinv.datasets.DIV2K)
- *Physics*: [`Inpainting`](https://deepinv.org/api/stubs/deepinv.physics.Inpainting.html.md#deepinv.physics.Inpainting)
- *Noise model*: [`ZeroNoise`](https://deepinv.org/api/stubs/deepinv.physics.ZeroNoise.html.md#deepinv.physics.ZeroNoise)
- *mask*: 0.3
- *img_size*: 256

Run this benchmark with

```python
from deepinv_bench import run_benchmark
my_solver = lambda y, physics: ...  # your solver here
results = run_benchmark(my_solver, "div2k_inpainting_easy")
```

#### WARNING
Runtimes are only indicative and may vary depending on various factors which are not controlled in the benchmark.

| solver_name                                                                                                                                               | PSNR         | LPIPS       |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|-------------|
| [DDRM[denoiser=DRUNet,zeta=0.95]](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/div2k_inpainting_easy/solvers/ddrm.py)         | 27.42 ± 2.49 | 0.10 ± 0.04 |
| [RAM](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/div2k_inpainting_easy/solvers/ram.py)                                      | 27.20 ± 2.32 | 0.13 ± 0.05 |
| [DDRM[denoiser=DiffUNet,zeta=0.95]](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/div2k_inpainting_easy/solvers/ddrm.py)       | 27.06 ± 2.67 | 0.09 ± 0.03 |
| [DiffPIR[denoiser=DiffUNet,zeta=0.95]](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/div2k_inpainting_easy/solvers/diffpir.py) | 25.29 ± 2.16 | 0.24 ± 0.10 |
| [DiffPIR[denoiser=DRUNet,zeta=0.95]](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/div2k_inpainting_easy/solvers/diffpir.py)   | 25.22 ± 2.19 | 0.25 ± 0.10 |
| [DPS[denoiser=DRUNet,max_iter=1000]](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/div2k_inpainting_easy/solvers/dps.py)       | 23.65 ± 2.73 | 0.41 ± 0.16 |
| [DPS[denoiser=DiffUNet,max_iter=1000]](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/div2k_inpainting_easy/solvers/dps.py)     | 23.40 ± 2.66 | 0.37 ± 0.15 |
| [DPIR[sigma=0.1]](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/div2k_inpainting_easy/solvers/dpir.py)                         | 10.20 ± 4.50 | 0.94 ± 0.23 |
| [DPIR[sigma=0.05]](https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/div2k_inpainting_easy/solvers/dpir.py)                        | 9.51 ± 4.25  | 1.01 ± 0.22 |
