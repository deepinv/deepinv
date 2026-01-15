import torch
from deepinv.models import Reconstructor


def run_benchmark(
    solver: Reconstructor | torch.nn.Module,
    benchmark_name: str,
    device: torch.device | str = "cpu",
) -> dict:
    r"""
    Run the specified benchmark using the given solver on the specified device.

    Benchmark names can be found in the :ref:`benchmark documentation <benchmarks>`.

    :param deepinv.models.Reconstructor, torch.nn.Module solver: The reconstruction solver to benchmark.
        It should receive a input `(y, physics)` and return the reconstructed `x_hat`, where `physics` is
        an instance of :class:`deepinv.physics.Physics` and `y` are the measurements (a :class:`torch.Tensor`).
    :param str benchmark_name: The name of the benchmark to run.
    :param torch.device, str device: The device (CPU or GPU) to run the benchmark on.
    :returns: A dictionary containing the benchmark results.
    """
    import benchopt

    # TODO
    results = {}
    return results
