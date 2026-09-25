import torch
from contextlib import contextmanager


@contextmanager
def _fork_rng(
    *, torch_global: bool = False, torch_generators: list[torch.Generator] | None = None
):
    r"""
    Fork any combination of RNGs (global, generators)

    Forking is opt-in: no RNG is forked by default. The global PyTorch RNG can be forked by enabling ``torch_global``, and additional torch generators can be forked by passing them in as ``torch_generators``.

    :param bool torch_global: Whether to fork the global PyTorch RNG state (default: ``False``).
    :param list[torch.Generator], None generators: List of torch.Generator objects to be forked, by default none is forked.
    :return: Context manager that forks the specified RNGs.
    """
    with torch.random.fork_rng(devices=None, enabled=torch_global, device_type=None):
        states_map = (
            {g: g.get_state() for g in torch_generators}
            if torch_generators is not None
            else {}
        )

        yield  # Run the code block inside the context manager

        for g, s in states_map.items():
            g.set_state(s)
