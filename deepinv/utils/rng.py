from torch.types import Device
import torch
from contextlib import contextmanager


def _get_generator(
    rng: int | torch.Generator | None, device: Device
) -> torch.Generator:
    """Returns a torch.Generator from different kind of inputs

    :param seed: a torch.Generator or a seed. If None, a random seed is generated from the global torch random generator
    :return: torch.Generator seeded with the argument or with the global torch RNG if None
    """
    if isinstance(rng, torch.Generator):
        if device is not None and rng.device != torch.device(device):
            raise ValueError(
                f"The random generator is not on the same device as the Physics Generator. "
                f"Got random generator on {rng.device} and the deepinv object on {device}."
            )
        return rng

    if rng is None:
        # Returns a seed that is dependent on the global torch.manual_seed
        seed = _get_seed(None)
    else:
        seed = rng

    return torch.Generator(device=device).manual_seed(seed)


def _get_seed(rng: torch.Generator | None) -> int:
    """Return an integer between 1 and 2**24 - 1. Useful to create a new seed from a Generator"""
    device = None if rng is None else rng.device
    seed = int(torch.randint(1 << 24 - 1, (1,), generator=rng, device=device))
    return seed


@contextmanager
def _fork_rng(
    *, torch_global: bool = False, torch_generators: list[torch.Generator] | None = None
):
    r"""
    Fork any combination of RNGs (global, generators)

    Forking is opt-in: no RNG is forked by default. The global PyTorch RNG can be forked by enabling ``torch_global``, and additional torch generators can be forked by passing them in as ``torch_generators``.

    :param bool torch_global: Whether to fork the global PyTorch RNG state (default: ``False``).
    :param list[torch.Generator], None torch_generators: List of torch.Generator objects to be forked, by default none is forked.
    :return: Context manager that forks the specified RNGs.
    """
    with torch.random.fork_rng(devices=None, enabled=torch_global):
        states_map = (
            {g: g.get_state() for g in torch_generators}
            if torch_generators is not None
            else {}
        )

        try:
            yield
        finally:
            for g, s in states_map.items():
                g.set_state(s)
