# Some utility functions
from __future__ import annotations


def _add_tuple(a: tuple, b: tuple, constant: float = 1) -> tuple:
    """Add two tuples element-wise with optional scaling of second tuple.
    It computes: output[i] = a[i] + b[i] * constant
    """
    assert len(a) == len(b), "Input tuples must have the same length"
    return tuple(a[i] + constant * b[i] for i in range(len(a)))


def _as_pair(value: int | tuple | list) -> tuple[int, int]:
    """Ensure value is a 2-tuple."""
    if isinstance(value, int):
        return (value, value)
    elif isinstance(value, (tuple, list)):
        if len(value) >= 2:
            return tuple(value[-2:])
        else:
            raise ValueError("Tuple/list must have at least 2 elements.")
    else:
        raise TypeError(f"Expected int or tuple/list, got {type(value).__name__}")
