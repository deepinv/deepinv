from __future__ import annotations


def _add_tuple(a: tuple, b: tuple, constant: float = 1) -> tuple:
    """Add two tuples element-wise with optional scaling of second tuple.
    It computes: output[i] = a[i] + b[i] * constant
    """
    if len(a) != len(b):
        raise ValueError("Input tuples must have the same length")
    return tuple(a[i] + constant * b[i] for i in range(len(a)))


def _as_pair(value: int | float | tuple | list) -> tuple[int, int]:
    """Ensure value is a 2-tuple."""
    if isinstance(value, (int, float)):
        return (value, value)
    elif isinstance(value, (tuple, list)):
        if len(value) >= 2:
            return tuple(value[-2:])
        else:
            raise ValueError("Tuple/list must have at least 2 elements.")
    else:
        raise TypeError(
            f"Expected int, float, or tuple/list, got {type(value).__name__}"
        )


def _as_sequence(val):
    r"""
    Ensure value is a sequence. If the input is a single int or float, it is converted to a tuple of length 1.
    """
    if isinstance(val, (int, float)):
        return (val,)
    if isinstance(val, (tuple, list)):
        return tuple(val)
    raise TypeError(f"Expected int/float or sequence for parameter, got {type(val)}.")


def _check_pairwise_leq(mins, maxs, name_mins: str, name_maxs: str):
    r"""
    Check that each component of `mins` is less than or equal to the corresponding component of `maxs`.
    Raises a ValueError with a descriptive message if any component violates the constraint.
    """
    if len(mins) != len(maxs):
        raise ValueError(
            f"{name_mins} and {name_maxs} must have the same length. Got {len(mins)} and {len(maxs)}."
        )

    for m, M in zip(mins, maxs, strict=True):
        if m > M:
            raise ValueError(
                f"Each component of {name_mins} should be less than or equal to the corresponding component of {name_maxs}. Got {name_mins} = {mins} and {name_maxs} = {maxs}."
            )
