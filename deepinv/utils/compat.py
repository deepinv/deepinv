import sys
import warnings


def zip_strict(*iterables, force_polyfill=False):
    """
    Strict version of :func:`zip` analogous to ``zip(..., strict=True)`` with support for Python < 3.10.

    By default, this function uses the built-in ``zip(..., strict=True)`` on Python 3.10 and later. To use the polyfill implementation even on Python 3.10+, set ``force_polyfill=True``.

    .. warning::

        The polyfill implementation may not be as reliable or performant as the built-in version available in Python 3.10 and later. It is recommended to use Python 3.10+ for production code. By default, this function will emit a warning when using the polyfill implementation to encourage upgrading to Python 3.10+, you can suppress this warning by setting ``force_polyfill=True``.

    :param iterable iterables: One or more iterables to zip together.
    :param bool force_polyfill: If ``True``, use the polyfill implementation even on Python 3.10+ and suppress the warning. This is mostly useful for testing. Defaults to ``False``.
    :raises ValueError: If the input iterables have different lengths.
    """
    if sys.version_info >= (3, 10) and not force_polyfill:
        for values in zip(*iterables, strict=True):  # novermin
            yield values
    else:
        if not force_polyfill:
            warnings.warn(
                "Using the polyfill implementation of zip(..., strict=True) for Python < 3.10. Consider upgrading to Python 3.10+ for improved reliability and performance. To suppress this warning, set force_polyfill=True.",
                stacklevel=2,
            )

        its = [iter(it) for it in iterables]
        while True:
            values = []
            first_exhausted = None
            for idx, it in enumerate(its):
                sentinel = object()
                value = next(it, sentinel)
                current_exhausted = value is sentinel
                if value is not sentinel:
                    values.append(value)

                if first_exhausted is None:
                    first_exhausted = current_exhausted

                if first_exhausted != current_exhausted:
                    # Reproduce the error message of the built-in zip(..., strict=True)
                    idx = idx + 1  # 1-based index for error messages
                    comparative = "shorter" if current_exhausted else "longer"
                    plural = "s" if idx >= 3 else ""
                    arg_range = f"1-{idx-1}" if idx >= 3 else "1"
                    raise ValueError(
                        f"zip() argument {idx} is {comparative} than argument{plural} {arg_range}"
                    )

            if values:
                yield tuple(values)
            else:
                break
