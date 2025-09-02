from itertools import zip_longest

def zip_strict(*iterables):
    """
    Strict version of :func:`zip` for Python < 3.10.

    Equivalent to ``zip(*iterables, strict=True)`` in Python 3.10+,
    raising a :class:`ValueError` if the iterables have different lengths.

    :param iterables: One or more iterables to zip together.
    :type iterables: iterable
    """
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo
