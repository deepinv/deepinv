import warnings
import functools


def _deprecated_alias(**aliases):
    """
    Decorator to support deprecated argument names in a class or a function.

    :param **aliases: mapping of old_name='new_name'

    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for old, new in aliases.items():
                if old in kwargs:
                    if new in kwargs:
                        raise TypeError(f"Cannot specify both '{old}' and '{new}'")
                    warnings.warn(
                        f"Argument '{old}' is deprecated and will be removed in a future version. "
                        f"Use '{new}' instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)

        return wrapper

    return decorator
