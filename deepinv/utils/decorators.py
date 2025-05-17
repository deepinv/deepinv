import warnings
import functools


def _deprecated_alias(**aliases):
    """
    Decorator to support deprecated argument names.

    Args:
        **aliases: mapping of old_name='new_name'

    Example:
        @_deprecated_alias(old_arg='new_arg')
        def __init__(self, new_arg=None):
            ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
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
            return func(self, *args, **kwargs)

        return wrapper

    return decorator
