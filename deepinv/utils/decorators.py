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


def _deprecated_func(func):
    """Decorator to mark a function or method as deprecated."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"Function '{func.__name__}' is deprecated and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return wrapper


def _deprecated_class(cls):
    """Decorator to mark a class as deprecated."""

    old_init = cls.__init__

    @functools.wraps(old_init)
    def new_init(self, *args, **kwargs):
        warnings.warn(
            f"Class '{cls.__name__}' is deprecated and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        old_init(self, *args, **kwargs)

    cls.__init__ = new_init

    return cls
