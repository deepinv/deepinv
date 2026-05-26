import warnings
import functools
from typing import Any


def _deprecated_argument(*arg_names):
    """
    Decorator to mark specific arguments of a function or method as deprecated, with no replacement.

    :param arg_names: names of the deprecated arguments
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for old_arg in arg_names:
                if old_arg in kwargs:
                    warnings.warn(
                        f"Argument '{old_arg}' is deprecated and will be removed in a future version. ",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    kwargs.pop(old_arg)
            return func(*args, **kwargs)

        return wrapper

    return decorator


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


def _deprecated_func_replaced_by(
    replacement, *, redirect=False, since=None, remove_in=None, extra=None
):
    """
    Decorator to deprecate a function in favor of another one.


    :param str replacement : The replacement function (callable) or its dotted path/name (string) used in the warning.
    :param bool redirect: If True and `replacement` is a callable, the call will be forwarded to it after issuing
        the warning. If `replacement` is a string while redirect=True, a TypeError is raised.
        Defaults to False.
    :param str since: Version in which the deprecation started (for the warning message).
    :param str remove_in: Version in which the deprecated function will be removed (for the warning message).
    :param str extra: Extra message to append to the warning.

    Examples
    --------
    @_deprecated_func_replaced_by(new_fn, redirect=True)
    def old_fn(x, y):
        ...

    @_deprecated_func_replaced_by("pkg.module.new_fn", remove_in="1.5")
    def old_fn(x, y):
        ...
    """

    def decorator(func):
        rep_name = (
            replacement
            if isinstance(replacement, str)
            else f"{replacement.__module__}.{replacement.__qualname__}"
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            parts = [f"Function '{func.__name__}' is deprecated"]
            timing = []
            if since:
                timing.append(f"since {since}")
            if remove_in:
                timing.append(f"and will be removed in {remove_in}")
            if timing:
                parts.append(" ".join(timing) + ".")
            else:
                parts.append("and will be removed in a future version.")
            parts.append(f"Use '{rep_name}' instead.")
            if extra:
                parts.append(str(extra))
            warnings.warn(" ".join(parts), DeprecationWarning, stacklevel=2)

            if redirect:
                if callable(replacement):
                    return replacement(*args, **kwargs)
                raise TypeError("redirect=True requires a callable 'replacement'.")
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _deprecate_attribute(
    self: Any,
    *,
    attr_name: str,
    attr_underscore_name: str,
    attr_initial_value: Any,
    deprecation_message: str,
    doc: str | None = None,
) -> None:
    """Deprecate an attribute.

    It wraps the attribute so that a warning is raised any time the attribute is read, written, or deleted.

    :param self: The instance to which the attribute is added.
    :param str attr_name: The name of the attribute to be deprecated.
    :param str attr_underscore_name: The name of the internal attribute to store the value.
    :param Any attr_initial_value: The initial value of the attribute.
    :param str deprecation_message: The deprecation warning message to be shown.
    :param str, None doc: The docstring for the deprecated attribute.
    """
    setattr(self, attr_underscore_name, attr_initial_value)

    # NOTE: Properties should be class attributes and not instance attributes.
    cls = type(self)

    # Only create the property once as it is bound to the class and not the instance
    if not hasattr(cls, attr_name):

        def fget(self) -> bool:
            val = getattr(self, attr_underscore_name)
            # warn last in case retrieval fails
            warnings.warn(deprecation_message, DeprecationWarning, stacklevel=2)
            return val

        def fset(self, value: bool) -> None:
            setattr(self, attr_underscore_name, value)
            # warn last in case setting fails
            warnings.warn(deprecation_message, DeprecationWarning, stacklevel=2)

        def fdel(self) -> None:
            delattr(self, attr_underscore_name)
            # warn last in case deletion fails
            warnings.warn(deprecation_message, DeprecationWarning, stacklevel=2)

        attr_value = property(fget=fget, fset=fset, fdel=fdel, doc=doc)
        setattr(cls, attr_name, attr_value)
