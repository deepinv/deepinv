import warnings
import functools


def warn_kwargs_use_params(func):
    r"""
    Decorator to warn users when they try to pass standard keyword arguments
    to update :class:`deepinv.physics.StackedPhysics` or :class:`deepinv.physics.ComposedPhysics` parameters, instead of using
    the `params` keyword argument.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bad = [k for k in kwargs if k != "params"]
        # if kwargs is not empty
        if bad:
            warnings.warn(
                f" The following keyword arguments were ignored: {bad}. Passing keyword arguments to update parameters of StackedPhysics or ComposedPhysics is no longer supported. Please use the ``params`` keyword argument instead, and pass a list or mapping of parameters dictionnary, e.g. ``params``={ {0: kwargs} } will update the parameters of the first physics in the stack with the provided kwargs.",
                category=UserWarning,
                stacklevel=2,
            )
        return func(*args, **kwargs)

    return wrapper


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


def _deprecated_attribute(*old_arg, **alias):
    """
    Decorator to mark specific attributes of a class as deprecated, with optional replacement.

    :param tuple[str] old_arg: name of the deprecated attribute with no replacement. If empty, alias is expected.
    :param dict[str, str] alias: mapping of old_attr='new_attr' for deprecated attributes with replacement. If empty, old_arg is expected.

    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if len(old_arg) > 0:
                warnings.warn(
                    f"Attribute '{old_arg[0]}' is deprecated with no replacement and will be removed in a future version.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            elif len(alias) > 0:
                for old, new in alias.items():
                    warnings.warn(
                        f"Attribute '{old}' is deprecated and will be removed in a future version. Please use '{new}' instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
            else:
                raise ValueError("Either old_arg or alias must be provided.")

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
