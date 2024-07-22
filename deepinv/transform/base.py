from __future__ import annotations
import torch
from typing import Tuple, Callable, Any


class Param(torch.Tensor):
    """
    Helper class that stores a tensor parameter for the sole purpose of allowing overriding negation.
    """

    @staticmethod
    def __new__(cls, x, neg=None):
        return torch.Tensor._make_subclass(cls, x)

    def __init__(self, x, neg: Callable = lambda x: -x):
        self._neg = neg

    def __neg__(self):
        return self._neg(torch.Tensor._make_subclass(torch.Tensor, self))


class Transform(torch.nn.Module):
    """Base class for image transforms.

    The base transform implements transform arithmetic and other methods to invert transforms and symmetrize functions.

    All transforms must implement ``get_params()`` to randomly generate e.g. rotation degrees or shift pixels, and ``transform()`` to deterministically transform an image given the params.

    To implement a new transform, please reimplement ``get_params()`` and ``transform()`` (with a ``**kwargs`` argument). See respective methods for details.

    Also handle deterministic (non-random) transformations by passing in fixed parameter values.

    |sep|

    Examples:

        Randomly transform an image:

        >>> import torch
        >>> from deepinv.transform import Shift, Rotate
        >>> x = torch.rand((1, 1, 2, 2)) # Define random image (B,C,H,W)
        >>> transform = Shift() # Define random shift transform
        >>> transform(x).shape
        torch.Size([1, 1, 2, 2])

        Deterministically transform an image:

        >>> y = transform(transform(x, x_shift=[1]), x_shift=[-1])
        >>> torch.all(x == y)
        tensor(True)

        Multiply transforms to create compound transforms (direct product of groups) - similar to ``torchvision.transforms.Compose``:

        >>> rotoshift = Rotate() * Shift() # Chain rotate and shift transforms
        >>> rotoshift(x).shape
        torch.Size([1, 1, 2, 2])

        Sum transforms to create stacks of transformed images (along the batch dimension).

        >>> transform = Rotate() + Shift() # Stack rotate and shift transforms
        >>> transform(x).shape
        torch.Size([2, 1, 2, 2])

        Randomly select from transforms - similar to ``torchvision.transforms.RandomApply``:

        >>> transform = Rotate() | Shift() # Randomly select rotate or shift transforms
        >>> transform(x).shape
        torch.Size([1, 1, 2, 2])

        Symmetrize a function for Reynolds Averaging:

        >>> f = lambda x: x.pow(2) # Function to be symmetrized
        >>> f_s = rotoshift.symmetrize(f)
        >>> f_s(x).shape
        torch.Size([1, 1, 2, 2])



    :param int n_trans: number of transformed versions generated per input image, defaults to 1
    :param torch.Generator rng: random number generator, if None, use torch.Generator(), defaults to None
    """

    def __init__(self, *args, n_trans: int = 1, rng: torch.Generator = None, **kwargs):
        super().__init__()
        self.n_trans = n_trans
        self.rng = torch.Generator() if rng is None else rng

    def get_params(self, x: torch.Tensor) -> dict:
        """Randomly generate transform parameters, one set per n_trans.

        Params are represented as tensors where the first dimension indexes batch and n_trans.

        E.g. rotation degrees or shift amounts. Override this to implement a custom transform.

        Params may be any Tensor-like object. For inverse transforms, params are negated by default.
        To change this behaviour (e.g. calculate reciprocal for inverse), wrap the param in a ``Param`` class: ``p = Param(p, neg=lambda x: 1/x)``

        :param torch.Tensor x: input image
        :return dict: keyword args of transform parameters e.g. {'theta': 30}
        """
        return NotImplementedError()

    def invert_params(self, params: dict) -> dict:
        """Invert transformation parameters. Pass variable of type ``Param`` to override negation (e.g. to take reciprocal).

        :param dict params: transform parameters as dict
        :return dict: inverted parameters.
        """
        return {k: -v for k, v in params.items()}

    def transform(self, x: torch.Tensor, **params) -> torch.Tensor:
        """Transform image given transform parameters.

        Given randomly generated params (e.g. rotation degrees), deterministically transform the image x.

        Override this to implement a custom transform.

        :param torch.Tensor x: input image of shape (B,C,H,W)
        :param **params: params e.g. degrees or shifts provided as keyword args.
        :return: torch.Tensor: transformed image.
        """
        return NotImplementedError()

    def forward(self, x: torch.Tensor, **params) -> torch.Tensor:
        """Perform random transformation on image.

        Calls ``get_params`` to generate random params for image, then ``transform`` to deterministically transform.

        For purely deterministic transformation, pass in custom params and ``get_params`` will be ignored.

        :param torch.Tensor x: input image of shape (B,C,H,W)
        :return torch.Tensor: randomly transformed images concatenated along the first dimension
        """
        return self.transform(x, **(self.get_params(x) if not params else params))

    def inverse(self, x: torch.Tensor, **params) -> torch.Tensor:
        """Perform random inverse transformation on image (i.e. when not a group).

        For purely deterministic transformation, pass in custom params and ``get_params`` will be ignored.

        :param torch.Tensor x: input image
        :return torch.Tensor: randomly transformed images
        """
        return self.transform(
            x, **self.invert_params(self.get_params(x) if not params else params)
        )

    def identity(self, x: torch.Tensor) -> torch.Tensor:
        """Sanity check function that should do nothing.

        This performs forward and inverse transform, which results in the exact original, down to interpolation effects.

        Interpolation effects will be visible in non-pixelwise transformations, such as arbitrary rotation, scale or projective transformation.

        :param torch.Tensor x: input image
        :return torch.Tensor: :math:`T_g^{-1}T_g x=x`
        """
        return self.symmetrize(f=lambda _x: _x)(x)

    def symmetrize(
        self, f: Callable[[torch.Tensor, Any], torch.Tensor]
    ) -> Callable[[torch.Tensor, Any], torch.Tensor]:
        """Symmetrise a function with a transform and its inverse.

        Given a function :math:`f(\cdot):X\rightarrow X` and a transform :math:`T_g`, return the function :math:`T_g^{-1} f(T_g \cdot)`

        This is useful for e.g. Reynolds averaging a function over a group.

        :param Callable[[torch.Tensor, Any], torch.Tensor] f: function acting on tensors.
        :return Callable[[torch.Tensor, Any], torch.Tensor]: decorated function.
        """

        def symmetrized(x, *args, **kwargs):
            params = self.get_params(x)
            return self.inverse(
                f(self.transform(x, **params), *args, **kwargs), **params
            )

        return symmetrized

    def __mul__(self, other: Transform):
        """
        Chains two transforms via the * operation.

        :param deepinv.transform.Transform other: other transform
        :return: (deepinv.transform.Transform) chained operator
        """

        class ChainTransform(Transform):
            def __init__(self, t1: Transform, t2: Transform):
                super().__init__()
                self.t1 = t1
                self.t2 = t2

            def get_params(self, x: torch.Tensor) -> dict:
                return self.t1.get_params(x) | self.t2.get_params(x)

            # def invert_params(self, params: dict) -> dict:
            #    return self.t1.invert_params(params) | self.t2.invert_params(params)

            def transform(self, x: torch.Tensor, **params) -> torch.Tensor:
                return self.t2.transform(self.t1.transform(x, **params), **params)

            def inverse(self, x: torch.Tensor, **params) -> torch.Tensor:
                return self.t1.inverse(self.t2.inverse(x, **params), **params)

        return ChainTransform(self, other)

    def __add__(self, other: Transform):
        """
        Stacks two transforms via the + operation.

        :param deepinv.transform.Transform other: other transform
        :return: (deepinv.transform.Transform) operator which produces stacked transformed images
        """

        class StackTransform(Transform):
            def __init__(self, t1: Transform, t2: Transform):
                super().__init__()
                self.t1 = t1
                self.t2 = t2

            def get_params(self, x: torch.Tensor) -> dict:
                return self.t1.get_params(x) | self.t2.get_params(x)

            # def invert_params(self, params: dict) -> dict:
            #    return self.t1.invert_params(params) | self.t2.invert_params(params)

            def transform(self, x: torch.Tensor, **params) -> torch.Tensor:
                return torch.cat(
                    (self.t1.transform(x, **params), self.t2.transform(x, **params)),
                    dim=0,
                )

            def inverse(self, x: torch.Tensor, **params) -> torch.Tensor:
                # x is assumed to be concatenated along first (batch) dimension of t1(x) and t2(x)
                x1, x2 = x[: len(x) // 2, ...], x[len(x) // 2 :, ...]
                return torch.cat(
                    (self.t1.inverse(x1, **params), self.t2.inverse(x2, **params)),
                    dim=0,
                )

        return StackTransform(self, other)

    def __or__(self, other: Transform):
        """
        Randomly selects from two transforms via the | operation.

        :param deepinv.transform.Transform other: other transform
        :return: (deepinv.transform.Transform) random selection operator
        """

        class EitherTransform(Transform):
            def __init__(self, t1: Transform, t2: Transform):
                super().__init__()
                self.t1 = t1
                self.t2 = t2
                self.recent_choice = None

            def get_params(self, x: torch.Tensor) -> dict:
                return self.t1.get_params(x) | self.t2.get_params(x)

            def choose(self):
                self.recent_choice = choice = torch.randint(
                    2, (1,), generator=self.rng
                ).item()
                return choice

            def transform(self, x: torch.Tensor, **params) -> torch.Tensor:
                choice = self.choose()
                return (
                    self.t1.transform(x, **params)
                    if choice
                    else self.t2.transform(x, **params)
                )

            def inverse(self, x: torch.Tensor, **params) -> torch.Tensor:
                choice = (
                    self.recent_choice
                    if self.recent_choice is not None
                    else self.choose()
                )
                return (
                    self.t1.inverse(x, **params)
                    if choice
                    else self.t2.inverse(x, **params)
                )

        return EitherTransform(self, other)
