from __future__ import annotations
from itertools import product
from typing import Tuple, Callable, Any
import torch
from deepinv.physics.time import TimeMixin


class TransformParam(torch.Tensor):
    """
    Helper class that stores a tensor parameter for the sole purpose of allowing overriding negation.
    """

    @staticmethod
    def __new__(cls, x, neg=None):
        x = x if isinstance(x, torch.Tensor) else torch.tensor([x])
        return torch.Tensor._make_subclass(cls, x)

    def __init__(self, x, neg: Callable = lambda x: -x):
        self._neg = neg

    def __neg__(self):
        return self._neg(torch.Tensor._make_subclass(torch.Tensor, self))

    def __getitem__(self, index):
        xi = super().__getitem__(index)
        return TransformParam(xi, neg=self._neg) if hasattr(self, "_neg") else xi


class Transform(torch.nn.Module, TimeMixin):
    r"""
    Base class for image transforms.

    The base transform implements transform arithmetic and other methods to invert transforms and symmetrize functions.

    All transforms must implement ``_get_params()`` to randomly generate e.g. rotation degrees or shift pixels,
    and ``_transform()`` to deterministically transform an image given the params.

    To implement a new transform, please reimplement ``_get_params()`` and ``_transform()`` (with a ``**kwargs`` argument).
    See respective methods for details.

    Also handle deterministic (non-random) transformations by passing in fixed parameter values.

    All transforms automatically handle video input (5D of shape ``(B,C,T,H,W)``) by flattening the time dimension.

    |sep|

    :Examples:

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

        # Accepts video input of shape (B,C,T,H,W):

        >>> transform(torch.rand((1, 1, 3, 2, 2))).shape
        torch.Size([1, 1, 3, 2, 2])

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

        Symmetrize a function by averaging over the group (also known as Reynolds averaging):

        >>> f = lambda x: x[..., [0]] * x # Function to be symmetrized
        >>> f_s = rotoshift.symmetrize(f)
        >>> f_s(x).shape
        torch.Size([1, 1, 2, 2])


    :param int n_trans: number of transformed versions generated per input image, defaults to 1
    :param torch.Generator rng: random number generator, if ``None``, use :class:`torch.Generator`, defaults to ``None``
    :param bool constant_shape: if ``True``, transformed images are assumed to be same shape as input.
        For most transforms, this will not be an issue as automatic cropping/padding should mean all outputs are same shape.
        If False, for certain transforms including :class:`deepinv.transform.Rotate`,
        ``transform`` will try to switch off automatic cropping/padding resulting in errors.
        However, ``symmetrize`` will still work but perform one-by-one (i.e. without collating over batch, which is less efficient).
    :param bool flatten_video_input: accept video (5D) input of shape ``(B,C,T,H,W)`` by flattening time dim before transforming and unflattening after all operations.
    """

    def __init__(
        self,
        *args,
        n_trans: int = 1,
        rng: torch.Generator = None,
        constant_shape: bool = True,
        flatten_video_input: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.n_trans = n_trans
        self.rng = torch.Generator() if rng is None else rng
        self.constant_shape = constant_shape
        self.flatten_video_input = flatten_video_input

    def _check_x_5D(self, x: torch.Tensor) -> bool:
        """If x 4D (i.e. 2D image), return False, if 5D (e.g. with a time dim), return True, else raise Error"""
        if len(x.shape) == 4:
            return False
        elif len(x.shape) == 5:
            return True
        else:
            raise ValueError("x must be either 4D or 5D.")

    def _get_params(self, x: torch.Tensor) -> dict:
        """
        Override this to implement a custom transform.
        See ``get_params`` for details.
        """
        return NotImplementedError()

    def get_params(self, x: torch.Tensor) -> dict:
        """Randomly generate transform parameters, one set per n_trans.

        Params are represented as tensors where the first dimension indexes batch and ``n_trans``.
        Params store e.g rotation degrees or shift amounts.

        Params may be any Tensor-like object. For inverse transforms, params are negated by default.
        To change this behaviour (e.g. calculate reciprocal for inverse), wrap the param in a ``TransformParam`` class:
        ``p = TransformParam(p, neg=lambda x: 1/x)``

        :param torch.Tensor x: input image
        :return dict: keyword args of transform parameters e.g. ``{'theta': 30}``
        """
        return (
            self._get_params(self.flatten_C(x))
            if self._check_x_5D(x) and self.flatten_video_input
            else self._get_params(x)
        )

    def invert_params(self, params: dict) -> dict:
        """Invert transformation parameters. Pass variable of type ``TransformParam`` to override negation (e.g. to take reciprocal).

        :param dict params: transform parameters as dict
        :return dict: inverted parameters.
        """
        return {k: -v for k, v in params.items()}

    def _transform(self, x: torch.Tensor, **params) -> torch.Tensor:
        """
        Override this to implement a custom transform.
        See ``transform`` for details.
        """
        return NotImplementedError()

    def transform(self, x: torch.Tensor, **params) -> torch.Tensor:
        """Transform image given transform parameters.

        Given randomly generated params (e.g. rotation degrees), deterministically transform the image x.

        :param torch.Tensor x: input image of shape (B,C,H,W)
        :param params: parameters e.g. degrees or shifts provided as keyword args.
        :return: torch.Tensor: transformed image.
        """
        transform = (
            self.wrap_flatten_C(self._transform)
            if self._check_x_5D(x) and self.flatten_video_input
            else self._transform
        )
        return transform(x, **params)

    def forward(self, x: torch.Tensor, **params) -> torch.Tensor:
        """Perform random transformation on image.

        Calls ``get_params`` to generate random params for image, then ``transform`` to deterministically transform.

        For purely deterministic transformation, pass in custom params and ``get_params`` will be ignored.

        :param torch.Tensor x: input image of shape (B,C,H,W)
        :return torch.Tensor: randomly transformed images concatenated along the first dimension
        """
        return self.transform(x, **(self.get_params(x) if not params else params))

    def inverse(self, x: torch.Tensor, batchwise=True, **params) -> torch.Tensor:
        """Perform random inverse transformation on image (i.e. when not a group).

        For purely deterministic transformation, pass in custom params and ``get_params`` will be ignored.

        :param torch.Tensor x: input image
        :param bool batchwise: if True, the output dim 0 expands to be of size ``len(x) * len(param)`` for the params of interest.
            If False, params will attempt to match each image in batch to keep constant ``len(out)=len(x)``. No effect when ``n_trans==1``
        :return torch.Tensor: randomly transformed images
        """
        inv_params = self.invert_params(self.get_params(x) if not params else params)

        if batchwise:
            return self.transform(x, **inv_params)

        assert len(x) % self.n_trans == 0, "batchwise must be True"
        B = len(x) // self.n_trans
        return torch.cat(
            [
                self.transform(
                    x[i].unsqueeze(0),
                    **{
                        k: p[[i // B]]
                        for k, p in inv_params.items()
                        if len(p) == self.n_trans
                    },
                )
                for i in range(len(x))
            ]
        )

    def identity(self, x: torch.Tensor, average: bool = False) -> torch.Tensor:
        """Sanity check function that should do nothing.

        This performs forward and inverse transform, which results in the exact original, down to interpolation and padding effects.

        Interpolation and padding effects will be visible in non-pixelwise transformations, such as arbitrary rotation, scale or projective transformation.

        :param torch.Tensor x: input image
        :param bool average: average over ``n_trans`` transformed versions to get same number as output images as input images. No effect when ``n_trans=1``.
        :return torch.Tensor: :math:`T_g^{-1}T_g x=x`
        """
        return self.symmetrize(f=lambda _x: _x, average=average)(x)

    def iterate_params(self, params):
        negs = [getattr(p, "_neg", None) for p in params.values()]
        param_lists = [p.tolist() for p in params.values()]

        return [
            {
                key: (
                    torch.tensor([comb[i]])
                    if negs[i] is None
                    else TransformParam([comb[i]], neg=negs[i])
                )
                for i, key in enumerate(params.keys())
            }
            for comb in list(product(*param_lists))
        ]

    def symmetrize(
        self,
        f: Callable[[torch.Tensor, Any], torch.Tensor],
        average: bool = False,
        collate_batch: bool = True,
    ) -> Callable[[torch.Tensor, Any], torch.Tensor]:
        r"""
        Symmetrise a function with a transform and its inverse.

        Given a function :math:`f(\cdot):X\rightarrow X` and a transform :math:`T_g`, returns the group averaged function  :math:`\sum_{i=1}^N T_{g_i}^{-1} f(T_{g_i} \cdot)` where :math:`N` is the number of random transformations.

        For example, this is useful for Reynolds averaging a function over a group. Set ``average=True`` to average over ``n_trans``.
        For example, use ``Rotate(n_trans=4, positive=True, multiples=90).symmetrize(f)`` to symmetrize f over the entire group.

        :param Callable[[torch.Tensor, Any], torch.Tensor] f: function acting on tensors.
        :param bool average: monte carlo average over all random transformations (in range ``n_trans``) when symmetrising to get same number of output images as input images. No effect when ``n_trans=1``.
        :param bool collate_batch: if ``True``, collect ``n_trans`` transformed images in batch dim and evaluate ``f`` only once.
            However, this requires ``n_trans`` extra memory. If ``False``, evaluate ``f`` for each transformation.
            Always will be ``False`` when transformed images aren't constant shape.
        :return Callable[[torch.Tensor, Any], torch.Tensor]: decorated function.
        """

        def symmetrized(x, *args, **kwargs):
            params = self.get_params(x)
            if self.constant_shape and collate_batch:
                # Collect over n_trans
                xt = self.inverse(
                    f(self.transform(x, **params), *args, **kwargs),
                    batchwise=False,
                    **params,
                )
                return xt.reshape(-1, *x.shape).mean(axis=0) if average else xt
            else:
                # Step through n_trans (or combinations) one-by-one
                out = []
                for _params in self.iterate_params(params):
                    out.append(
                        self.inverse(
                            f(self.transform(x, **_params), *args, **kwargs), **_params
                        )
                    )
                return (
                    torch.stack(out, dim=1).mean(dim=1) if average else torch.cat(out)
                )

        return lambda x, *args, **kwargs: (
            self.wrap_flatten_C(symmetrized)(x, *args, **kwargs)
            if self._check_x_5D(x) and self.flatten_video_input
            else symmetrized(x, *args, **kwargs)
        )

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
                self.constant_shape = t1.constant_shape and t2.constant_shape

            def _get_params(self, x: torch.Tensor) -> dict:
                return self.t1._get_params(x) | self.t2._get_params(x)

            def _transform(self, x: torch.Tensor, **params) -> torch.Tensor:
                return self.t2._transform(self.t1._transform(x, **params), **params)

            def inverse(
                self, x: torch.Tensor, batchwise=True, **params
            ) -> torch.Tensor:
                # If batchwise False, carefully match each set of params to each subset of n_transformed images in batch
                if batchwise:
                    return self.t1.inverse(self.t2.inverse(x, **params), **params)

                out = []
                for i in range(self.t2.n_trans):
                    _x = torch.chunk(x, self.t2.n_trans)[i]
                    __x = self.t2.inverse(
                        _x,
                        **{
                            k: p[[i]]
                            for k, p in params.items()
                            if len(p) == self.t2.n_trans
                        },
                    )
                    ___x = self.t1.inverse(__x, batchwise=False, **params)
                    out.append(___x)

                return torch.cat(out)

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

            def _get_params(self, x: torch.Tensor) -> dict:
                return self.t1._get_params(x) | self.t2._get_params(x)

            def _transform(self, x: torch.Tensor, **params) -> torch.Tensor:
                return torch.cat(
                    (self.t1._transform(x, **params), self.t2._transform(x, **params)),
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

            def _get_params(self, x: torch.Tensor) -> dict:
                return self.t1._get_params(x) | self.t2._get_params(x)

            def choose(self):
                self.recent_choice = choice = torch.randint(
                    2, (1,), generator=self.rng
                ).item()
                return choice

            def _transform(self, x: torch.Tensor, **params) -> torch.Tensor:
                choice = self.choose()
                return (
                    self.t1._transform(x, **params)
                    if choice
                    else self.t2._transform(x, **params)
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
