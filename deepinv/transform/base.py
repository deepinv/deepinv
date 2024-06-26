from __future__ import annotations
import torch
from typing import Tuple, Callable, Any

class Transform(torch.nn.Module):
    """Base class for image transforms.

    The base transform implements transform arithmetic and other methods to invert transforms and symmetrize functions.

    All transforms must implement ``get_params()`` to randomly generate e.g. rotation degrees or shift pixels, and ``transform()`` to deterministically transform an image given the params.

    To implement a new transform, please reimplement ``get_params()``, ``invert_params()`` (if needed) and ``transform()``.
    
    |sep|

    Examples: TODO from loss docs

        Randomly transform an image:

        >>> #TODO

        Multiply transforms to create compound transforms (direct product of groups).

        >>> #TODO

        Sum transforms to create stacks of transformed images (along the batch dimension).

        >>> #TODO

        Symmetrize a function for Reynolds Averaging:
        
        >>> #TODO


    :param int n_trans: number of transformed versions generated per input image, defaults to 1
    :param torch.Generator rng: random number generator, if None, use torch.Generator(), defaults to None
    """

    def __init__(self, *args, n_trans: int = 1, rng: torch.Generator = None, **kwargs):
        super().__init__()
        self.n_trans = n_trans
        self.rng = torch.Generator() if rng is None else rng

    def get_params(self, x: torch.Tensor) -> dict:
        """Randomly generate transform parameters.

        E.g. rotation degrees or shift amounts. Override this to implement a custom transform.

        :param torch.Tensor x: input image
        :return dict: keyword args of transform parameters e.g. {'theta': 30}
        """
        return NotImplementedError()

    def invert_params(self, **params) -> dict:
        """Invert transformation parameters.

        This may need to be overriden for custom transforms. By default, it negates each parameter.

        :param **params: transform parameters as keyword args
        :return dict: inverted parameters.
        """
        return {k: -v for k, v in params.items()}

    def transform(self, x: torch.Tensor, **params):
        """Transform image given transform parameters.
        
        Given randomly generated params (e.g. rotation degrees), deterministically transform the image x.
        
        Override this to implement a custom transform.

        :param torch.Tensor x: input image of shape (B,C,H,W)
        :param **params: params e.g. degrees or shifts provided as keyword args.
        :return: torch.Tensor: transformed image.
        """
        return NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform random transformation on image.

        :param torch.Tensor x: input image
        :return torch.Tensor: randomly transformed images
        """
        return self.transform(x, **self.get_params(x))
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Perform random inverse transformation on image (i.e. when not a group)

        :param torch.Tensor x: input image
        :return torch.Tensor: randomly transformed images
        """
        return self.transform(x, **self.invert_params(**self.get_params(x)))

    def identity(self, x: torch.Tensor) -> torch.Tensor:
        """Sanity check function that should do nothing.

        :param torch.Tensor x: input image
        :return torch.Tensor: :math:`T_g^{-1}T_g x=x`
        """
        #TODO add unit test to test T.identity(x) == x (approximately)
        return self.symmetrize(f=lambda x: x)

    def symmetrize(self, f: Callable[[torch.Tensor, Any], torch.Tensor]) -> Callable[[torch.Tensor, Any], torch.Tensor]:
        """Symmetrise a function with a transform and its inverse.

        Given a function :math:`f(\cdot):X\rightarrow X` and a transform :math:`T_g`, return the function :math:`T_g^{-1} f(T_g \cdot)`

        This is useful for e.g. Reynolds averaging a function over a group.

        :param Callable[[torch.Tensor, Any], torch.Tensor] f: function acting on tensors.
        :return Callable[[torch.Tensor, Any], torch.Tensor]: decorated function.
        """
        def symmetrized(x, *args, **kwargs):
            params = self.get_params(x)
            return self.transform(
                f(
                    self.transform(
                        x, **params
                    ), *args, **kwargs
                ), **self.invert_params(**params)
            )
        return symmetrized

    def __mul__(self, other: Transform):
        """
        Chains two transforms via the * operation.

        :param deepinv.transform.Transform other: other transform
        :return: (deepinv.transform.Transform) chained operator
        """

        class ChainTransform(Transform):
            def __init__(self, t1, t2):
                super().__init__()
                self.t1 = t1
                self.t2 = t2

            def forward(self, x: torch.Tensor):
                return self.t2(self.t1(x))

        return ChainTransform(self, other)

    def __add__(self, other: Transform):
        """
        Stacks two transforms via the + operation.

        :param deepinv.transform.Transform other: other transform
        :return: (deepinv.transform.Transform) operator which produces stacked transformed images
        """

        class StackTransform(Transform):
            def __init__(self, t1, t2):
                super().__init__()
                self.t1 = t1
                self.t2 = t2

            def forward(self, x: torch.Tensor):
                return torch.cat((self.t1(x), self.t2(x)), dim=0)

        return StackTransform(self, other)
