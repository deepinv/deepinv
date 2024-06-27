from __future__ import annotations
import torch


class Transform(torch.nn.Module):
    """Base class for image transforms.

    Multiply transforms to create compound transforms (direct product of groups).

    Sum transforms to create stacks of transformed images (along the batch dimension).

    :param int n_trans: number of transformed versions generated per input image, defaults to 1
    :param torch.Generator rng: random number generator, if None, use torch.Generator(), defaults to None
    """

    def __init__(self, *args, n_trans: int = 1, rng: torch.Generator = None, **kwargs):
        super().__init__()
        self.n_trans = n_trans
        self.rng = torch.Generator() if rng is None else rng

    def forward(self, x: torch.Tensor):
        """Transform image.

        :param torch.Tensor x: input image of shape (B,C,H,W)
        """
        return NotImplementedError()

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
