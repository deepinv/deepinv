from dataclasses import dataclass
import torch

from .base import Transform


@dataclass
class Homography(Transform):
    r"""

    TODO merge from #173

    :param n_trans: number of transformed versions generated per input image.
    :param torch.Generator rng: random number generator, if None, use torch.Generator(), defaults to None
    """

    def __post_init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def rand(self, maxi: float, mini: float = None) -> torch.Tensor:
        if mini is None:
            mini = -maxi
        return (mini - maxi) * torch.rand(self.n_trans, generator=self.rng) + maxi
