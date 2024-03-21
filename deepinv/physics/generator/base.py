# %%
import torch
import torch.nn as nn
from typing import List
import numpy as np


class Generator:
    r"""
    Base class for parameter generation of physics.

    :param torch.Tensor params: the parameter of a physic from :meth:`deepinv.physics`, e.g., the filter of :meth:`deepinv.physics.Blur()`.
    :param dict kwargs: default keyword arguments to be passed to :meth:`Generator` for generating new parameters.

    """

    def __init__(self, params: torch.Tensor, **kwargs) -> None:
        self.params = params
        self.kwargs = kwargs
        self.factory_kwargs = {"device": params.device, "dtype": params.dtype}
        # Set attributes
        for k, v in kwargs.items():
            setattr(self, k, v)

    def step(self, *args, **kwargs):
        r"""
        Updates the parameter of the physic
        """

        if not kwargs:
            self.kwargs = kwargs

        new_params = self.__call__(*args, **self.kwargs)
        # print(new_params.shape)
        self.params.zero_()
        self.params.add_(new_params)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        r"""
        Return new parameter
        """
        return torch.zeros_like(self.params)


class GeneratorMixture:
    r"""
    Base class for mixing multiple generators.

    :param list[Generator] generators: the generators instantiated from :meth:`deepinv.physics.Generator`.
    :param list[float] probs: the probability of each generator to be used at each step
    """

    def __init__(self, generators: List[Generator], probs: List[float]) -> None:
        assert np.sum(probs) == 1, "The sum of the probabilities must be 1."
        self.generators = generators
        self.probs = probs
        self.cum_probs = np.cumsum(probs)

    def step(self, *args, **kwargs):
        r"""
        Updates the parameter of the physic
        """
        if not kwargs:
            self.kwargs = kwargs
        p = np.random.uniform()
        idx = np.searchsorted(self.cum_probs, p)
        self.generators[idx].step(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        p = np.random.uniform()
        idx = np.searchsorted(self.cum_probs, p)
        return self.generators[idx](*args, **kwargs)


if __name__ == "__main__":

    class Physic(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            # self.params = nn.Parameter(torch.tensor([1., 2., 3.]), requires_grad=False)
            self.params = torch.tensor([1.0, 2.0, 3.0])
            self.kwargs = kwargs

        def forward(self, *args, **kwargs):
            pass

    # %%
    P = Physic()
    print(P.params)
    g1 = Generator(P.params, l=1, n=2)
    g2 = Generator(P.params, l=1, n=2)
    G = GeneratorMixture([g1, g2], [0.5, 0.5])
    G.step()
    print(P.params)
    # %%
