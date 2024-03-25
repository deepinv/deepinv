# %%
import torch
import torch.nn as nn
from typing import List
import numpy as np
import warnings

class Generator(nn.Module):
    r"""
    Base class for parameter generation of physics.

    :param torch.Tensor params: parameters to be fed to a physics from :meth:`deepinv.physics`, e.g., a blur filter to be used in :meth:`deepinv.physics.Blur()`.
    :param dict kwargs: default keyword arguments to be passed to :meth:`Generator` for generating new parameters.

    """

    def __init__(self, shape: tuple, device='cpu', dtype=torch.float32, **kwargs) -> None:
        super().__init__()
        if type(shape) == int :
            self.shape = (1, shape, shape)
        elif type(shape) == float:
            self.shape = (1, int(shape), int(shape))
        elif type(shape) == tuple:
            if len(shape) == 1 :
                self.shape = (1, shape[0], shape[0])
            elif len(shape) == 2:
                self.shape = (1, shape[0], shape[1])
            elif len(shape) == 3:
                self.shape = shape
            elif len(shape) == 4:
                self.shape = shape[1:]
                warnings.warn('Batch_size should be called when using the .step() method. Trimming it out.')
            else:
                raise ValueError('Wrong shape. Should (B, C, W, H), (C, W, H), (W, H), (W,) or W')
        else:
            raise ValueError('Wrong shape argument')


        self.kwargs = kwargs
        self.factory_kwargs = {"device": device, "dtype": dtype}
        # Set attributes
        for k, v in kwargs.items():
            setattr(self, k, v)

    def step(self, batch_size, *args, **kwargs):
        r"""
        Updates the parameter of the physics
        """
        if not kwargs:
            self.kwargs = kwargs
        
        return torch.zeros((batch_size, ) + self.shape)
        

class GeneratorMixture(Generator):
    r"""
    Base class for mixing multiple generators.

    :param list[Generator] generators: the generators instantiated from :meth:`deepinv.physics.Generator`.
    :param list[float] probs: the probability of each generator to be used at each step
    """

    def __init__(self, generators: List[Generator], probs: List[float]) -> None:
        super().__init__(generators[0].shape)
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
        #self.factory_kwargs = {"device": self.params.device, "dtype": self.params.dtype}
        p = torch.rand(1).item() #np.random.uniform()
        idx = np.searchsorted(self.cum_probs, p)
        return self.generators[idx].step(*args, **kwargs)


if __name__ == "__main__":

    class Physic(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            self.params = nn.Parameter(
                torch.tensor([1.0, 2.0, 3.0]), requires_grad=False
            )
            # self.params = torch.tensor([1.0, 2.0, 3.0])
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
