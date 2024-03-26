# %%
import torch
import torch.nn as nn
from typing import List
import numpy as np
import warnings

class PhysicsGenerator(nn.Module):
    r"""
    Base class for parameter generation of physics.

    :param torch.Tensor params: parameters to be fed to a physics from :meth:`deepinv.physics`, e.g., a blur filter to be used in :meth:`deepinv.physics.Blur()`.
    :param dict kwargs: default keyword arguments to be passed to :meth:`Generator` for generating new parameters.

    """

    def __init__(
        self, step= lambda **kwargs: {}, device="cpu", dtype=torch.float32, **kwargs
    ) -> None:
        super().__init__()
        #if type(shape) == int :
        #    self.shape = (1, shape, shape)
        #elif type(shape) == float:
        #    self.shape = (1, int(shape), int(shape))
        #elif type(shape) == tuple:
        #    if len(shape) == 1:
        #        self.shape = (1, shape[0], shape[0])
        #    elif len(shape) == 2:
        #        self.shape = (1, shape[0], shape[1])
        #    elif len(shape) == 3:
        #        self.shape = shape
        #    elif len(shape) == 4:
        #        self.shape = shape[1:]
        #        warnings.warn('Batch_size should be called when using the .step() method. Trimming it out.')
        #    else:
        #        raise ValueError('Wrong shape. Should (B, C, W, H), (C, W, H), (W, H), (W,) or W')
        #else:
        #    raise ValueError('Wrong shape argument')

        self.step_func = step
        self.kwargs = kwargs
        self.factory_kwargs = {"device": device, "dtype": dtype}
        # Set attributes
        for k, v in kwargs.items():
            setattr(self, k, v)

    def step(self, batch_size=1, **kwargs):
        r"""
        Updates the parameter of the physics
        """
        if not kwargs:
            self.kwargs = kwargs
        
        return self.step_func(**kwargs)

    def __add__(self, other):
        def step(**kwargs):
            x = self.step(**kwargs)
            y = other.step(**kwargs)
            d = {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}
            return d
        return PhysicsGenerator(step=step)


class GeneratorMixture(PhysicsGenerator):
    r"""
    Base class for mixing multiple generators.

    :param list[Generator] generators: the generators instantiated from :meth:`deepinv.physics.Generator`.
    :param list[float] probs: the probability of each generator to be used at each step
    """

    def __init__(self, generators: List[PhysicsGenerator], probs: List[float]) -> None:
        super().__init__(generators[0].shape)
        assert torch.sum(probs) == 1, "The sum of the probabilities must be 1."
        self.generators = generators
        self.probs = probs
        self.cum_probs = torch.cumsum(probs)

    def step(self, batch_size):
        r"""
        Updates the parameter of the physic
        """
        #self.factory_kwargs = {"device": self.params.device, "dtype": self.params.dtype}
        p = torch.rand(1).item() #np.random.uniform()
        idx = torch.searchsorted(self.cum_probs, p)
        return self.generators[idx].step(batch_size)



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
    g1 = PhysicsGenerator(P.params, l=1, n=2)
    g2 = PhysicsGenerator(P.params, l=1, n=2)
    G = GeneratorMixture([g1, g2], [0.5, 0.5])
    G.step()
    print(P.params)
    # %%
