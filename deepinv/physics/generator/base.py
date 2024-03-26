# %%
import torch
import torch.nn as nn
from typing import List

class PhysicsGenerator(nn.Module):
    r"""
    Base class for parameter generation of physics.

    :param torch.device device: default 'cpu'
    :param torch.dtype dtype: default torch.float32
    """    

    def __init__(
        self, device="cpu", dtype=torch.float32
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype

    def step(self, batch_size):
        r"""
        Generates a physics parameter
        """
        return None

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
    

