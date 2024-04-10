import torch
import torch.nn as nn
from typing import List


class PhysicsGenerator(nn.Module):
    r"""
    Base class for parameter generation of physics parameters.

    PhysicsGenerators can be summed to create larger generators (see :meth:`deepinv.physics.generator.PhysicsGenerator.__add__`),
    or mixed to create a generator that randomly selects (see :meth:`deepinv.physics.generator.GeneratorMixture`).


    |sep|

    :Examples:

        Generating blur and noise levels:

        >>> from deepinv.physics.generator import MotionBlurGenerator, SigmaGenerator
        >>> generator = MotionBlurGenerator(psf_size = (3, 3), num_channels = 1) + SigmaGenerator()
        >>> params_dict = generator.step(batch_size=1)
        >>> print(params_dict)
            {'filter': tensor([[[[0.0000, 0.1006, 0.0000],
                                [0.0000, 0.8994, 0.0000],
                                [0.0000, 0.0000, 0.0000]]]]),
             'sigma': tensor([0.1577])}


    """

    def __init__(
        self, step=lambda **kwargs: {}, device="cpu", dtype=torch.float32, **kwargs
    ) -> None:
        super().__init__()

        self.step_func = step
        self.kwargs = kwargs
        self.factory_kwargs = {"device": device, "dtype": dtype}
        # Set attributes
        for k, v in kwargs.items():
            setattr(self, k, v)

    def step(self, batch_size=1, **kwargs):
        r"""
        Generates new parameter for the forward operator

        :param int batch_size: the number of samples to generate.
        :returns: A dictionary with the new parameters, ie ``{param_name: param_value}``.
        """
        if not kwargs:
            self.kwargs = kwargs

        return self.step_func(**kwargs)

    def __add__(self, other):
        r"""
        Creates a new generator from the sum of two generators.

        :param Generator other: the other generator to be added.
        """

        def step(**kwargs):
            x = self.step(**kwargs)
            y = other.step(**kwargs)
            d = {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}
            return d

        return PhysicsGenerator(step=step)


class GeneratorMixture(PhysicsGenerator):
    r"""
    Base class for mixing multiple generators of type :class:`PhysicsGenerator`.

    :param list[PhysicsGenerator] generators: the generators instantiated from :meth:`deepinv.physics.generator.PhysicsGenerator`.
    :param list[float] probs: the probability of each generator to be used at each step

    |sep|

    :Examples:

        Mixing two types of blur

        >>> from deepinv.physics.generator import MotionBlurGenerator, DiffractionBlurGenerator, GeneratorMixture
        >>> g1 = MotionBlurGenerator(psf_size = (3, 3), num_channels = 1)
        >>> g2 = DiffractionBlurGenerator(psf_size = (3, 3), num_channels = 1)
        >>> generator = GeneratorMixture([g1, g2], [0.5, 0.5])
        >>> params_dict = generator.step(batch_size=1)

    """

    def __init__(self, generators: List[PhysicsGenerator], probs: List[float]) -> None:
        super().__init__()
        probs = torch.tensor(probs)
        assert torch.sum(probs) == 1, "The sum of the probabilities must be 1."
        self.generators = generators
        self.probs = probs
        self.cum_probs = torch.cumsum(probs, dim=0)

    def step(self, batch_size=1, **kwargs):
        r"""
        Updates the parameter of the physics

        :param int batch_size: the number of samples to generate.
        :returns: A dictionary with the new parameters, ie ``{param_name: param_value}``.
        """
        # self.factory_kwargs = {"device": self.params.device, "dtype": self.params.dtype}
        p = torch.rand(1).item()  # np.random.uniform()
        idx = torch.searchsorted(self.cum_probs, p)
        return self.generators[idx].step(batch_size, **kwargs)
