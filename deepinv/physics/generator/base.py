import torch
import torch.nn as nn
from typing import Union
from hashlib import sha256


def seed_from_string(seed: str) -> int:
    """Generate 64 bit seed from string.

    Taken from https://stackoverflow.com/questions/41699857/initialize-pseudo-random-generator-with-a-string

    :param str seed: string
    :return: integer seed
    """
    return int(sha256(seed.encode("utf-8")).hexdigest(), 16) % 0xFFFF_FFFF_FFFF_FFFF


class PhysicsGenerator(nn.Module):
    r"""
    Base class for parameter generation of physics parameters.

    Physics generators are used to generate the parameters :math:`\theta` of (parameter-dependent) forward operators.

    Generators can be summed to create larger generators via :func:`deepinv.physics.generator.PhysicsGenerator.__add__`,
    or mixed to create a generator that randomly selects them via :class:`deepinv.physics.generator.GeneratorMixture`.

    :param Callable step: a function that generates the parameters of the physics, e.g.,
        the filter of the :class:`deepinv.physics.Blur`. This function should return the parameters in a dictionary with
        the corresponding key and value pairs.
    :param torch.Generator rng: (optional) a pseudorandom random number generator for the parameter generation.
        If ``None``, the default Generator of PyTorch will be used.
    :param str device: cpu or cuda
    :param torch.dtype dtype: the data type of the generated parameters

    |sep|

    :Examples:

        Generating blur and noise levels:

        >>> import torch
        >>> from deepinv.physics.generator import MotionBlurGenerator, SigmaGenerator
        >>> # combine a PhysicsGenerator for blur and noise level parameters
        >>> generator = MotionBlurGenerator(psf_size = (3, 3), num_channels = 1) + SigmaGenerator()
        >>> params_dict = generator.step(batch_size=1, seed=0) # dict_keys(['filter', 'sigma'])
        >>> print(params_dict['filter'])
        tensor([[[[0.0000, 0.1006, 0.0000],
                  [0.0000, 0.8994, 0.0000],
                  [0.0000, 0.0000, 0.0000]]]])
        >>> print(params_dict['sigma'])
        tensor([0.2532])
    """

    def __init__(
        self,
        step=lambda **kwargs: {},
        rng: torch.Generator = None,
        device="cpu",
        dtype=torch.float32,
        **kwargs,
    ) -> None:
        super().__init__()

        self.step_func = step
        self.kwargs = kwargs
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.device = device
        if rng is None:
            self.rng = torch.Generator(device=device)
        else:
            # Make sure that the random generator is on the same device as the physics generator
            assert rng.device == torch.device(
                device
            ), f"The random generator is not on the same device as the Physics Generator. Got random generator on {rng.device} and the Physics Generator named {self.__class__.__name__} on {self.device}."
            self.rng = rng

        # NOTE: There is no use in moving RNG states from one device to another
        # as Generator.set_state only supports inputs living on the CPU. Yet,
        # by registering the initial random state as a buffer, it might be
        # moved to another device. This might hinder performance as the tensor
        # will need to be moved back to the CPU if it needs to be used later.
        # We could fix that by letting it be a regular class attribute instead
        # of a buffer but it would prevent it from being included in the
        # state dicts which is undesirable.
        self.register_buffer("initial_random_state", self.rng.get_state().to(device))

        # Set attributes
        for k, v in kwargs.items():
            setattr(self, k, v)

    def step(self, batch_size: int = 1, seed: int = None, **kwargs):
        r"""
        Generates a batch of parameters for the forward operator.

        :param int batch_size: the number of samples to generate.
        :param int seed: the seed for the random number generator.
        :returns: A dictionary with the new parameters, that is ``{param_name: param_value}``.
        """
        self.rng_manual_seed(seed)
        if kwargs is not None:
            self.kwargs = kwargs

        return self.step_func(batch_size, seed, **kwargs)

    def rng_manual_seed(self, seed: Union[int, str] = None):
        r"""
        Sets the seed for the random number generator.

        :param int, str seed: the seed to set for the random number generator. If string passed,
            generate seed from the hash of the string.
            If not provided, the current state of the random number generator is used.
            Note: The `torch.manual_seed` is triggered when a the random number generator is not initialized.
        """
        if seed is not None:
            if isinstance(seed, str):
                seed = seed_from_string(seed)
            elif not isinstance(seed, int):
                raise ValueError("seed must either be int or str.")

            self.rng = self.rng.manual_seed(seed)

    def reset_rng(self):
        r"""
        Reset the random number generator to its initial state.
        """
        # NOTE: Generator.set_state expects a tensor living on the CPU.
        self.rng.set_state(self.initial_random_state.cpu())

    def __add__(self, other):
        r"""
        Creates a new generator from the sum of two generators.

        :param Generator other: the other generator to be added.
        :returns: A new generator that generates a larger dictionary with parameters of the two generators.
        """

        def step(batch_size: int = 1, seed: int = None, **kwargs):
            self.rng_manual_seed(seed)
            other.rng_manual_seed(seed)
            x = self.step(batch_size, seed=seed, **kwargs)
            y = other.step(batch_size, seed=seed, **kwargs)
            d = {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}
            return d

        return PhysicsGenerator(step=step)

    def average(self, n: int = 2000, batch_size: int = 1) -> dict:
        """Calculate average of physics generator.
        :param int n: number of samples to average over, defaults to 2000
        :param int n: number of samples to compute in parallel, higher means faster but more costly memory-wise, defaults to 1
        :returns: A dictionary with the new parameters, that is ``{param_name: param_value}``.
        """
        assert n > 0, "n must be positive"
        assert batch_size >= 1, "batch_size must be positive"
        params_sum = None
        keys = None
        n_processed = 0
        while n_processed < n:
            n_batch = min(n - n_processed, batch_size)
            params = self.step(batch_size=n_batch)
            n_processed += n_batch
            params_partial_sum = {
                k: v.sum(0, keepdim=True) for (k, v) in params.items()
            }
            if params_sum is None:
                params_sum = params_partial_sum
                keys = set(params_sum.keys())
            else:
                assert keys == set(
                    params.keys()
                ), "Different calls to PhysicsGenerator.step resulted in dictionaries with different keys"
                for k in keys:
                    params_sum[k] += params_partial_sum[k]
        params_avg = {k: v / n for (k, v) in params_sum.items()}
        return params_avg


class GeneratorMixture(PhysicsGenerator):
    r"""
    Base class for mixing multiple :class:`physics generators <deepinv.physics.generator.PhysicsGenerator>`.

    The mixture randomly selects a subset of batch elements
    to be generated by each generator according to the probabilities given in the constructor.

    :param list[PhysicsGenerator] generators: the generators instantiated from :class:`deepinv.physics.generator.PhysicsGenerator`.
    :param list[float] probs: the probability of each generator to be used at each step

    |sep|

    :Examples:

        Mixing two types of blur

        >>> from deepinv.physics.generator import MotionBlurGenerator, DiffractionBlurGenerator
        >>> from deepinv.physics.generator import GeneratorMixture
        >>> _ = torch.manual_seed(0)
        >>> g1 = MotionBlurGenerator(psf_size = (3, 3), num_channels = 1)
        >>> g2 = DiffractionBlurGenerator(psf_size = (3, 3), num_channels = 1)
        >>> generator = GeneratorMixture([g1, g2], [0.5, 0.5])
        >>> params_dict = generator.step(batch_size=1)
        >>> print(params_dict.keys())
        dict_keys(['filter'])

    """

    def __init__(
        self,
        generators: list[PhysicsGenerator],
        probs: list[float],
        rng: torch.Generator = None,
    ) -> None:
        super().__init__(rng=rng)
        probs = torch.tensor(probs)
        assert torch.sum(probs) == 1, "The sum of the probabilities must be 1."
        self.generators = generators
        self.probs = probs
        self.cum_probs = torch.cumsum(probs, dim=0)

    def step(self, batch_size: int = 1, seed: int = None, **kwargs):
        r"""
        Returns a new set of physics' parameters,
        according to the probabilities given in the constructor.

        :param int batch_size: the number of samples to generate.
        :param int seed: the seed for the random number generator.
        :returns: A dictionary with the new parameters, ie ``{param_name: param_value}``.
        """
        p = torch.rand(1, generator=self.rng).item()  # np.random.uniform()
        idx = torch.searchsorted(self.cum_probs, p)
        return self.generators[idx].step(batch_size, seed, **kwargs)
