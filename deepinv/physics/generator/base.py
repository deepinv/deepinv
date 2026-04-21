from __future__ import annotations
import torch
import torch.nn as nn
from hashlib import sha256
import warnings


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
        if rng is None:
            self.rng = torch.Generator(device=device)
        else:
            # Make sure that the random generator is on the same device as the physics generator
            assert rng.device == torch.device(
                device
            ), f"The random generator is not on the same device as the Physics Generator. Got random generator on {rng.device} and the Physics Generator named {self.__class__.__name__} on {device}."
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

    @property
    def device(self) -> torch.device:
        return self.rng.device

    def step(self, batch_size: int = 1, seed: int = None, **kwargs) -> dict:
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

    def rng_manual_seed(self, seed: int | str = None):
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

    def average(self, n: int = 2000, batch_size: int = 1, **kwargs) -> dict:
        """Calculate average of physics generator.
        :param int n: number of samples to average over, defaults to 2000
        :param int n: number of samples to compute in parallel, higher means faster but more costly memory-wise, defaults to 1
        :param kwargs: kwargs to pass to `step` method.
        :returns: A dictionary with the new parameters, that is ``{param_name: param_value}``.
        """
        assert n > 0, "n must be positive"
        assert batch_size >= 1, "batch_size must be positive"
        params_sum = None
        keys = None
        n_processed = 0
        while n_processed < n:
            n_batch = min(n - n_processed, batch_size)
            params = self.step(batch_size=n_batch, **kwargs)
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
                    if params_partial_sum[k] is not None:
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
    :param: bool use_batch_sampling: whether to sample a different generator for each element in the batch. This is only possible if all generators in the mixture produce parameters with the same keys and shapes. If not, a single generator will be sampled per batch. Defaults to `True`.
    :param str device: device on which the generator is located, defaults to "cpu"
    :param torch.Generator rng: a pseudorandom random number generator for the parameter generation. If ``None``, a generator will be created on the specified device with a random seed.
    :param bool verbose: whether to print warnings about the batch-compatibility of the generators, defaults to False.

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
        use_batch_sampling: bool = True,
        device: str | torch.device = "cpu",
        rng: torch.Generator = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(device=device, rng=rng)

        probs = torch.tensor(probs, device=device)
        assert torch.sum(probs) == 1, "The sum of the probabilities must be 1."

        self.register_buffer("probs", probs)
        self.register_buffer("cum_probs", torch.cumsum(probs, dim=0))

        self.generators = generators

        self.use_batch_sampling = use_batch_sampling
        if self.use_batch_sampling:
            self.use_batch_sampling = self._compatible_generators(
                generators, verbose=verbose
            )

    @staticmethod
    def _compatible_generators(
        generators: list[PhysicsGenerator], verbose: bool = False
    ) -> bool:
        r"""
        Static method to check if each generator in the mixture produces parameters with the same keys and shapes. If they do, then the mixture can sample a different generator for each element in the batch. If not, then a single generator will be sampled per batch.
        """

        generators_keys = []
        generators_params = []
        for g in generators:
            params = g.step(batch_size=1)
            generators_params.append(params)
            generators_keys.append(set(params.keys()))

        for i in range(len(generators_keys)):
            for j in range(i + 1, len(generators_keys)):
                if generators_keys[i] != generators_keys[j]:
                    if verbose:
                        warnings.warn(
                            f"Generators {i} and {j} have different keys. Got {generators_keys[i]} and {generators_keys[j]}. Generators are not batch-compatible, a single generator will be sampled per batch."
                        )
                    return False

                for key in generators_keys[i]:
                    if (
                        generators_params[i][key].shape
                        != generators_params[j][key].shape
                    ):
                        if verbose:
                            warnings.warn(
                                f"Generators {i} and {j} have different shapes for key {key}. Got {generators_params[i][key].shape} and {generators_params[j][key].shape}. Generators are not batch-compatible, a single generator will be sampled per batch."
                            )
                        return False

        if verbose:
            warnings.warn(
                "All generators have the same keys and shapes. Generators are batch-compatible, the mixture will sample, with replacement, a different generator per batch"
            )

        return True

    def step(self, batch_size: int = 1, seed: int = None, **kwargs):
        r"""
        Returns a new set of physics' parameters,
        according to the probabilities given in the constructor.

        :param int batch_size: the number of samples to generate.
        :param int seed: the seed for the random number generator.
        :returns: A dictionary with the new parameters, ie ``{param_name: param_value}``.
        """

        if self.use_batch_sampling:
            # Sample a random generator for EACH element in the batch according to self.probs
            p = torch.rand(batch_size, generator=self.rng, device=self.device)
            # Get generator index for each sample
            generator_indices = torch.searchsorted(self.cum_probs, p)

            # Group batch elements by their assigned generator
            result = {}
            for gen_idx, generator in enumerate(self.generators):
                mask = generator_indices == gen_idx
                if mask.any():
                    # Find which batch positions use this generator
                    batch_positions = torch.where(mask)[0]
                    num_samples = batch_positions.size(0)

                    # Generate parameters for just these samples
                    params = generator.step(batch_size=num_samples, seed=seed, **kwargs)

                    # Store with position information for later reordering
                    for key, value in params.items():
                        if key not in result:
                            result[key] = torch.empty(
                                batch_size,
                                *value.shape[1:],
                                dtype=value.dtype,
                                device=value.device,
                            )
                        result[key][batch_positions] = value

            return result

        else:
            p = torch.rand(
                1, generator=self.rng, device=self.device
            )  # np.random.uniform()
            idx = torch.searchsorted(self.cum_probs, p)
            return self.generators[idx].step(batch_size=batch_size, seed=seed, **kwargs)
