from __future__ import annotations
from collections.abc import Mapping, Sequence
import torch
import torch.nn as nn
from hashlib import sha256
import warnings
from deepinv.utils.nn import devices_equal


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
    Generators of stacked or composed physics can be built with :meth:`stack` and ``*``
    (see :class:`deepinv.physics.generator.CombinedGenerator`).

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
            if not devices_equal(rng.device, device):
                raise ValueError(
                    f"The random generator is not on the same device as the Physics Generator. Got random generator on {rng.device} and the Physics Generator named {self.__class__.__name__} on {device}."
                )
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

    def __mul__(self, other: PhysicsGenerator) -> ComposedGenerator:
        r"""
        Combines two generators to generate the parameters of composed physics, mirroring
        :meth:`deepinv.physics.Physics.__mul__`: if ``g1`` generates parameters for ``physics1`` and
        ``g2`` for ``physics2``, then ``g1 * g2`` generates parameters for ``physics1 * physics2``
        (i.e. :math:`A_1 \circ A_2`, where :math:`A_2` is applied first).

        See :class:`deepinv.physics.generator.ComposedGenerator` for more details.

        :param deepinv.physics.generator.PhysicsGenerator other: generator of the parameters of the physics applied first.
        :return: (:class:`deepinv.physics.generator.ComposedGenerator`) combined generator.
        """
        if not isinstance(other, PhysicsGenerator):
            return NotImplemented
        return ComposedGenerator((other, self))

    def average(self, n: int = 2000, batch_size: int = 1, **kwargs) -> dict:
        """Calculate average of physics generator.

        :param int n: number of samples to average over, defaults to 2000
        :param int batch_size: number of samples to compute in parallel, higher means faster but more costly memory-wise, defaults to 1
        :param kwargs: kwargs to pass to `step` method.
        :returns: A dictionary with the new parameters, that is ``{param_name: param_value}``.
        """
        if n <= 0:  # pragma: no cover
            raise ValueError(f"n must be positive, got {n}")
        if batch_size < 1:  # pragma: no cover
            raise ValueError(f"batch_size must be positive, got {batch_size}")
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
        if torch.sum(probs) != 1:  # pragma: no cover
            raise ValueError(
                f"The sum of the probabilities must be 1, got {torch.sum(probs).item()}"
            )

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


class StackedGenerator(PhysicsGenerator):
    r"""
    Stacks generators the same way as :class:`deepinv.physics.StackedPhysics` stacks physics. If ``g1`` generates parameters for ``physics1`` and
    ``g2`` for ``physics2``, then ``StackedGenerator(g1, g2)`` generates parameters for ``stack(physics1, physics2)``.

    Each generator is assigned to the physics with the same index in ``physics.physics_list``. The generated parameters
    are returned in a flat dictionary with keys ``"{index}.{name}"``, which stacked physics route to the
    physics at ``index``. The output can thus be used as any other generator output, e.g. ``physics.update(**params)``
    or ``physics(x, **params)``.

    As for physics, stacking stacked generators flattens them, so that the indices of the generated  parameters always match the ones of the combined physics, e.g. ``StackedGenerator(g1, StackedGenerator(g2, g3))`` is equivalent to ``StackedGenerator(g1, g2, g3)``.

    .. note::

        This differs from summing generators (``g1 + g2``), which merges parameters of a single physics
        (e.g. a blur filter and a noise level). Summed generators can be combined, and conversely
        a combined generator can be summed with a generator whose (non-indexed) parameters are then sent to all physics.

    If a seed is given to :meth:`step`, the generator at index ``i`` is seeded with ``seed + i``, so that
    identical generators produce different, yet reproducible, parameters.

    :param list[PhysicsGenerator | None] | Mapping[int, PhysicsGenerator] generators: one generator per physics,
        or a mapping ``{index: generator}``. No parameters are generated for physics without generator
        (``None`` entry or missing index).

    |sep|

    :Examples:

        Generating different motion blurs for a stack of two blur operators:

        >>> import torch
        >>> from deepinv.physics.generator import StackedGenerator, MotionBlurGenerator
        >>> generator = StackedGenerator([MotionBlurGenerator((3, 3)), MotionBlurGenerator((5, 5))])
        >>> params = generator.step(batch_size=2)
        >>> print(params.keys())
        dict_keys(['0.filter', '1.filter])

    """

    def __init__(
        self,
        generators: Sequence[PhysicsGenerator | None] | Mapping[int, PhysicsGenerator],
        **kwargs,
    ) -> None:

        if isinstance(generators, (list, tuple)):
            generators = dict(enumerate(generators))

        default_num_physics = max(generators.keys()) + 1

        flat_items, flat_idx = {}, 0
        for index in range(default_num_physics):
            generator = generators.get(index, None)
            if isinstance(generator, StackedGenerator):
                for sub_index, sub_generator in enumerate(generator.generators):
                    flat_items[flat_idx] = sub_generator
                    flat_idx += 1
            else:
                if generator is not None:
                    flat_items[flat_idx] = generator
                flat_idx += 1

        default_num_physics = flat_idx

        device = next(iter(flat_items.values())).device if flat_items else "cpu"
        super().__init__(device=device)

        self.generators = nn.ModuleList(
            [flat_items.get(index, None) for index in range(default_num_physics)]
        )

    def __getitem__(self, index: int) -> PhysicsGenerator | None:
        r"""
        Returns the generator of the physics at ``index``, or ``None`` if the physics has no generator.

        :param int index: index of the physics.
        """
        return self.generators[index]

    def step(self, batch_size: int = 1, seed: int = None, **kwargs) -> dict:
        r"""
        Generates a batch of parameters for each physics.

        :param int batch_size: the number of samples to generate.
        :param int, str seed: the seed for the random number generators. The generator at index ``i`` is seeded with ``seed+i``.
        :param kwargs: keyword arguments passed to the ``step`` method of all generators.
        :returns: A flat dictionary with the new parameters, that is ``{"{index}.{param_name}": param_value}``.
        """
        params = {}
        for index, generator in enumerate(self.generators):
            if generator is None:
                continue

            sub_seed = None if seed is None else seed + index
            sub_params = generator.step(batch_size=batch_size, seed=sub_seed, **kwargs)
            params.update({f"{index}.{k}": v for k, v in sub_params.items()})
        return params


class ComposedGenerator(PhysicsGenerator):
    r"""
    Composes generators the same way as :class:`deepinv.physics.ComposedPhysics` composes physics. If ``g1`` generates parameters for ``physics1`` and
    ``g2`` for ``physics2``, then ``g1 * g2``, or equivalently``ComposedGenerator(g1, g2)``, generates parameters for ``physics1 * physics2`` (i.e. :math:`A_1 \circ A_2`, where :math:`A_2` is applied first).

    Each generator is assigned to the physics with the same index in ``physics.physics_list``. The generated parameters
    are returned in a flat dictionary with keys ``"{index}.{name}"``, which composed physics route to the
    physics at ``index``. The output can thus be used as any other generator output, e.g. ``physics.update(**params)``
    or ``physics(x, **params)``.

    As for physics, composing composed generators flattens them, so that the indices of the generated  parameters always match the ones of the combined physics, e.g. ``ComposedGenerator(g1, ComposedGenerator(g2, g3))`` is equivalent to ``ComposedGenerator(g1, g2, g3)``.

    .. note::

        This differs from summing generators (``g1 + g2``), which merges parameters of a single physics
        (e.g. a blur filter and a noise level). Summed generators can be combined, and conversely
        a combined generator can be summed with a generator whose (non-indexed) parameters are then sent to all physics.

    If a seed is given to :meth:`step`, the generator at index ``i`` is seeded with ``seed + i``, so that
    identical generators produce different, yet reproducible, parameters.

    :param list[PhysicsGenerator | None] | Mapping[int, PhysicsGenerator] generators: one generator per physics,
        or a mapping ``{index: generator}``. No parameters are generated for physics without generator
        (``None`` entry or missing index).

    |sep|

    :Examples:
        Generating different motion blurs for a composition of two blur operators:

        >>> import torch
        >>> from deepinv.physics.generator import ComposedGenerator, MotionBlurGenerator
        >>> generator = ComposedGenerator([MotionBlurGenerator((3, 3)), MotionBlurGenerator((5, 5))])
        >>> params = generator.step(batch_size=2)
        >>> print(params.keys())
        dict_keys(['0.filter', '1.filter])

    """

    def __init__(
        self,
        generators: Sequence[PhysicsGenerator | None] | Mapping[int, PhysicsGenerator],
    ) -> None:
        if isinstance(generators, (list, tuple)):
            generators = dict(enumerate(generators))

        default_num_physics = max(generators.keys()) + 1

        flat_items, flat_idx = {}, 0
        for index in range(default_num_physics):
            generator = generators.get(index, None)
            if isinstance(generator, ComposedGenerator):
                for sub_index, sub_generator in enumerate(generator.generators):
                    flat_items[flat_idx] = sub_generator
                    flat_idx += 1
            else:
                if generator is not None:
                    flat_items[flat_idx] = generator
                flat_idx += 1

        default_num_physics = flat_idx

        device = next(iter(flat_items.values())).device if flat_items else "cpu"
        super().__init__(device=device)

        self.generators = nn.ModuleList(
            [flat_items.get(index, None) for index in range(default_num_physics)]
        )

    def __getitem__(self, index: int) -> PhysicsGenerator | None:
        r"""
        Returns the generator of the physics at ``index``, or ``None`` if the physics has no generator.

        :param int index: index of the physics.
        """
        return self.generators[index]

    def step(self, batch_size: int = 1, seed: int = None, **kwargs) -> dict:
        r"""
        Generates a batch of parameters for each physics.

        :param int batch_size: the number of samples to generate.
        :param int, str seed: the seed for the random number generators. The generator at index ``i`` is seeded with ``seed+i``.
        :param kwargs: keyword arguments passed to the ``step`` method of all generators.
        :returns: A flat dictionary with the new parameters, that is ``{"{index}.{param_name}": param_value}``.
        """
        params = {}
        for index, generator in enumerate(self.generators):
            if generator is None:
                continue

            sub_seed = None if seed is None else seed + index
            sub_params = generator.step(batch_size=batch_size, seed=sub_seed, **kwargs)
            params.update({f"{index}.{k}": v for k, v in sub_params.items()})
        return params
