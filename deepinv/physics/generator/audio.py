from __future__ import annotations
import numpy as np
import torch
from deepinv.physics.generator import PhysicsGenerator


class RIRGenerator(PhysicsGenerator):
    r"""
    Random room impulse response (RIR) generator for the :class:`deepinv.physics.Reverberation` operator.

    At every call to :meth:`step`, a random shoebox room is sampled (dimensions and
    reverberation time :math:`T_{60}`), together with a random position for a single
    omnidirectional microphone and a point source, both kept at a minimum distance
    from the walls. The corresponding RIR is then simulated with the image-source
    method (plus a stochastic ray-tracing tail) using
    `pyroomacoustics <https://github.com/LCAV/pyroomacoustics>`_ :footcite:t:`scheibler2018pyroomacoustics`,
    and truncated/zero-padded to `filter_length` samples.

    This follows the RIR simulation procedure used to build the datasets of
    :footcite:t:`lemercier2023diffusion` (see the
    `derevdps <https://github.com/sp-uhh/derevdps>`_ repository).

    .. important::

        This generator requires the optional dependency `pyroomacoustics
        <https://github.com/LCAV/pyroomacoustics>`_, which can be installed with
        ``pip install pyroomacoustics``.

    .. note::

        Simulating a RIR is a CPU-bound, non-differentiable operation performed with
        `numpy`/`pyroomacoustics`, and is not batched internally: ``batch_size`` RIRs
        are simulated sequentially in a Python loop. Also note that
        `pyroomacoustics`'s ray tracing relies on the global `numpy` random state,
        which is reseeded from ``self.rng`` at the beginning of every :meth:`step`
        call for reproducibility.

    :param int filter_length: length (number of samples) `K` of the generated RIRs.
        Simulated RIRs are truncated if longer, or zero-padded if shorter.
    :param int fs: sampling frequency (in Hz) used to simulate and discretize the RIR.
    :param tuple[float, float] t60_range: range (in seconds) from which the
        reverberation time :math:`T_{60}` of the room is sampled uniformly.
    :param tuple[float, float] room_width_range: range (in meters) from which the
        room's `x` and `y` dimensions are sampled uniformly.
    :param tuple[float, float] room_height_range: range (in meters) from which the
        room's `z` dimension is sampled uniformly.
    :param float min_distance_to_wall: minimum distance (in meters) enforced between
        the microphone/source and any wall of the room.
    :param int max_rir_order: maximum image-source reflection order used by
        :class:`pyroomacoustics.room.ShoeBox` (the order suggested by Sabine's
        formula is capped by this value, for speed).
    :param torch.Generator rng: pseudo random number generator used to reseed
        `numpy`'s global random state at the beginning of every :meth:`step` call, for
        reproducibility. If ``None``, the current `numpy` random state is used as is.
    :param str device: the device to create the tensors on. Defaults to ``"cpu"``.
    :param type dtype: the data type of the generated tensors. Defaults to ``torch.float32``.

    |sep|

    :Examples:

    >>> from deepinv.physics.generator import RIRGenerator
    >>> from deepinv.physics import Reverberation
    >>> generator = RIRGenerator(filter_length=2000, fs=16000)  # doctest: +SKIP
    >>> rir = generator.step(batch_size=2)["filter"]  # doctest: +SKIP
    >>> print(rir.shape)  # doctest: +SKIP
    torch.Size([2, 1, 2000])
    >>> physics = Reverberation(**generator.step(batch_size=2))  # doctest: +SKIP

    """

    def __init__(
        self,
        filter_length: int = 4000,
        fs: int = 16000,
        t60_range: tuple[float, float] = (0.4, 1.0),
        room_width_range: tuple[float, float] = (5.0, 15.0),
        room_height_range: tuple[float, float] = (2.0, 6.0),
        min_distance_to_wall: float = 1.0,
        max_rir_order: int = 3,
        rng: torch.Generator = None,
        device: str = "cpu",
        dtype: type = torch.float32,
    ) -> None:
        kwargs = {
            "filter_length": filter_length,
            "fs": fs,
            "t60_range": t60_range,
            "room_width_range": room_width_range,
            "room_height_range": room_height_range,
            "min_distance_to_wall": min_distance_to_wall,
            "max_rir_order": max_rir_order,
        }
        super().__init__(device=device, dtype=dtype, rng=rng, **kwargs)

    def _sample_point(self, room_dim: np.ndarray) -> np.ndarray:
        return np.array(
            [
                np.random.uniform(
                    self.min_distance_to_wall, room_dim[i] - self.min_distance_to_wall
                )
                for i in range(3)
            ]
        )

    def _simulate_one_rir(self) -> np.ndarray:
        try:
            import pyroomacoustics as pra
        except ImportError:
            raise ImportError(
                "pyroomacoustics is required to use RIRGenerator. "
                "It can be installed with `pip install pyroomacoustics`."
            )

        room_dim = np.array(
            [
                np.random.uniform(*self.room_width_range),
                np.random.uniform(*self.room_width_range),
                np.random.uniform(*self.room_height_range),
            ]
        )
        t60 = np.random.uniform(*self.t60_range)

        mic_position = self._sample_point(room_dim)
        source_position = self._sample_point(room_dim)

        e_absorption, max_order = pra.inverse_sabine(t60, room_dim)
        room = pra.ShoeBox(
            room_dim,
            fs=self.fs,
            materials=pra.Material(e_absorption),
            max_order=min(self.max_rir_order, max_order),
            ray_tracing=True,
        )
        room.set_ray_tracing()
        room.add_microphone(mic_position)
        room.add_source(source_position)
        room.compute_rir()

        rir = np.asarray(room.rir[0][0], dtype=np.float32)
        rir = rir / (np.max(np.abs(rir)) + 1e-12)
        return rir

    def step(self, batch_size: int = 1, seed: int = None, **kwargs) -> dict:
        r"""
        Generates a batch of random room impulse responses (RIRs).

        :param int batch_size: number of RIRs to generate.
        :param int seed: the seed for the random number generator.
        :return: dictionary with key ``filter``: tensor of shape ``(batch_size, 1, filter_length)``.
        """
        self.rng_manual_seed(seed)
        np_seed = int(
            torch.randint(
                0, 2**31 - 1, (1,), generator=self.rng, device=self.rng.device
            )
        )
        np.random.seed(np_seed)

        rirs = torch.zeros(batch_size, 1, self.filter_length, **self.factory_kwargs)
        for i in range(batch_size):
            rir = self._simulate_one_rir()
            length = min(len(rir), self.filter_length)
            rirs[i, 0, :length] = torch.from_numpy(rir[:length]).to(
                **self.factory_kwargs
            )

        return {"filter": rirs}
