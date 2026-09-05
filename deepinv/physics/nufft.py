import torch
import numpy as np

from deepinv.physics.mri import MultiCoilMRI
from deepinv.utils.mixins import MRIMixin


class NonCartesianMRI(MultiCoilMRI, MRIMixin):
    """
    Non-Cartesian (multi-coil) MRI via `mri-nufft`.

    This physics wraps non-uniform FFT forward and adjoint operators provided by the `mri-nufft` `library <https://mind-inria.github.io/mri-nufft/index.html>`_, and models non-Cartesian MRI sequences such as
    radial or spiral sampling.

    The physics also supports other `mri-nufft` functionality such as density compensation.

    We assume that `x` is of shape `(B,2,H,W)` and kspace `y` are `(B,2,N,S)` where `N` = coils and `S` = num shots * num samples per shot.

    .. note::
        Only supports 2D acquisition for now. For 3D/stacked physics, please open a feature request issue on GitHub.

    .. note::
        This physics supports batching, along as the backend accepts it. See `mri-nufft backend docs <https://mind-inria.github.io/mri-nufft/backend.html>`_.

    .. tip::
        This is a thin wrapper of `mri-nufft`. Learn more about their `extensive MRI support <https://mind-inria.github.io/mri-nufft/index.html>`_, such as more advanced trajectories,
        trajectory estimation, various coil map estimation algorithms or off-resonance correction.

    :param tuple img_size: reconstructed image size `(H, W)` (no channel dim).
    :param int num_shots: number of sampling shots `Nc` (e.g. spokes)
    :param int num_samples_per_shot: number of samples per shot `Ns`
    :param str trajectory: `radial` or `spiral`, passed to `mri-nufft`.
    :param str, float tilt: radial spoke tilt for mrinufft radial trajectory. Or set to `golden`/`grasp` and `in_out=True` for fastMRI breast data.
    :param bool in_out: if `True`, radial spokes span the full diameter (edge-to-edge through centre) instead of centre-out.
    :param torch.Tensor, int, None coil_maps: complex coil sensitivity maps of shape `(H,W)`, `(N,H,W)` or `(B,N,H,W)`. `int` `N` simulates `N` birdcage maps (requires `sigpy`). `None` = single-coil (flat map).
    :param str backend: mri-nufft backend. Use `finufft` for CPU, `cufinufft` for CUDA. Set to `mps` to use finufft on Apple MPS, which avoids a torch threading clash with `libomp`.
    :param bool normalize: normalise by empirical norm
    :param bool density_compensation: optionally perform Voronoi density compensation by multiplying density in the adjoint.
        Use this only if you are doing adjoint or root-sum-square reconstruction, for which it significantly improves the result. Note this breaks adjointness of `A` and `A_adjoint`.
        **Important**: for any other reconstruction method e.g. pseudo-inverse/conjugate-gradient/least squares or optimization, set `density_compensation` to `False`.
    :param torch.device device: physics device
    """

    def __init__(
        self,
        img_size: tuple[int, ...],
        num_shots: int = 100,
        num_samples_per_shot: int = 500,
        trajectory: str = "radial",
        tilt: str | float = "uniform",
        in_out: bool = False,
        coil_maps: torch.Tensor | int | None = None,
        backend: str = "finufft",
        normalize: bool = False,
        density_compensation: bool = False,
        device: torch.device = "cpu",
        **kwargs,
    ):
        super().__init__(
            img_size=img_size,
            coil_maps=coil_maps,
            mask=None,
            three_d=False,
            device=device,
            **kwargs,
        )

        dtype = None
        if backend == "mps":
            import os

            # one libomp from torch, another from finufft. Therefore allow it
            os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
            # otherwise will get lock between two libomps.
            os.environ["OMP_NUM_THREADS"] = "1"
            torch.set_num_threads(1)
            backend = "finufft"
            dtype = "float32"

        try:
            import mrinufft
        except ImportError:
            raise ImportError(
                "mri-nufft is required for NonCartesianMRI. Install with `pip install mri-nufft[finufft]` (CPU or MPS) or `pip install mri-nufft[cufinufft]` (GPU)."
            )

        if trajectory == "radial":
            if isinstance(tilt, str) and tilt.lower() in ("golden", "grasp"):
                # pre-scale by (1+in_out) as mri-nufft divides tilt by it
                tilt = np.pi * (5**0.5 - 1) / 2 * (1 + in_out)
            self.samples = mrinufft.initialize_2D_radial(
                Nc=num_shots, Ns=num_samples_per_shot, tilt=tilt, in_out=in_out
            )
        elif trajectory == "spiral":
            self.samples = mrinufft.initialize_2D_spiral(
                num_shots, num_samples_per_shot, tilt="uniform", in_out=True
            )
        else:
            raise ValueError(
                f"Unsupported trajectory '{trajectory}'. Use 'radial' or 'spiral'."
            )

        if dtype is not None:
            # finufft plan requires initialising with correct dtype
            self.samples = self.samples.astype(dtype)

        self.backend = backend

        _, op = mrinufft.operators.base.FourierOperatorBase.interfaces[self.backend]
        self.E = op(
            self.samples,
            self.img_size[-2:],
            squeeze_dims=False,
            n_coils=self.coil_maps.shape[1],
        )

        # Compute density compensation factor of shape 1,1,S
        self.register_buffer(
            "density",
            (
                torch.from_numpy(
                    mrinufft.density.voronoi(self.samples).astype(self.samples.dtype)
                )
                .view(1, 1, -1)
                .to(device)
                if density_compensation
                else None
            ),
        )

        # Normalizing physics: default = don't normalize: divide by 1 in A and adjoint.
        # if normalize=True, divide by empirically calculated operator norm such that
        # resulting operator has norm 1.
        self.operator_norm = 1.0
        if normalize:
            self.operator_norm = self.compute_norm(
                torch.randn(1, 2, *self.img_size[-2:], device=device),
                squared=False,
            )

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """MRI-NUFFT forward operator.

        :param torch.Tensor x: image of shape B,2,H,W
        :return: :class:`torch.Tensor`, multicoil kspace of shape B,2,N,S, where N is coil dim, and S is shots * samples dim
        """
        self.update_parameters(**kwargs)
        self.E.n_batchs = x.shape[0]

        Sx = self.coil_maps * self.to_torch_complex(x)[:, None]  # B,N,H,W
        Ax = self.E.op(Sx)  # B,N,S

        return self.from_torch_complex(Ax).float() / self.operator_norm

    def A_adjoint(
        self,
        y: torch.Tensor,
        rss: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        MRI-NUFFT adjoint operator.

        :param torch.Tensor y: multi-coil kspace measurements with shape B,2,N,S where N is coil dimension.
        :param bool rss: perform root-sum-square reconstruction over coils and take magnitude.
        :returns: (:class:`torch.Tensor`) image of shape `(B,2,H,W)` if not rss else `(B,1,H,W)`
        """

        self.update_parameters(**kwargs)
        self.E.n_batchs = y.shape[0]

        y_complex = self.to_torch_complex(y)  # B,N,S

        if self.density is not None:
            y_complex = y_complex * self.density

        out = self.E.adj_op(y_complex)  # B,N,H,W

        if rss:
            x = self.rss(self.from_torch_complex(out), multicoil=True)  # B,1,H,W
        else:
            x = self.from_torch_complex((self.coil_maps.conj() * out).sum(1))  # B,2,H,W

        return x.float() / self.operator_norm

    def estimate_coil_maps(
        self, y: torch.Tensor, method: str = "low_frequency", **kwargs
    ) -> torch.Tensor:
        """Estimate coil sensitivity maps from non-Cartesian kspace via `mri-nufft`.

        Unlike :meth:`deepinv.physics.MultiCoilMRI.estimate_coil_maps` which uses ACS region,
        non-Cartesian estimation reconstructs low-frequency per-coil images
        See `mrinufft.extras.get_smaps` for details.

        .. note::
            The estimation uses `mri-nufft` and is performed on CPU.

        :param torch.Tensor y: multi-coil kspace `(B,2,N,S)`.
        :param str method: `mri-nufft` smaps method, either `low_frequency` or `espirit`.
        :return: complex coil maps `(B,N,H,W)`.
        """
        from mrinufft.extras import get_smaps

        fn = get_smaps(method)

        return torch.from_numpy(
            np.stack(
                [
                    fn(
                        self.samples,
                        self.img_size[-2:],
                        kspace_data=yb,
                        backend=self.backend,
                        **kwargs,
                    )
                    for yb in self.to_torch_complex(y).numpy(force=True)
                ]
            )
        ).to(y.device)

    def noise(self, x, **kwargs) -> torch.Tensor:
        r"""
        Bypass MultiCoilMRI Cartesian masked noise
        """
        return self.noise_model(x, **kwargs)
