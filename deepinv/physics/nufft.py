import math

import torch
import numpy as np

from deepinv.physics.mri import MultiCoilMRI
from deepinv.physics.generator import PhysicsGenerator
from deepinv.utils.mixins import MRIMixin


class NonCartesianMRI(MultiCoilMRI, MRIMixin):
    """
    Non-Cartesian (multi-coil) MRI via `mri-nufft`.

    Loops over batch. Optional Voronoi density compensation.

    Data `x` is `(B,2,H,W)`, measurements `y` are `(B,2,N,S)` where `N` = coils and `S` = `num_shots * num_samples_per_shot` flattened.

    Inherits from Cartesian MultiCoilMRI so inherits simulate_birdcage_csm, update_parameters, and check_coil_maps.

    :param tuple img_size: reconstructed image size, `(H, W)` for 2D or `(Nz, H, W)` for a stacked 3D acquisition (channel dim excluded).
    :param int num_shots: number of sampling shots (`Nc`) (e.g. spokes).
    :param int num_samples_per_shot: number of samples per shot (`Ns`).
    :param int, None num_partitions: for 3D only - number of kz partitions the in-plane 2D trajectory is stacked over. `None` uses `Nz` (fully sampled); fewer undersamples kz -> aliasing along z.
    :param str trajectory: `radial` or `spiral`.
    :param str, float tilt: radial spoke tilt for mrinufft radial trajectory. Or set to `golden`/`grasp` and in_out=True for fastMRI breast data.
    :param bool in_out: if `True`, radial spokes span the full diameter (edge-to-edge through centre) instead of centre-out.
    :param torch.Tensor, int, None coil_maps: complex coil sensitivity maps of shape `(H,W)`, `(N,H,W)` or `(B,N,H,W)`. `int` `N` simulates `N` birdcage maps (requires `sigpy`). `None` = single-coil (flat map).
    :param str backend: mri-nufft backend. `finufft` for CPU, `cufinufft` for CUDA. Set to `mps` to use finufft. NOTE: mps requires torch single-thread, and float32 only (note apple cpu allows float64 but not mps).
    :param torch.Tensor, None b0_map: static off-resonance field map, `(H, W)` in 2D or `(Nz, H, W)` in 3D, in Hz. If given, the forward is wrapped with time-segmented B0 correction (`mri-nufft` `MRIFourierCorrected`) -> spiral blur / geometric distortion. `None` = ideal (no off-resonance).
    :param float readout_dwell: time between successive readout samples (s); with `b0_map`, blur scales with `readout_dwell * num_samples_per_shot * peak_Hz`.
    :param bool normalize: normalise by empirical norm
    :param str density_mode: `None` (no compensation), `compensate` (multiply density in the adjoint), or `adjointness` (split as sqrt-density in both forward and adjoint so adjointness holds).
        Set to compensate for adjoint or RSS.
        For anything else (dagger, other recons), set to None.
        NOTE we do it ourselves, instead of letting mri-nufft do it.
    :param torch.device device: physics device
    """

    def __init__(
        self,
        img_size: tuple[int, ...] = (320, 320),
        num_shots: int = 100,
        num_samples_per_shot: int = 500,
        num_partitions: int | None = None,
        trajectory: str = "radial",
        tilt: str | float = "uniform",
        in_out: bool = False,
        coil_maps: torch.Tensor | int | None = None,
        backend: str = "finufft",
        normalize: bool = False,
        density_mode: str | None = None,
        b0_map: torch.Tensor | None = None,
        readout_dwell: float = 4e-6,
        device: torch.device = "cpu",
        **kwargs,
    ):
        three_d = len(img_size) == 3  # (Nz, H, W) -> stacked non-Cartesian 3D; (H, W) -> 2D
        super().__init__(img_size=img_size, coil_maps=coil_maps, mask=None, three_d=three_d, device=device, **kwargs)

        dtype = None
        if backend == "mps":
            import os
            os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # one libomp from torch, another from finufft. Therefore allow it
            os.environ["OMP_NUM_THREADS"] = "1"  # otherwise will get lock between two libomps.
            torch.set_num_threads(1)
            backend = "finufft"
            dtype = "float32"

        try:
            import mrinufft
        except ImportError as e:
            raise ImportError("mri-nufft is required for NonCartesianMRI. Install with `pip install mrinufft[finufft]` (CPU or MPS) or `pip install mrinufft[cufinufft]` (GPU).") from e

        if trajectory == "radial":
            if isinstance(tilt, str) and tilt.lower() in ("golden", "grasp"):
                tilt = np.pi * (5**0.5 - 1) / 2 * (1 + in_out) # pre-scale by (1+in_out) as mri-nufft divides tilt by it
            self.samples = mrinufft.initialize_2D_radial(Nc=num_shots, Ns=num_samples_per_shot, tilt=tilt, in_out=in_out)
        elif trajectory == "spiral":
            self.samples = mrinufft.initialize_2D_spiral(num_shots, num_samples_per_shot, tilt="uniform", in_out=True)
        else:
            raise ValueError(f"Unsupported trajectory '{trajectory}'. Use 'radial' or 'spiral'.")

        if dtype is not None:
            self.samples = self.samples.astype(dtype) # finufft plan requires initialising with correct dtype

        self.backend = backend

        if three_d: # stack-of-stars/spirals: FFT along kz + 2D NUFFT in-plane
            from mrinufft.operators.stacked import MRIStackedNUFFT
            Nz = self.img_size[-3]
            n_part = num_partitions if num_partitions is not None else Nz
            z_index = np.unique(np.linspace(0, Nz - 1, n_part).round().astype(int))  # sampled kz planes; a subset undersamples kz -> z aliasing
            self.E = MRIStackedNUFFT(self.samples, (*self.img_size[-2:], Nz), backend=self.backend, smaps=None,
                                     z_index=z_index, n_coils=self.coil_maps.shape[1], squeeze_dims=False)
            # TODO: full 3D trajectories (kooshball, cones, seiffert) need a plain 3D NUFFT (interfaces[...]) - not stackable, and far heavier at 256^3
            # TODO: rotate the in-plane trajectory per partition (golden-angle along kz) instead of identical stacks
        else:
            _, op = mrinufft.operators.base.FourierOperatorBase.interfaces[self.backend]
            self.E = op(self.samples, self.img_size[-2:], squeeze_dims=False, n_coils=self.coil_maps.shape[1])

        if b0_map is not None:  # time-segmented B0 correction: monotonic readout time along each in-plane shot
            from mrinufft.operators.off_resonance import MRIFourierCorrected
            n_shots, n_pts = self.samples.shape[0], self.samples.shape[1]
            readout_time = torch.arange(n_pts, device=device, dtype=torch.float32).mul(readout_dwell).expand(n_shots, n_pts).contiguous()
            b0 = b0_map.to(device=device, dtype=torch.float32)
            mask_shape = self.img_size[-2:]
            if three_d:                                 # stacked operator stacks along the last axis
                b0 = b0.moveaxis(0, -1)                 # (Nz, H, W) -> (H, W, Nz)
                mask_shape = (*self.img_size[-2:], self.img_size[-3])
            self.E = MRIFourierCorrected(
                self.E,
                b0_map=b0,
                readout_time=readout_time,
                mask=torch.ones(mask_shape, dtype=torch.bool, device=device)
            )

        if density_mode is None:
            density = None
        elif density_mode in ("compensate", "adjointness"):
            density = torch.from_numpy(mrinufft.density.voronoi(self.samples).astype(self.samples.dtype)).view(1, 1, -1)  # 1,1,S
        else:
            raise ValueError("`density_mode` must be None, 'compensate', or 'adjointness'")
        self.register_buffer("density", density.to(device) if density is not None else None)
        self.density_mode = density_mode

        self.operator_norm = 1.0
        if normalize:
            self.operator_norm = self.compute_norm(torch.randn(1, 2, *(self.img_size[-3:] if three_d else self.img_size[-2:]), device=device), squared=False,)


    def A(self, x: torch.Tensor) -> torch.Tensor:  # B,2,[Nz,]H,W -> B,2,N,S
        Sx = self.coil_maps * self.to_torch_complex(x)[:, None]  # B,N,[Nz,]H,W
        if self.three_d:
            Sx = Sx.moveaxis(-3, -1)  # B,N,H,W,Nz: the stacked NUFFT stacks along the last axis
        Ax = torch.cat([self.E.op(Sx[i : i + 1]) for i in range(Sx.shape[0])], dim=0)  # B,N,S
        if self.density_mode == "adjointness":
            Ax = Ax * self.density.sqrt()
        return self.from_torch_complex(Ax).float() / self.operator_norm

    def A_adjoint(
        self,
        y: torch.Tensor,
        coil_maps: torch.Tensor = None,
        rss: bool = False, # ignores coil maps
        **kwargs,
    ) -> torch.Tensor:  # B,2,N,S -> B,2,H,W (or B,1,H,W if rss)
        self.update_parameters(coil_maps=coil_maps, **kwargs)

        y_complex = self.to_torch_complex(y)  # B,N,S

        if self.density_mode == "adjointness":
            y_complex = y_complex * self.density.sqrt()
        elif self.density_mode == "compensate":
            y_complex = y_complex * self.density

        out = torch.cat(
            [self.E.adj_op(y_complex[i : i + 1]) for i in range(y_complex.shape[0])], dim=0
        )  # B,N,H,W,Nz (3D) or B,N,H,W
        if self.three_d:
            out = out.moveaxis(-1, -3)  # B,N,Nz,H,W

        if rss:
            x = self.rss(self.from_torch_complex(out), multicoil=True)  # B,1,H,W
        else:
            x = self.from_torch_complex((self.coil_maps.conj() * out).sum(1))  # B,2,H,W

        return x.float() / self.operator_norm

    def A_dagger(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """SENSE recon, disregarding density compensation
        """
        density_mode, self.density_mode = self.density_mode, None
        try:
            return super().A_dagger(y, **kwargs)
        finally:
            self.density_mode = density_mode

    def estimate_coil_maps(
        self, y: torch.Tensor, method: str = "low_frequency", **kwargs
    ) -> torch.Tensor:
        """Estimate coil sensitivity maps from non-Cartesian kspace via `mri-nufft`.

        Unlike :meth:`deepinv.physics.MultiCoilMRI.estimate_coil_maps` which uses ACS region,
        non-Cartesian estimation reconstructs low-frequency per-coil images
        See `mri-nufft`s :func:`mrinufft.extras.get_smaps` for details.

        :param torch.Tensor y: kspace `(B,2,N,S)`.
        :param str method: `mri-nufft` smaps method.
        :return: complex coil maps `(B,N,H,W)`.
        """
        if self.three_d:
            raise NotImplementedError("TODO get_smaps assumes 2D; for 3D stacks estimate per-partition or use a 3D smaps method")
        from mrinufft.extras import get_smaps
        fn = get_smaps(method)

        return torch.from_numpy(np.stack([
            fn(self.samples, self.img_size[-2:], kspace_data=yb, backend=self.backend, **kwargs)
            for yb in self.to_torch_complex(y).numpy(force=True)
        ])).to(y.device)

    def noise(self, x, **kwargs) -> torch.Tensor:
        r"""
        Use LinearPhysics noise model, rather than MultiCoilMRI masked noise
        """
        return self.noise_model(x, **kwargs)