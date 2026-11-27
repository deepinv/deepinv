from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from deepinv.models.base import Reconstructor
from deepinv.models.utils import load_state_dict_from_url


class CryoLithe(Reconstructor):
    r"""
    CryoLithe reconstruction network for cryo-electron tomography.

    CryoLithe :footcite:p:`kishore2025cryolithe` is a supervised network estimating a denoised, missing-wedge-corrected volume from an aligned tilt series.
    Each voxel (or wavelet coefficient) is estimated independently by an MLP from small patches of the ramp-filtered tilts around its projections, which makes it fast, memory efficient and robust to distribution shifts.

    Native implementation of the `original code <https://github.com/swing-research/CryoLithe>`_ (MIT license), loading the `released weights <https://huggingface.co/sada-group/CryoLithe>`_ of two variants:

    * `"cryolithe"`: predicts a one-level `bior3.1` wavelet decomposition of the volume (8 times fewer points);
    * `"cryolithe-pixel"`: predicts each voxel (slower, slightly better).

    Tilt series have shape `(B, C, N, H, W)` with the tilt axis along `H`, volumes `(B, C, D, H, W)`, as in :class:`deepinv.physics.CryoET`.
    Tilt series are standardized, and the output keeps their contrast (macromolecules appear dark in raw data).

    .. note::

        Pretrained models were trained on real tilt series without CTF correction, with up to 41 tilts. Any number of tilts is supported.

    .. warning::

        A 25M-parameter network is evaluated for every voxel, which is slow on CPU. Use a GPU (with `dtype=torch.float16`), bin the tilt series or use `roi`.

    :param str, None pretrained: `"download"` for the weights of `variant` (see :meth:`download_weights`), path to an original checkpoint or a state dict of this module, or `None` for random weights.
    :param str variant: `"cryolithe"` (wavelet) or `"cryolithe-pixel"`.
    :param float, None patch_scale: scale of the extracted patches, larger values increase their field of view. Default: value of the checkpoint.
    :param int batch_size: number of points evaluated simultaneously.
    :param torch.dtype dtype: inference dtype, `torch.float32` or `torch.float16` (GPU only).
    :param torch.device, str device: device of the model.

    |sep|

    :Example:

        Reconstruct a random tilt series with a randomly initialized network:

        >>> import torch
        >>> import deepinv as dinv
        >>> physics = dinv.physics.CryoET(img_size=(8, 16, 16), angles=11, ctf=False)
        >>> model = dinv.models.CryoLithe(pretrained=None)
        >>> y = physics(torch.rand(1, 1, 8, 16, 16))
        >>> model(y, physics).shape
        torch.Size([1, 1, 8, 16, 16])
    """

    def __init__(
        self,
        pretrained: str | None = "download",
        variant: str = "cryolithe",
        patch_scale: float | None = None,
        batch_size: int = 4096,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        super().__init__(device=device)
        if variant not in ("cryolithe", "cryolithe-pixel"):
            raise ValueError(
                f"Unknown variant '{variant}', choose 'cryolithe' or 'cryolithe-pixel'."
            )
        self.variant = variant
        self.wavelet = variant == "cryolithe"
        self.batch_size = int(batch_size)
        self.patch_size = 21

        self.net = _SliceSet(output_size=8 if self.wavelet else 1)
        self.register_buffer("ramp_filter", _ram_lak_filter(600))
        self.register_buffer("patch_scale", torch.ones(1))

        if pretrained == "download":
            self.load_state_dict(self.download_weights(variant))
        elif pretrained is not None:
            with torch.serialization.safe_globals(_checkpoint_safe_globals()):
                checkpoint = torch.load(
                    pretrained, map_location="cpu", weights_only=True
                )
            self.load_state_dict(_convert_checkpoint(checkpoint))

        if patch_scale is not None:
            self.patch_scale.fill_(patch_scale)

        self.to(device=device, dtype=dtype)
        self.eval()

    @staticmethod
    def download_weights(
        variant: str = "cryolithe", model_dir: str | None = None
    ) -> dict[str, Tensor]:
        r"""
        Downloads the pretrained weights (about 300 MB) from Hugging Face, loaded with `weights_only=True`.

        :param str variant: `"cryolithe"` (wavelet) or `"cryolithe-pixel"`.
        :param str, None model_dir: directory where the checkpoint is cached. Default: PyTorch hub cache.
        :return: (dict) state dict of this module.
        """
        if variant not in ("cryolithe", "cryolithe-pixel"):
            raise ValueError(
                f"Unknown variant '{variant}', choose 'cryolithe' or 'cryolithe-pixel'."
            )
        with torch.serialization.safe_globals(_checkpoint_safe_globals()):
            checkpoint = load_state_dict_from_url(
                f"https://huggingface.co/sada-group/CryoLithe/resolve/main/{variant}/checkpoint.pth?download=true",
                model_dir=model_dir,
                map_location="cpu",
                file_name=f"{variant}_checkpoint.pth",
                weights_only=True,
            )
        return _convert_checkpoint(checkpoint)

    def forward(
        self,
        y: Tensor,
        physics=None,
        angles: Tensor | None = None,
        depth: int | None = None,
        roi: tuple[slice, slice, slice] | None = None,
        **kwargs,
    ) -> Tensor:
        r"""
        Reconstructs a volume from a tilt series.

        :param torch.Tensor y: tilt series of shape `(B, C, N, H, W)`.
        :param deepinv.physics.CryoET physics: cryo-ET operator giving the tilt angles and the volume depth.
        :param torch.Tensor angles: tilt angles in degrees of shape `(N,)` or `(B, N)`, overriding those of `physics`.
        :param int depth: number of slices `D`, overriding that of `physics`. Default: `max(H, W) // 2`.
        :param tuple[slice, slice, slice] roi: contiguous region of interest `(slice_d, slice_h, slice_w)`. As voxels are estimated independently, this equals cropping the full reconstruction, at a fraction of the cost.
        :return: (:class:`torch.Tensor`) volume of shape `(B, C, D, H, W)`, or the shape of `roi`.
        """
        if y.dim() != 5:
            raise ValueError(
                f"The tilt series must have shape (B, C, N, H, W), got {tuple(y.shape)}."
            )
        if angles is None:
            if physics is None or not hasattr(physics, "angles"):
                raise ValueError("Tilt angles must be given through physics or angles.")
            angles = physics.angles
        if depth is None and physics is not None and hasattr(physics, "volume_size"):
            depth = physics.volume_size[0]
            if tuple(physics.volume_size[1:]) != tuple(y.shape[-2:]):
                raise ValueError(
                    "CryoLithe reconstructs the volume on the grid of the detector, but the volume size "
                    f"{physics.volume_size} does not match the detector size {tuple(y.shape[-2:])}."
                )
        if depth is None:
            depth = max(y.shape[-2:]) // 2

        B, C, N = y.shape[:3]
        angles = torch.as_tensor(angles, device=y.device).reshape(-1, N)
        out = []
        for b in range(B):
            a = angles[b if angles.shape[0] > 1 else 0]
            out.append(
                torch.stack(
                    [self.reconstruct(y[b, c], a, depth, roi) for c in range(C)]
                )
            )
        return torch.stack(out)

    @torch.no_grad()
    def reconstruct(
        self,
        tilts: Tensor,
        angles: Tensor,
        depth: int,
        roi: tuple[slice, slice, slice] | None = None,
    ) -> Tensor:
        r"""
        Reconstructs a single volume.

        :param torch.Tensor tilts: tilt series of shape `(N, H, W)`.
        :param torch.Tensor angles: tilt angles in degrees of shape `(N,)`.
        :param int depth: number of slices of the volume.
        :param tuple[slice, slice, slice] roi: contiguous region of interest `(slice_d, slice_h, slice_w)`.
        :return: (:class:`torch.Tensor`) volume of shape `(D, H, W)`, or the shape of `roi`.
        """
        dtype = self.ramp_filter.dtype
        H, W = tilts.shape[-2:]
        tilts = tilts.to(self.ramp_filter.device, torch.float32)
        tilts = (tilts - tilts.mean()) / tilts.std(unbiased=False)

        pad_h = pad_w = 0
        if H > W:
            pad_w = (H - W) // 2
        elif W > H:
            pad_h = (W - H) // 2
        tilts = F.pad(tilts, (pad_w, pad_w, pad_h, pad_h))
        n1, n2 = tilts.shape[-2:]
        if depth > max(n1, n2):
            warnings.warn(
                f"depth={depth} is larger than the size of the projections, using {max(n1, n2)}."
            )
            depth = max(n1, n2)

        tilts = self._ramp_filter(tilts)
        angles = torch.deg2rad(angles.to(tilts))

        # output ranges on the (padded) grid, in the (H, W, D) layout of CryoLithe
        if roi is None:
            roi = (slice(None),) * 3
        sd, sh, sw = (r.indices(n) for r, n in zip(roi, (depth, H, W), strict=True))
        if any(s[2] != 1 for s in (sd, sh, sw)):
            raise ValueError("The region of interest must use contiguous slices.")
        out_ranges = [
            (sh[0] + pad_h, sh[1] + pad_h),
            (sw[0] + pad_w, sw[1] + pad_w),
            (sd[0], sd[1]),
        ]

        shape = (n1, n2, depth)
        if self.wavelet:
            # voxel i depends on the wavelet coefficients ceil((i - 1) / 2) <= c <= floor((i + 2) / 2)
            shape = tuple((s + 3) // 2 for s in shape)
            ranges = [
                (max(0, -((1 - i0) // 2)), min(n, (i1 + 1) // 2 + 1))
                for (i0, i1), n in zip(out_ranges, shape, strict=True)
            ]
        else:
            ranges = out_ranges
        points = _generate_points(*shape, ranges=ranges, device=tilts.device)
        block = tuple(r[1] - r[0] for r in ranges)

        values = []
        for chunk in torch.split(points, self.batch_size):
            patches = _extract_patches(
                tilts, angles, chunk, self.patch_size, self.patch_scale.float()
            )
            values.append(self.net(patches.to(dtype), angles.to(dtype)).float())
        vol = torch.cat(values).T.reshape(-1, *block)  # (C_out, H, W, D)

        if self.wavelet:
            vol = _bior31_waverec3(vol[None])[0]
            # voxel j of the reconstructed block is voxel j + 2 * c0 of the full volume
            crop = [
                slice(i0 - 2 * c0, i1 - 2 * c0)
                for (i0, i1), (c0, _) in zip(out_ranges, ranges, strict=True)
            ]
            vol = vol[crop[0], crop[1], crop[2]]
        else:
            vol = vol[0]
        return vol.permute(2, 0, 1)

    def _ramp_filter(self, tilts: Tensor) -> Tensor:
        r"""Applies the learned ramp filter perpendicular to the tilt axis (last dimension)."""
        n = tilts.shape[-1]
        vec = self.ramp_filter.float()
        response = torch.zeros(2 * n, device=tilts.device)
        size = min(vec.numel(), n)
        response[: size // 2] = vec[: size // 2]
        response[(-size) // 2 :] = vec[(-size) // 2 :]
        response = torch.fft.fft(response)
        spectrum = torch.fft.fft(tilts, n=2 * n, dim=-1) * response
        return torch.fft.ifft(spectrum, n=2 * n, dim=-1).real[..., :n]


class _MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden, n_layers, skip=True):
        super().__init__()
        self.skip = skip
        self.layers = nn.ModuleList(
            nn.Linear(in_features if i == 0 else hidden, hidden, bias=False)
            for i in range(n_layers)
        )
        self.last_layer = nn.Linear(hidden, out_features, bias=False)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            out = F.relu(layer(x))
            x = out + x if (self.skip and i > 0) else out
        return self.last_layer(x)


class _RadonSet(nn.Module):
    r"""Permutation-invariant network over the projections, for one row of the patches."""

    def __init__(
        self,
        patch_size=21,
        set_input=64,
        set_output=128,
        set_hidden=128,
        set_layers=1,
        slice_hidden=512,
        slice_layers=5,
        mlp_output=32,
    ):
        super().__init__()
        self.up_layer = nn.Linear(patch_size, set_input, bias=False)
        self.setMlP = _MLP(set_input, set_output, set_hidden, set_layers)
        self.mlp = _MLP(set_output, mlp_output, slice_hidden, slice_layers)

    def forward(self, x, encoding):
        x = self.up_layer(x) * encoding
        x = self.setMlP(x).mean(dim=1)
        return self.mlp(x)


class _SliceSet(nn.Module):
    r"""CryoLithe network: one :class:`_RadonSet` per patch row, combined by an MLP."""

    def __init__(
        self,
        output_size,
        patch_size=21,
        mlp_output=32,
        combine_hidden=512,
        combine_layers=5,
        encoding_size=64,
        encoding_base=1000.0,
    ):
        super().__init__()
        self.encoding_size = encoding_size
        self.encoding_base = encoding_base
        self.slice_transformers = nn.ModuleList(
            _RadonSet(patch_size=patch_size, mlp_output=mlp_output)
            for _ in range(patch_size)
        )
        self.combination_mlp = _MLP(
            patch_size * mlp_output, output_size, combine_hidden, combine_layers
        )

    def forward(self, patches, angles):
        r"""
        :param torch.Tensor patches: patches of shape `(B, N, P, P)`, rows along the tilt axis.
        :param torch.Tensor angles: tilt angles in radians of shape `(N,)`.
        """
        x = patches.permute(0, 2, 1, 3)
        d = self.encoding_size
        freqs = self.encoding_base ** (
            2 * torch.arange(d // 2, device=angles.device) / d
        )
        arg = torch.rad2deg(angles)[:, None] / freqs.to(angles.dtype)
        encoding = torch.stack([torch.sin(arg), torch.cos(arg)], dim=-1).flatten(-2)
        x = [
            st(x[:, i], encoding[None]) for i, st in enumerate(self.slice_transformers)
        ]
        return self.combination_mlp(torch.cat(x, dim=1))


def _generate_points(n1, n2, n3, ranges=None, device=None):
    r"""
    Points `(z, w, h)` of a `(n1, n2, n3)` grid in the normalized coordinates of CryoLithe, ordered as `(H, W, D)`, restricted to the index `ranges` if given.
    """
    n = max(n1, n2, n3)
    h = torch.linspace(-1, 1, n1, device=device) * n1 / n
    w = torch.linspace(-1, 1, n2, device=device) * n2 / n
    z = torch.linspace(-1, 1, n3, device=device) * n3 / n
    if ranges is not None:
        h, w, z = (c[r0:r1] for c, (r0, r1) in zip((h, w, z), ranges, strict=True))
    hh, ww, zz = torch.meshgrid(h, w, z, indexing="ij")
    return torch.stack([zz, ww, hh], dim=-1).reshape(-1, 3)


def _ram_lak_filter(size):
    r"""
    Spatial kernel of the discrete Ram-Lak filter in FFT ordering, scaled by 2 as in :func:`skimage.transform.iradon`.
    """
    n = torch.cat([torch.arange(0, size // 2 + 1), torch.arange(size // 2 - 1, 0, -1)])
    n = n[:size].double()
    h = torch.where(n % 2 == 1, -1 / (torch.pi * n) ** 2, torch.zeros_like(n))
    h[0] = 0.25
    return (2 * h).float()


def _extract_patches(tilts, angles, points, patch_size, patch_scale):
    r"""
    Extracts the patches centred at the projections of `points` on each tilt, of shape `(M, N, P, P)`.
    """
    N, n1, n2 = tilts.shape
    M = points.shape[0]
    P = patch_size
    scale = torch.tensor([n1, n2], device=tilts.device, dtype=tilts.dtype)
    scale = scale / scale.max()

    u = (
        points[None, :, 1] * torch.cos(angles)[:, None]
        + points[None, :, 0] * torch.sin(angles)[:, None]
    )
    v = points[None, :, 2].expand(N, M)
    u, v = u / scale[1], v / scale[0]

    # patch offsets on the pixel grid (align_corners=True convention)
    offsets = torch.arange(P, device=tilts.device, dtype=tilts.dtype) - P // 2
    x_patch = offsets * 2 / (n2 - 1)
    y_patch = offsets * 2 / (n1 - 1)

    gx = u[:, :, None, None] + patch_scale * x_patch[None, None, None, :]
    gy = v[:, :, None, None] + patch_scale * y_patch[None, None, :, None]
    grid = torch.stack([gx.expand(-1, -1, P, P), gy.expand(-1, -1, P, P)], dim=-1)
    patches = F.grid_sample(
        tilts[:, None],
        grid.reshape(N, M, P * P, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return patches.reshape(N, M, P, P).permute(1, 0, 2, 3)


def _bior31_waverec3(coeffs):
    r"""
    One-level inverse 3D DWT with the `bior3.1` wavelet, matching `ptwt.waverec3`.

    :param torch.Tensor coeffs: coefficients of shape `(B, 8, n1, n2, n3)`, ordered as `a, aad, ada, add, daa, dad, dda, ddd`.
    :return: volume of shape `(B, 2 n1 - 2, 2 n2 - 2, 2 n3 - 2)`.
    """
    s = math.sqrt(2) / 8
    lo = torch.tensor([s, 3 * s, 3 * s, s], device=coeffs.device, dtype=coeffs.dtype)
    hi = torch.tensor(
        [-2 * s, -6 * s, 6 * s, 2 * s], device=coeffs.device, dtype=coeffs.dtype
    )
    filters = [
        a[:, None, None] * b[None, :, None] * c[None, None, :]
        for a in (lo, hi)
        for b in (lo, hi)
        for c in (lo, hi)
    ]
    vol = F.conv_transpose3d(coeffs, torch.stack(filters)[:, None], stride=2)[:, 0]
    return vol[:, 2:-2, 2:-2, 2:-2]


class _VectorModelReal(nn.Module):
    r"""Placeholder for the learned ramp filter pickled in the original CryoLithe checkpoints."""


def _checkpoint_safe_globals():
    r"""Globals allowed when loading original CryoLithe checkpoints with `weights_only=True`."""
    import importlib
    import numpy as np

    try:
        multiarray = importlib.import_module("numpy._core.multiarray")
    except ImportError:  # numpy < 2
        multiarray = importlib.import_module("numpy.core.multiarray")
    safe = [
        _VectorModelReal,
        (_VectorModelReal, "filter_models.vector.VectorModel_real"),
        (_VectorModelReal, "CryoLithe.filter_models.vector.VectorModel_real"),
        (multiarray.scalar, "numpy.core.multiarray.scalar"),
        (multiarray.scalar, "numpy._core.multiarray.scalar"),
        np.dtype,
    ]
    dtypes = getattr(np, "dtypes", None)
    if dtypes is not None:
        safe += [getattr(dtypes, n) for n in dir(dtypes) if n.endswith("DType")]
    return safe


def _convert_checkpoint(checkpoint: dict) -> dict[str, Tensor]:
    r"""Converts an original CryoLithe checkpoint to a state dict of :class:`CryoLithe`."""
    if "model_state_dict" not in checkpoint:
        return checkpoint  # already a state dict of this module
    state = {f"net.{k}": v for k, v in checkpoint["model_state_dict"].items()}
    ramp = checkpoint.get("ramp")
    vector = ramp.vector if hasattr(ramp, "vector") else ramp["vector"]
    state["ramp_filter"] = vector.detach().float().reshape(-1)
    patch_scale = checkpoint.get("patch_scale")
    state["patch_scale"] = (
        torch.ones(1)
        if patch_scale is None
        else patch_scale.detach().float().reshape(1)
    )
    return state
