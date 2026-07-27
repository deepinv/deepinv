from __future__ import annotations
import torch
from torch import Tensor
from .base import Denoiser
from .utils import array2tensor, tensor2array
from deepinv.physics.functional import dct
from deepinv.utils import image_to_patches


class BM3D(Denoiser):
    r"""
    BM3D denoiser.

    The BM3D denoiser was introduced by :footcite:t:`dabov2007image`.

    :param bool use_legacy: Whether to use the legacy implementation of BM3D. Default: `True`
    :param device: Device to run the fast implementation of BM3D on. Default: `"cpu"`
    :param dict kwargs: additional keyword arguments for the fast implementation of BM3D. See the note below for details.

    .. note::
        Unlike other denoisers from the library, this denoiser is applied sequentially to each noisy image in the batch
        (no parallelization). Furthermore, it does not support backpropagation.

    .. note::
        Additional keyword arguments are supported for the fast implementation of BM3D (when ``use_legacy=False``), which include:

        - ``patch_size``: size of each image patch. Default: 8
        - ``search_radius``: search window radius for block matching. Default: 19
        - ``search_step``: step size for block matching. Default: 1
        - ``ref_stride``: stride for selecting reference patches. Default: 3
        - ``chunk_size``: number of groups to process in parallel. Default: 2048
        - ``ht_group_size``: group size for stage 1 (hard-thresholding). Default: 16
        - ``wiener_group_size``: group size for stage 2 (Wiener filtering). Default: 32
        - ``spatial_ht_transform``: spatial transform for stage 1. Default: `"bior1.5"`
        - ``spatial_wiener_transform``: spatial transform for stage 2. Default: `"dct"`
        - ``group_ht_transform``: group transform for stage 1. Default: `"haar"`
        - ``group_wiener_transform``: group transform for stage 2. Default: `"haar"`
        - ``hard_threshold``: hard-thresholding parameter for stage 1. Default: 3.0
        - ``wiener_mu2``: Wiener filtering parameter for stage 2. Default: 0.4

    .. warning::
        When ``use_legacy=True``, the denoiser calls the BM3D denoiser from the `BM3D python package <https://pypi.org/project/bm3d/>`_.
        It can be installed with ``pip install bm3d``.
        This implementation always runs on the CPU regardless of the device of the input tensor.

        When ``use_legacy=False``, the denoiser calls a custom re-implementation of BM3D.
        It requires ``ptwt``, which can be installed with ``pip install ptwt``.
        It runs on the device specified by the ``device`` parameter, and is significantly faster than the legacy implementation, especially when the input tensor is on the GPU.
        However, it may produce slightly different results than the legacy implementation.
    """

    def __init__(
        self, use_legacy: bool = True, device: torch.device | str = "cpu", **kwargs
    ):
        super(BM3D, self).__init__()
        self.use_legacy = use_legacy
        self.bm3d_lib = None
        if self.use_legacy:
            try:
                import bm3d

                self.bm3d_lib = bm3d
            except ImportError:  # pragma: no cover
                raise ImportError(
                    "bm3d package not found. Please install it with `pip install bm3d`."
                )
        else:
            # patch_size
            self.p = int(kwargs.get("patch_size", 8))
            # grouping settings
            self.search_radius = int(max(0, int(kwargs.get("search_radius", 19))))
            self.search_step = int(max(1, int(kwargs.get("search_step", 1))))
            self.ref_stride = int(max(1, int(kwargs.get("ref_stride", 3))))
            self.chunk_size = int(
                max(1, int(kwargs.get("chunk_size", 2048)))
            )  # number of groups to process in parallel
            self.offset_radius = int(
                self.search_radius // self.search_step
            )  # search radius (steps)
            self.n_candidates = (
                int(2 * self.offset_radius + 1) ** 2
            )  # upper bound on group size
            self.ht_group_size = int(
                min(max(1, int(kwargs.get("ht_group_size", 16))), self.n_candidates)
            )  # stage 1 (hard-threshold) group size
            self.wiener_group_size = int(
                min(max(1, int(kwargs.get("wiener_group_size", 32))), self.n_candidates)
            )  # stage 2 (Wiener filtering) group size
            # 2d transforms (along spatial dimensions)
            spatial_ht, spatial_ht_inv = self._transform_matrices(
                self.p, kwargs.get("spatial_ht_transform", "bior1.5"), device=device
            )  # biorthogonal WT, (p, p)
            self.register_buffer("spatial_ht", spatial_ht)
            self.register_buffer("spatial_ht_inv", spatial_ht_inv)
            spatial_wiener, spatial_wiener_inv = self._transform_matrices(
                self.p, kwargs.get("spatial_wiener_transform", "dct"), device=device
            )  # DCT, (p, p)
            self.register_buffer("spatial_wiener", spatial_wiener)
            self.register_buffer("spatial_wiener_inv", spatial_wiener_inv)
            # 1d transforms (along group dimension)
            group_ht, group_ht_inv = self._transform_matrices(
                self.ht_group_size,
                kwargs.get("group_ht_transform", "haar"),
                device=device,
            )  # Haar, (k_ht, k_ht)
            self.register_buffer("group_ht", group_ht)
            self.register_buffer("group_ht_inv", group_ht_inv)
            group_wiener, group_wiener_inv = self._transform_matrices(
                self.wiener_group_size,
                kwargs.get("group_wiener_transform", "haar"),
                device=device,
            )  # Haar, (k_w, k_w)
            self.register_buffer("group_wiener", group_wiener)
            self.register_buffer("group_wiener_inv", group_wiener_inv)
            # Kaiser window, applied to weights during aggregation to reduce boundary artifact
            win_1d = torch.kaiser_window(
                self.p,
                beta=2.0,
                periodic=False,
                dtype=torch.float32,
                device=device,
            )  # (p,)
            self.register_buffer("win", win_1d[:, None] * win_1d[None, :])  # (p, p)
            # helper arrays for aggregation
            self.register_buffer(
                "patch_y",
                torch.arange(self.p, dtype=torch.int64, device=device)[
                    None, None, None, :, None
                ],
            )  # (1, 1, 1, p, 1)
            self.register_buffer(
                "patch_x",
                torch.arange(self.p, dtype=torch.int64, device=device)[
                    None, None, None, None, :
                ],
            )  # (1, 1, 1, 1, p)
            # other configs
            self.hard_threshold = float(kwargs.get("hard_threshold", 3.0))
            self.wiener_mu2 = float(kwargs.get("wiener_mu2", 0.4))
            self.to(device)

    def _get_wavelet_matrix_torch(
        self,
        n: int,
        wavelet: str,
        level: int | None = None,
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor]:
        try:
            import ptwt
        except ImportError:  # pragma: no cover
            raise ImportError(
                "ptwt package not found. Please install it with `pip install ptwt`."
            )
        if level is None:
            level = n.bit_length() - 1
        I = torch.eye(n, dtype=torch.float32, device=device)
        matrix = torch.empty((n, n), dtype=torch.float32, device=device)
        for i in range(n):
            basis = I[:, i]
            coeffs = ptwt.wavedec(basis, wavelet, mode="periodic", level=level)
            matrix[:, i] = torch.hstack(
                [coeff[: 2 ** max(0, j - 1)] for j, coeff in enumerate(coeffs)]
            )  # truncate output to match the behavior of `pywt.wavedec` when `model='periodization'`
        matrix = matrix / torch.linalg.norm(matrix, dim=0)
        return matrix, torch.linalg.inv(matrix)

    def _get_dct_matrix(
        self, n: int, norm: str = "ortho", device: torch.device | None = None
    ) -> tuple[Tensor, Tensor]:
        matrix = dct(torch.eye(n, dtype=torch.float32, device=device), norm=norm).T
        inverse = matrix.T if norm == "ortho" else torch.linalg.inv(matrix)
        return matrix, inverse

    def _transform_matrices(
        self, n: int, transform_name: str, device: torch.device | None = None
    ) -> tuple[Tensor, Tensor]:
        n = int(n)
        transform_name = transform_name.lower()
        if transform_name == "dct":
            return self._get_dct_matrix(n, norm="ortho", device=device)
        elif (n & (n - 1)) == 0:
            return self._get_wavelet_matrix_torch(n, transform_name, device=device)
        raise ValueError(f"Unsupported transform '{transform_name}' for size n={n}.")

    def _patch_tensor(
        self,
        img: Tensor,
    ) -> Tensor:  # extract all `hp*wp` patches
        patches = image_to_patches(
            img.permute(2, 0, 1).unsqueeze(0),  # (1, c, h, w)
            self.p,
            1,
            False,
        )  # (1, c, hp, wp, p, p)
        return (
            patches.permute(2, 3, 0, 1, 4, 5)
            .contiguous()
            .reshape(-1, img.shape[2], self.p, self.p)
        )  # (hp*wp, c, p, p)

    def _reference_grid(
        self, hp: int, wp: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:  # get locations of all reference patches
        ys = torch.arange(0, hp, self.ref_stride, dtype=torch.int64, device=device)
        xs = torch.arange(0, wp, self.ref_stride, dtype=torch.int64, device=device)
        # always include the last patch to avoid boundary artifact
        if ((hp - 1) // self.ref_stride) * self.ref_stride != hp - 1:
            ys = torch.cat(
                (ys, torch.as_tensor([hp - 1], dtype=torch.int64, device=device))
            )
        if ((wp - 1) // self.ref_stride) * self.ref_stride != wp - 1:
            xs = torch.cat(
                (xs, torch.as_tensor([wp - 1], dtype=torch.int64, device=device))
            )
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return yy.reshape(-1), xx.reshape(-1)  # (n_refs,), (n_refs,)

    def _search_offsets(
        self, device: torch.device
    ) -> tuple[Tensor, Tensor]:  # get candidate offsets
        offsets = torch.arange(
            -self.offset_radius * self.search_step,
            self.offset_radius * self.search_step + 1,
            self.search_step,
            dtype=torch.int64,
            device=device,
        )  # (2*offset_radius + 1,)
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        return yy.reshape(-1), xx.reshape(-1)  # (n_candidates,), (n_candidates,)

    def _spatial_forward(self, blocks: Tensor, transform: Tensor) -> Tensor:
        tmp = torch.einsum("ai,nkcij->nkcaj", transform, blocks)
        return torch.einsum("bj,nkcaj->nkcab", transform, tmp)

    def _spatial_inverse(self, coeffs: Tensor, inverse: Tensor) -> Tensor:
        tmp = torch.einsum("ia,nkcab->nkcib", inverse, coeffs)
        return torch.einsum("jb,nkcib->nkcij", inverse, tmp)

    def _group_forward(self, blocks: Tensor, transform: Tensor) -> Tensor:
        return torch.einsum("mk,nkcab->nmcab", transform, blocks)

    def _group_inverse(self, coeffs: Tensor, inverse: Tensor) -> Tensor:
        return torch.einsum("km,nmcab->nkcab", inverse, coeffs)

    def _aggregate(
        self,
        w: int,
        c: int,
        accum: Tensor,
        weight: Tensor,
        pos_y: Tensor,
        pos_x: Tensor,
        patch_c: Tensor,
        patches: Tensor,
        patch_weight: Tensor,
    ) -> None:
        # all `p` pixel indices within all `k` patches for all `n` groups
        iy = pos_y[:, :, None, None, None] + self.patch_y  # (n, k, 1, p, 1)
        ix = pos_x[:, :, None, None, None] + self.patch_x  # (n, k, 1, 1, p)
        idx = ((iy * w + ix) * c + patch_c).reshape(-1)
        weighted_window = (
            self.win[None, None, None, :, :] * patch_weight
        )  # (n, 1, c, p, p)
        values = patches * weighted_window  # (n, k, c, p, p)
        weights = weighted_window.expand_as(patches)  # (n, k, c, p, p)
        accum.index_add_(0, idx, values.reshape(-1))  # accum[idx] += values.ravel()
        weight.index_add_(0, idx, weights.reshape(-1))  # weight[idx] += weights.ravel()

    def _estimate(
        self,
        h: int,
        w: int,
        c: int,
        n_refs: int,
        ref_y: Tensor,
        ref_x: Tensor,
        off_y: Tensor,
        off_x: Tensor,
        patch_c: Tensor,
        hp: int,
        wp: int,
        sigma: float,
        match_patches: Tensor,
        noisy_patches: Tensor,
        group_size: int,
        spatial_transform: Tensor,
        spatial_inverse: Tensor,
        group_transform: Tensor,
        group_inverse: Tensor,
        stage: str,
        device: torch.device,
    ) -> Tensor:
        # patch features
        match_metric = match_patches[:, 0, :, :].reshape(
            -1, self.p * self.p
        )  # (hp*wp, p*p)
        patch_norm = self.p * self.p
        # initialize accumulators
        accum = torch.zeros(h * w * c, dtype=torch.float32, device=device)  # (h*w*c,)
        weight = torch.zeros_like(accum)  # (h*w*c,)
        # process `chunk_size` groups in parallel
        for start in range(0, n_refs, self.chunk_size):
            # indices handling
            end = min(start + self.chunk_size, n_refs)
            ry = ref_y[start:end]  # (chunk_size,)
            rx = ref_x[start:end]  # (chunk_size,)
            ref_idx = ry * wp + rx  # (chunk_size,)
            cand_y_raw = ry[:, None] + off_y[None, :]  # (chunk_size, n_candidates)
            cand_x_raw = rx[:, None] + off_x[None, :]  # (chunk_size, n_candidates)
            valid = (
                (cand_y_raw >= 0)
                & (cand_y_raw < hp)
                & (cand_x_raw >= 0)
                & (cand_x_raw < wp)
            )  # (chunk_size, n_candidates)
            cand_y = torch.clamp(cand_y_raw, 0, hp - 1)  # (chunk_size, n_candidates)
            cand_x = torch.clamp(cand_x_raw, 0, wp - 1)  # (chunk_size, n_candidates)
            cand_idx = cand_y * wp + cand_x  # (chunk_size, n_candidates)
            # block matching
            ref = match_metric[ref_idx]  # (chunk_size, p*p)
            cand = match_metric[cand_idx]  # (chunk_size, n_candidates, p*p)
            dist = (
                torch.cdist(
                    ref[:, None, :],  # (chunk_size, 1, p*p)
                    cand,  # (chunk_size, n_candidates, p*p)
                    p=2.0,
                ).squeeze(1)
                ** 2
                / patch_norm
            )  # (chunk_size, n_candidates)
            dist = torch.where(
                valid,
                dist,
                torch.inf,
            )  # (chunk_size, n_candidates)
            nearest = torch.topk(
                dist, k=group_size, dim=1, largest=False
            ).indices  # select top-`group_size`
            nearest_dist = torch.gather(dist, 1, nearest)  # (chunk_size, group_size)
            order = torch.argsort(nearest_dist, dim=1)  # (chunk_size, group_size)
            nearest = torch.gather(nearest, 1, order)  # (chunk_size, group_size)
            group_y = torch.gather(cand_y, 1, nearest)  # (chunk_size, group_size)
            group_x = torch.gather(cand_x, 1, nearest)  # (chunk_size, group_size)
            group_idx = torch.gather(cand_idx, 1, nearest)  # (chunk_size, group_size)
            # extract matched patches from noisy image and stack into a group
            noisy_group = noisy_patches[group_idx].reshape(
                -1, group_size, c, self.p, self.p
            )  # (chunk_size, group_size, c, p, p)
            # forward 3d transform
            noisy_coeff = self._group_forward(
                self._spatial_forward(noisy_group, spatial_transform), group_transform
            )  # (chunk_size, group_size, c, p, p)
            # stage 1 (hard-threshold)
            if stage == "hard":
                mask = (
                    torch.abs(noisy_coeff) >= self.hard_threshold * sigma
                )  # (chunk_size, group_size, c, p, p)
                coeff = noisy_coeff * mask
                denom = torch.clamp_min(
                    mask.sum(dim=(1, 3, 4)).to(dtype=torch.float32),
                    1.0,
                )  # (chunk_size, group_size)
            # stage 2 (Wiener filtering)
            else:
                pilot_group = match_patches[group_idx].reshape(
                    -1, group_size, c, self.p, self.p
                )  # (chunk_size, group_size, c, p, p)
                pilot_coeff = self._group_forward(
                    self._spatial_forward(pilot_group, spatial_transform),
                    group_transform,
                )  # (chunk_size, group_size, c, p, p)
                # compute gain
                pilot_power = (
                    pilot_coeff * pilot_coeff
                )  # (chunk_size, group_size, c, p, p)
                noise_power = self.wiener_mu2 * sigma * sigma  # scalar
                wiener = pilot_power / (
                    pilot_power + noise_power
                )  # (chunk_size, group_size, c, p, p)
                # apply gain
                coeff = noisy_coeff * wiener
                denom = torch.clamp_min(
                    torch.sum(wiener * wiener, dim=(1, 3, 4)),
                    1.0,
                )  # (chunk_size, group_size)
            # inverse 3d transform
            filtered = self._spatial_inverse(
                self._group_inverse(coeff, group_inverse), spatial_inverse
            )  # (chunk_size, group_size, c, p, p)
            # compute aggregation weights
            patch_weight = (1.0 / (sigma * sigma * denom))[
                :, None, :, None, None
            ]  # (chunk_size, 1, c, 1, 1)
            # aggregate
            self._aggregate(
                w, c, accum, weight, group_y, group_x, patch_c, filtered, patch_weight
            )
        # normalize
        out = accum.reshape(h, w, c) / torch.clamp_min(
            weight.reshape(h, w, c),
            torch.finfo(torch.float32).eps,
        )  # (h, w, c)
        return out

    def _bm3d_fast(self, y: Tensor, sigma: float) -> Tensor:
        # y
        device = y.device
        h, w, c = y.shape
        if h < self.p or w < self.p:
            raise ValueError(
                f"Expect the height and width of the input image to be at least {self.p}, got {h} and {w}."
            )
        # patches
        hp, wp = h - self.p + 1, w - self.p + 1  # hp*wp = total number of patches
        ref_y, ref_x = self._reference_grid(hp, wp, device)  # (n_refs,), (n_refs,)
        off_y, off_x = self._search_offsets(device)  # (n_candidates,), (n_candidates,)
        n_refs = int(ref_y.numel())  # total number of reference patches
        patch_c = torch.arange(c, dtype=torch.int64, device=device)[
            None, None, :, None, None
        ]  # (1, 1, c, 1, 1)
        # stage 1 (hard-threshold)
        y_patches = self._patch_tensor(y)  # (hp*wp, c, p, p)
        basic = self._estimate(
            h,
            w,
            c,
            n_refs,
            ref_y,
            ref_x,
            off_y,
            off_x,
            patch_c,
            hp,
            wp,
            sigma,
            y_patches,
            y_patches,
            self.ht_group_size,
            self.spatial_ht,
            self.spatial_ht_inv,
            self.group_ht,
            self.group_ht_inv,
            "hard",
            device,
        )  # (h, w, c)
        # stage 2 (Wiener filtering)
        basic_patches = self._patch_tensor(basic)  # (hp*wp, c, p, p)
        denoised = self._estimate(
            h,
            w,
            c,
            n_refs,
            ref_y,
            ref_x,
            off_y,
            off_x,
            patch_c,
            hp,
            wp,
            sigma,
            basic_patches,
            y_patches,
            self.wiener_group_size,
            self.spatial_wiener,
            self.spatial_wiener_inv,
            self.group_wiener,
            self.group_wiener_inv,
            "wiener",
            device,
        )  # (h, w, c)
        return denoised

    def forward(self, x: Tensor, sigma: float | Tensor) -> Tensor:
        sigma = self._handle_sigma(sigma, batch_size=x.size(0))
        if self.use_legacy:
            out = torch.empty_like(x)
            for i in range(x.shape[0]):
                out[i, :, :, :] = array2tensor(
                    self.bm3d_lib.bm3d(tensor2array(x[i, :, :, :]), sigma[i].item())
                )
        else:
            x = x.permute(0, 2, 3, 1)  # (b, h, w, c)
            out = torch.empty_like(x)
            for i in range(x.shape[0]):
                out[i, :, :, :] = self._bm3d_fast(
                    x[i, :, :, :],  # (h, w, c)
                    sigma[i].item(),
                )  # (h, w, c)
            out = out.permute(0, 3, 1, 2)  # (b, c, h, w)
        return out
