from __future__ import annotations
import numpy as np
import torch
from torch import Tensor
from .base import Denoiser
from .utils import array2tensor, tensor2array

EPS = 2 ** (-52)


class BM3D(Denoiser):
    r"""
    BM3D denoiser.

    The BM3D denoiser was introduced by :footcite:t:`dabov2007image`.


    .. note::

        Unlike other denoisers from the library, this denoiser is applied sequentially to each noisy image in the batch
        (no parallelization). Furthermore, it does not support backpropagation.

    .. warning::

        This module wraps the BM3D denoiser from the `BM3D python package <https://pypi.org/project/bm3d/>`_.
        It can be installed with ``pip install bm3d``.

    """

    def __init__(self, use_legacy: bool = True):
        super(BM3D, self).__init__()
        self.use_legacy = use_legacy

    def forward(self, x: Tensor, sigma: float | Tensor, **kwargs) -> Tensor:
        if self.use_legacy:
            try:
                import bm3d
            except ImportError:  # pragma: no cover
                raise ImportError(
                    "bm3d package not found. Please install it with `pip install bm3d`."
                )

        out = torch.empty_like(x)

        sigma = self._handle_sigma(sigma, batch_size=x.size(0))

        for i in range(x.shape[0]):
            if self.use_legacy:
                out[i, :, :, :] = array2tensor(
                    bm3d.bm3d(tensor2array(x[i, :, :, :]), sigma[i].item())
                )
            else:
                out[i, :, :, :] = bm3d_fast(
                    x[i, :, :, :].permute(1, 2, 0),
                    sigma[i].item(),
                    patch_size=kwargs.get("patch_size", 8),
                    ht_group_size=kwargs.get("ht_group_size", 16),
                    wiener_group_size=kwargs.get("wiener_group_size", 32),
                    search_radius=kwargs.get("search_radius", 19),
                    search_step=kwargs.get("search_step", 1),
                    ref_stride=kwargs.get("ref_stride", 3),
                    hard_threshold=kwargs.get("hard_threshold", 3.0),
                    wiener_mu2=kwargs.get("wiener_mu2", 0.4),
                    chunk_size=kwargs.get("chunk_size", 2048),
                ).permute(2, 0, 1)
        return out


def get_wavelet_matrix_torch(n, wavelet, level=None):
    try:
        import ptwt
    except ImportError:  # pragma: no cover
        raise ImportError(
            "ptwt package not found. Please install it with `pip install ptwt`."
        )
    if level is None:
        level = n.bit_length() - 1
    I = torch.eye(n, dtype=torch.float32)
    matrix = torch.empty((n, n), dtype=torch.float32)
    for i in range(n):
        basis = I[:, i]
        coeffs = ptwt.wavedec(basis, wavelet, mode="periodic", level=level)
        matrix[:, i] = torch.hstack(
            [coeff[: 2 ** max(0, j - 1)] for j, coeff in enumerate(coeffs)]
        )  # truncate output to match the behavior of `pywt.wavedec` when `model='periodization'`
    matrix = matrix / torch.linalg.norm(matrix, dim=0)
    return matrix, torch.linalg.inv(matrix)


def get_dct_matrix(n, norm="ortho"):
    try:
        from scipy.fft import dct
    except ImportError:  # pragma: no cover
        raise ImportError(
            "scipy package not found. Please install it with `pip install scipy`."
        )
    matrix = dct(np.eye(n, dtype=np.float32), norm=norm, axis=0)
    inverse = matrix.T if norm == "ortho" else np.linalg.inv(matrix)
    return torch.as_tensor(matrix, dtype=torch.float32), torch.as_tensor(
        inverse, dtype=torch.float32
    )


def get_transform_matrices(n, transform_name):
    n = int(n)
    transform_name = transform_name.lower()
    if transform_name == "dct":
        return get_dct_matrix(n, norm="ortho")
    elif (n & (n - 1)) == 0:
        return get_wavelet_matrix_torch(n, transform_name)
    raise ValueError(f"Unsupported transform '{transform_name}' for size n={n}.")


@torch.no_grad()
def bm3d_fast(
    y,
    sigma,
    patch_size=8,
    ht_group_size=16,
    wiener_group_size=32,
    search_radius=19,
    search_step=1,
    ref_stride=3,
    hard_threshold=3.0,
    wiener_mu2=0.4,
    chunk_size=2048,
):
    # y
    y = torch.as_tensor(y, dtype=torch.float32)
    squeeze = y.ndim == 2
    if squeeze:
        y = y[..., None]  # (h, w, c)
    h, w, c = y.shape
    device = y.device
    # sigma
    sigma = torch.as_tensor(sigma, dtype=torch.float32, device=device)
    if sigma.ndim == 0:
        sigma = torch.full(
            (c,), float(sigma), dtype=torch.float32, device=device
        )  # (c,)
    elif sigma.numel() == c:
        sigma = sigma.reshape(c).to(dtype=torch.float32)  # (c,)
    if torch.all(sigma <= 0).item():  # edge case: no denoising
        return y[..., 0] if squeeze else y
    sigma_ch = torch.as_tensor(sigma, dtype=torch.float32, device=device)  # (c,)
    sigma2_ch = sigma_ch * sigma_ch  # (c,)
    # patch_size
    p = min(int(patch_size), h, w)
    # grouping settings
    search_radius = max(0, int(search_radius))
    search_step = max(1, int(search_step))
    ref_stride = max(1, int(ref_stride))
    chunk_size = max(1, int(chunk_size))  # number of groups to process in parallel
    offset_radius = search_radius // search_step  # search radius (steps)
    hp, wp = h - p + 1, w - p + 1  # hp*wp = total number of patches
    n_candidates = (2 * offset_radius + 1) ** 2  # upper bound on group size
    ht_group_size = int(
        min(max(1, int(ht_group_size)), n_candidates)
    )  # stage 1 (hard-threshold) group size
    wiener_group_size = int(
        min(max(1, int(wiener_group_size)), n_candidates)
    )  # stage 2 (Wiener filtering) group size

    # transform matrices
    def _transform_matrices(n, transform_name):  # get transform matrices
        forward, inverse = get_transform_matrices(n, transform_name)
        return forward.to(device=device, dtype=torch.float32), inverse.to(
            device=device, dtype=torch.float32
        )

    # 2d transforms (along spatial dimensions)
    spatial_ht, spatial_ht_inv = _transform_matrices(
        p, "bior1.5"
    )  # biorthogonal WT, (p, p)
    spatial_wiener, spatial_wiener_inv = _transform_matrices(p, "dct")  # DCT, (p, p)
    # 1d transforms (along group dimension)
    group_ht, group_ht_inv = _transform_matrices(
        ht_group_size, "haar"
    )  # Haar, (k_ht, k_ht)
    group_wiener, group_wiener_inv = _transform_matrices(
        wiener_group_size, "haar"
    )  # Haar, (k_w, k_w)
    # Kaiser window, applied to weights during aggregation to reduce boundary artifact
    win_1d = torch.kaiser_window(
        p, beta=2.0, periodic=False, device=device, dtype=torch.float32
    )  # (p,)
    win = win_1d[:, None] * win_1d[None, :]  # (p, p)

    # helper functions
    def _patch_tensor(img):  # extract all `hp*wp` patches
        patches = img.unfold(0, p, 1).unfold(1, p, 1)  # Result: (hp, wp, c, p, p)
        return patches.contiguous().reshape(-1, img.shape[2], p, p)  # (hp*wp, c, p, p)

    def _reference_grid():  # get locations of all reference patches
        ys = torch.arange(0, hp, ref_stride, dtype=torch.int64, device=device)
        xs = torch.arange(0, wp, ref_stride, dtype=torch.int64, device=device)
        # always include the last patch to avoid boundary artifact
        if ((hp - 1) // ref_stride) * ref_stride != hp - 1:
            ys = torch.cat(
                (ys, torch.as_tensor([hp - 1], dtype=torch.int64, device=device))
            )
        if ((wp - 1) // ref_stride) * ref_stride != wp - 1:
            xs = torch.cat(
                (xs, torch.as_tensor([wp - 1], dtype=torch.int64, device=device))
            )
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return yy.reshape(-1), xx.reshape(-1)  # (n_refs,), (n_refs,)

    def _search_offsets():  # get candidate offsets
        offsets = torch.arange(
            -offset_radius * search_step,
            offset_radius * search_step + 1,
            search_step,
            dtype=torch.int64,
            device=device,
        )  # (2*offset_radius + 1,)
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        return yy.reshape(-1), xx.reshape(-1)  # (n_candidates,), (n_candidates,)

    ref_y, ref_x = _reference_grid()  # (n_refs,), (n_refs,)
    off_y, off_x = _search_offsets()  # (n_candidates,), (n_candidates,)
    n_refs = int(ref_y.numel())  # total number of reference patches
    # helper arrays
    patch_y = torch.arange(p, dtype=torch.int64, device=device)[
        None, None, None, :, None
    ]  # (1, 1, 1, p, 1)
    patch_x = torch.arange(p, dtype=torch.int64, device=device)[
        None, None, None, None, :
    ]  # (1, 1, 1, 1, p)
    patch_c = torch.arange(c, dtype=torch.int64, device=device)[
        None, None, :, None, None
    ]  # (1, 1, c, 1, 1)

    # functions for transforms
    def _spatial_forward(blocks, transform):
        tmp = torch.einsum("ai,nkcij->nkcaj", transform, blocks)
        return torch.einsum("bj,nkcaj->nkcab", transform, tmp)

    def _spatial_inverse(coeffs, inverse):
        tmp = torch.einsum("ia,nkcab->nkcib", inverse, coeffs)
        return torch.einsum("jb,nkcib->nkcij", inverse, tmp)

    def _group_forward(blocks, transform):
        return torch.einsum("mk,nkcab->nmcab", transform, blocks)

    def _group_inverse(coeffs, inverse):
        return torch.einsum("km,nmcab->nkcab", inverse, coeffs)

    # functions for aggregation
    def _aggregate(accum, weight, pos_y, pos_x, patches, patch_weight):
        # all `p` pixel indices within all `k` patches for all `n` groups
        iy = pos_y[:, :, None, None, None] + patch_y  # (n, k, 1, p, 1)
        ix = pos_x[:, :, None, None, None] + patch_x  # (n, k, 1, 1, p)
        idx = ((iy * w + ix) * c + patch_c).reshape(-1)
        weighted_window = win[None, None, None, :, :] * patch_weight  # (n, 1, c, p, p)
        values = patches * weighted_window  # (n, k, c, p, p)
        weights = weighted_window.expand_as(patches)  # (n, k, c, p, p)
        accum.index_add_(0, idx, values.reshape(-1))  # accum[idx] += values.ravel()
        weight.index_add_(0, idx, weights.reshape(-1))  # weight[idx] += weights.ravel()

    # main routine for both stages
    def _estimate(
        match_patches,
        noisy_patches,
        group_size,
        spatial_transform,
        spatial_inverse,
        group_transform,
        group_inverse,
        stage,
    ):
        # patch features
        match_metric = match_patches[:, 0, :, :].reshape(-1, p * p)  # (hp*wp, p*p)
        patch_norm = torch.tensor(p * p, dtype=torch.float32, device=device)
        # initialize accumulators
        accum = torch.zeros(h * w * c, dtype=torch.float32, device=device)  # (h*w*c,)
        weight = torch.zeros_like(accum)  # (h*w*c,)
        # process `chunk_size` groups in parallel
        for start in range(0, n_refs, chunk_size):
            # indices handling
            end = min(start + chunk_size, n_refs)
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
                torch.sum((cand - ref[:, None, :]) ** 2, dim=2) / patch_norm
            )  # (chunk_size, n_candidates)
            dist = torch.where(
                valid, dist, torch.full_like(dist, torch.inf)
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
                -1, group_size, c, p, p
            )  # (chunk_size, group_size, c, p, p)
            # forward 3d transform
            noisy_coeff = _group_forward(
                _spatial_forward(noisy_group, spatial_transform), group_transform
            )  # (chunk_size, group_size, c, p, p)
            # stage 1 (hard-threshold)
            if stage == "hard":
                threshold = (
                    torch.tensor(hard_threshold, dtype=torch.float32, device=device)
                    * sigma_ch[None, None, :, None, None]
                )  # (1, 1, c, 1, 1)
                mask = (
                    torch.abs(noisy_coeff) >= threshold
                )  # (chunk_size, group_size, c, p, p)
                coeff = noisy_coeff * mask
                denom = torch.maximum(
                    mask.sum(dim=(1, 3, 4)).to(dtype=torch.float32),
                    torch.tensor(1.0, dtype=torch.float32, device=device),
                )  # (chunk_size, group_size)
            # stage 2 (Wiener filtering)
            else:
                pilot_group = match_patches[group_idx].reshape(
                    -1, group_size, c, p, p
                )  # (chunk_size, group_size, c, p, p)
                pilot_coeff = _group_forward(
                    _spatial_forward(pilot_group, spatial_transform), group_transform
                )  # (chunk_size, group_size, c, p, p)
                # compute gain
                pilot_power = (
                    pilot_coeff * pilot_coeff
                )  # (chunk_size, group_size, c, p, p)
                noise_power = (
                    torch.tensor(wiener_mu2, dtype=torch.float32, device=device)
                    * sigma2_ch[None, None, :, None, None]
                )  # (1, 1, c, 1, 1)
                wiener = pilot_power / (
                    pilot_power + noise_power
                )  # (chunk_size, group_size, c, p, p)
                # apply gain
                coeff = noisy_coeff * wiener
                denom = torch.maximum(
                    torch.sum(wiener * wiener, dim=(1, 3, 4)),
                    torch.tensor(1.0, dtype=torch.float32, device=device),
                )  # (chunk_size, group_size)
            # inverse 3d transform
            filtered = _spatial_inverse(
                _group_inverse(coeff, group_inverse), spatial_inverse
            )  # (chunk_size, group_size, c, p, p)
            # compute aggregation weights
            patch_weight = (1.0 / (sigma2_ch[None, :] * denom))[
                :, None, :, None, None
            ]  # (chunk_size, 1, c, 1, 1)
            # aggregate
            _aggregate(accum, weight, group_y, group_x, filtered, patch_weight)
        # normalize
        out = accum.reshape(h, w, c) / torch.maximum(
            weight.reshape(h, w, c),
            torch.tensor(EPS, dtype=torch.float32, device=device),
        )  # (h, w, c)
        return out

    # stage 1 (hard-threshold)
    y_patches = _patch_tensor(y)  # (hp*wp, c, p, p)
    basic = _estimate(
        y_patches,
        y_patches,
        ht_group_size,
        spatial_ht,
        spatial_ht_inv,
        group_ht,
        group_ht_inv,
        "hard",
    )  # (h, w, c)
    # stage 2 (Wiener filtering)
    basic_patches = _patch_tensor(basic)  # (hp*wp, c, p, p)
    denoised = _estimate(
        basic_patches,
        y_patches,
        wiener_group_size,
        spatial_wiener,
        spatial_wiener_inv,
        group_wiener,
        group_wiener_inv,
        "wiener",
    )  # (h, w, c)
    return denoised[..., 0] if squeeze else denoised
