from __future__ import annotations
import math, sys, io, requests
from pathlib import Path
import torch
import torch.nn.functional as F

from deepinv.loss.metric.metric import Metric
from deepinv.physics.functional.convolution import conv2d
from deepinv.physics.functional.imresize import imresize_matlab
from deepinv.models.utils import load_state_dict_from_url, get_weights_url


class LPIPS(Metric):
    r"""
    Learned Perceptual Image Patch Similarity (LPIPS) metric.

    Calculates the LPIPS :math:`\text{LPIPS}(\hat{x},x)` where :math:`\hat{x}=\inverse{y}`.

    Computes the perceptual similarity between two images, based on a pre-trained deep neural network.
    Uses implementation from `torchmetrics <https://lightning.ai/docs/torchmetrics/stable/image/learned_perceptual_image_patch_similarity.html>`_.

    The inputs `x_net`, `x` must both have 3 channels and be in `[0, 1]`. Optionally use `norm_inputs` argument to clip to `[0, 1]`.

    .. note::

        By default, no reduction is performed in the batch dimension.

    :Example:

    ::

        from deepinv.utils import load_example
        from deepinv.loss.metric import LPIPS
        m = LPIPS()
        x = torch.ones(2, 3, 32, 32)
        x_net = x - 0.01
        m(x_net, x)

    :param str net_type: network architecture to use. Options: 'alex', 'vgg', 'squeeze'. Default: 'alex'.
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2`` normalizes by :math:`\ell_2` spatial norm, ``min_max`` normalizes by min and max of each input.
    :param bool check_input_range: if True, ``pyiqa`` will raise error if inputs aren't in the appropriate range ``[0, 1]``.
    :param int, tuple[int], None center_crop: If not `None` (default), center crop the tensor(s) before computing the metrics.
        If an `int` is provided, the cropping is applied equally on all spatial dimensions (by default, all dimensions except the first two).
        If `tuple` of `int`, cropping is performed over the last `len(center_crop)` dimensions. If positive values are provided, a standard center crop is applied.
        If negative (or zero) values are passed, cropping will be done by removing `center_crop` pixels from the borders (useful when tensors vary in size across the dataset).
    :param str, torch.device device: LPIPS net device.
    """

    def __init__(self, net_type="alex", device=None, **kwargs):
        super().__init__(**kwargs)
        from torchmetrics.functional.image.lpips import _lpips_update, _NoTrainLpips

        # Pre-load LPIPS net
        self.lpips_fn = _lpips_update

        # Load LPIPS. Note torchvision internally uses torch.hub.load_state_dict_from_url which
        # annoyingly unpredictably prints to stdout, so we suppress this.
        _stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            self.lpips_net = _NoTrainLpips(net=net_type).to(device=device)
        finally:
            sys.stdout = _stdout

        self.lower_better = True

    def metric(self, x_net, x, *args, **kwargs):
        if x_net.ndim != 4 or x.ndim != 4:
            raise ValueError(
                f"LPIPS metric requires 4D input (B, C, H, W), but got shapes {x_net.shape}, {x.shape}."
            )

        if not (x_net.shape[1] == x.shape[1] == 3):
            raise ValueError(
                f"LPIPS metric only supports 3-channel input, but got channels for x_net, x as {x_net.shape[1]}, {x.shape[1]}."
            )

        min_val, max_val = torch.aminmax(torch.cat([x_net, x], dim=0))
        if not ((min_val >= 0.0) & (max_val <= 1.0)):
            raise ValueError("LPIPS metric requires x_net and x to be between 0 and 1.")

        return self.lpips_fn(
            x_net,
            x,
            net=self.lpips_net,
            normalize=True,
        )


class NIQE(Metric):
    r"""
    Natural Image Quality Evaluator (NIQE) metric.

    Calculates the NIQE :math:`\text{NIQE}(\hat{x})` where :math:`\hat{x}=\inverse{y}`.
    It is a no-reference image quality metric that estimates the quality of images.

    This implementation is based on the original Matlab code (available at http://live.ece.utexas.edu/research/quality/niqe_release.zip).
    One exception is that the original code always converted the image to float64. This implementation converts
    to dtype specified at init, but always use float64 when calculating the pseudoinverse.

    .. note::

        The input image must be sufficiently large compared to ``patch_size`` to ensure an adequate number of
        patches can be extracted. NIQE fits a Multivariate Gaussian (MVG) model to
        the Natural Scene Statistics (NSS) features (MSCN coefficients) of these
        patches, then measures the distance between this model and a reference MVG
        pre-fitted on pristine natural images. Too few patches yield an unreliable
        covariance matrix estimate, degrading the accuracy of the quality score.

    .. note::

        ``denominator`` defaults to 1. This was used in the original work, with fitting and testing data in [0,255]. When working with
        another intensity scale, change ``denominator`` appropriately to ensure it doesn't dominate over σ. For example, ``denominator=1/255``
        is a good starting point for intensity scale [0,1].

    .. note::

        By default, no reduction is performed in the batch dimension.

    :param str weights_path: Path to weights created with ``.create_weights``. If 'download' (default), downloads the weights provided by :footcite:t:`mittal2012making`. If None, mu and cov are not initialized (useful when fitting custom weights).
    :param float denominator: stabilizer to add to the std in the image normalization step (eq.1). Defaults to 1
    :param bool round_tensor: whether to round the input. The original NIQE implementation used rounding and requires input to be range [0, 255]. Do not set round_tensor if incoming tensors will be in [0,1] style ranges. Defaults to False.
    :param int patch_size: spatial size of the square patches used to compute NSS features. Larger values yield more
        robust per-patch statistics but require larger inputs and produce fewer patches. Defaults to 96.
    :param int patch_overlap: number of pixels overlapped between adjacent patches (stride is ``patch_size - patch_overlap``).
        Increase to extract more patches from a given image. Defaults to 0.
    :param torch.device, str device: device to use for the metric computation. Default: 'cpu'.
    :param torch.dtype dtype: dtype used for the metric computation (the pseudoinverse is always computed in float64). Default: ``torch.float32``.
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2``normalizes by L2 spatial norm, ``min_max`` normalizes by min and max of each input.
    :param int, tuple[int], None center_crop: If not `None` (default), center crop the tensor(s) before computing the metrics.
        If an `int` is provided, the cropping is applied equally on all spatial dimensions (by default, all dimensions except the first two).
        If `tuple` of `int`, cropping is performed over the last `len(center_crop)` dimensions. If positive values are provided, a standard center crop is applied.
        If negative (or zero) values are passed, cropping will be done by removing `center_crop` pixels from the borders (useful when tensors vary in size across the dataset).
    """

    def __init__(
        self,
        weights_path: str | Path | None = "download",
        denominator: float = 1,
        round_tensor: bool = False,
        patch_size: int = 96,
        patch_overlap: int = 0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.round = round_tensor
        self.lower_better = True
        self.patch_size = patch_size
        self.device = device
        self.n_scales = 2
        self.patch_overlap = patch_overlap
        self.denominator = denominator
        self.dtype = dtype
        if weights_path == "download":
            url = get_weights_url("demo", "niqe_weights.pt")
            params = load_state_dict_from_url(
                url,
                map_location=lambda storage, loc: storage,
                file_name="niqe_weights.pt",
                weights_only=True,
            )

        elif weights_path is not None:
            params = torch.load(weights_path, weights_only=True)
        else:
            self.mu_p, self.cov_p = None, None
        if weights_path is not None:
            mu, cov = params["mu"], params["cov"]
            self.mu_p = mu.to(dtype=dtype, device=device)

            self.cov_p = cov.to(dtype=dtype, device=device)

    def estimate_aggd_param(self, vecs: torch.Tensor, eps: float = 1e-12):
        v = vecs
        neg = v < 0
        pos = v > 0

        cnt_neg = neg.sum(dim=1)
        cnt_pos = pos.sum(dim=1)

        # Allocate outputs as NaN by default (MATLAB mean([]) -> NaN)
        left_ms = torch.full(
            (v.shape[0],), float("nan"), device=v.device, dtype=v.dtype
        )
        right_ms = torch.full(
            (v.shape[0],), float("nan"), device=v.device, dtype=v.dtype
        )

        # Only compute where there are samples
        if (cnt_neg > 0).any():
            left_ms[cnt_neg > 0] = ((v * v) * neg).sum(dim=1)[cnt_neg > 0] / cnt_neg.to(
                v.dtype
            )[cnt_neg > 0]
        if (cnt_pos > 0).any():
            right_ms[cnt_pos > 0] = ((v * v) * pos).sum(dim=1)[
                cnt_pos > 0
            ] / cnt_pos.to(v.dtype)[cnt_pos > 0]

        leftstd = torch.sqrt(left_ms)
        rightstd = torch.sqrt(right_ms)

        gammahat = leftstd / torch.clamp(rightstd, min=eps)
        rhat = (v.abs().mean(dim=1) ** 2) / torch.clamp(v.pow(2).mean(dim=1), min=eps)

        gam = torch.arange(0.2, 10.0 + 1e-9, 0.001, device=v.device, dtype=v.dtype)
        r_gam = (self._gamma(2.0 / gam) ** 2) / (
            self._gamma(1.0 / gam) * self._gamma(3.0 / gam)
        )

        rhatnorm = (rhat * (gammahat**3 + 1.0) * (gammahat + 1.0)) / torch.clamp(
            (gammahat**2 + 1.0) ** 2, min=eps
        )

        diff = (r_gam.unsqueeze(0) - rhatnorm.unsqueeze(1)).pow(2)
        idx = diff.argmin(dim=1)
        alpha = gam[idx]

        beta_factor = torch.sqrt(self._gamma(1.0 / alpha) / self._gamma(3.0 / alpha))
        betal = leftstd * beta_factor
        betar = rightstd * beta_factor
        return alpha, betal, betar

    def _patch_features(
        self, structdis: torch.Tensor, k: int, stride: int
    ) -> torch.Tensor:
        """
        structdis: (B,1,H,W)
        returns: (B, L, 18), L is #patches
        """
        B, C, H, W = structdis.shape
        assert C == 1
        base_u = F.unfold(
            structdis, kernel_size=(k, k), stride=(stride, stride)
        )  # (B, k*k, L)

        L = base_u.shape[-1]
        base = base_u.transpose(1, 2).contiguous().view(B * L, k * k)

        a0, bl0, br0 = self.estimate_aggd_param(base)
        feat_cols = [a0, (bl0 + br0) * 0.5]

        patches = base_u.transpose(1, 2).contiguous().view(B * L, 1, k, k)
        shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in shifts:
            shifted = torch.roll(patches, shifts=(dr, dc), dims=(2, 3))
            pair_vec = (patches * shifted).view(B * L, k * k)

            a, bl, br = self.estimate_aggd_param(pair_vec)
            meanparam = (br - bl) * (self._gamma(2.0 / a) / self._gamma(1.0 / a))
            feat_cols += [a, meanparam, bl, br]

        feats = torch.stack(feat_cols, dim=1)  # (B*L, 18)
        return feats.view(B, L, 18)

    def niqe(self, x_net: torch.Tensor) -> torch.Tensor:
        kernel = self._gen_gauss_kernel()

        all_feats = []

        for scale in range(1, self.n_scales + 1):
            mu = conv2d(x_net, kernel, "replicate")
            mu_sq = mu * mu
            sigma = torch.sqrt(
                torch.abs(conv2d(x_net * x_net, kernel, "replicate") - mu_sq)
            )
            structdis = (x_net - mu) / (sigma + self.denominator)
            k = max(1, self.patch_size // scale)
            ov = self.patch_overlap // scale
            strd = max(1, k - ov)

            feats = self._patch_features(structdis, k, strd)  # (B, L, 18)
            all_feats.append(feats)

            if scale < self.n_scales:
                x_net = imresize_matlab(
                    x_net,
                    scale=0.5,
                    kernel="cubic",
                    antialiasing=True,
                    padding_type="reflect",
                )

        X = torch.cat(all_feats, dim=2)  # (B, L, 36)
        mu_d, cov_d = self._nanstats_rowdrop(X)  # MATLAB-like nanmean/nancov

        cov_p = self.cov_p.expand_as(cov_d)  # (B,36,36)
        mu_p = self.mu_p  # (36,)
        invcov = torch.linalg.pinv(
            0.5 * (cov_d.to(torch.float64) + cov_p.to(torch.float64))
        ).to(
            self.dtype
        )  # (B,36,36)
        diff = (mu_p.unsqueeze(0) - mu_d).unsqueeze(1)  # (B,1,36)
        score = torch.sqrt((diff @ invcov @ diff.transpose(1, 2)).squeeze())
        return score

    def _gen_gauss_kernel(self):
        # sigma per original code: 7/6, window size 7
        sigma = 7 / 6
        radius = 3
        ax = torch.arange(-radius, radius + 1, device=self.device, dtype=self.dtype)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
        kernel /= kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

    def _gamma(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(torch.lgamma(x))

    def _nanstats_rowdrop(self, X: torch.Tensor):
        """
        Returns:
        mu:  (B, F)
        cov: (B, F, F)
        Drops rows (patches) with any non-finite feature, per batch item.
        """
        B, L, Fdim = X.shape
        mu = X.new_full((B, Fdim), float("nan"))
        cov = X.new_full((B, Fdim, Fdim), float("nan"))

        for b in range(B):
            Xb = X[b]  # (L,F)
            valid = torch.isfinite(Xb).all(dim=1)
            Xv = Xb[valid]  # (Lv,F)

            Lv = Xv.shape[0]
            if Lv == 0:
                continue  # leave as NaN like MATLAB nanmean/nancov on all-NaN
            mu_b = Xv.mean(dim=0)
            mu[b] = mu_b

            if Lv < 2:
                continue  # covariance undefined -> NaN (match MATLAB behavior)
            Xc = Xv - mu_b
            cov[b] = (Xc.t() @ Xc) / (Lv - 1)

        return mu, cov

    def metric(self, x_net: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.mu_p is None or self.cov_p is None:
            raise RuntimeError(
                "NIQE weights not loaded. Either pass weights_path at init or call create_weights first."
            )
        if x_net.ndim != 4:  # pragma: no cover
            raise RuntimeError(
                f"NIQE expects batched, 2D data, but got tensor with {x_net.ndim} dimensions (shape: {x_net.shape})"
            )

        _, C, H, W = x_net.shape

        if H < self.patch_size or W < self.patch_size:  # pragma: no cover
            raise RuntimeError(
                f"NIQE requires images to have height and width larger than or equal to its patch size {self.patch_size}, but got batch of shape {x_net.shape}"
            )
        stride = self.patch_size - self.patch_overlap
        n_patches_h = (H - self.patch_size) // stride + 1
        n_patches_w = (W - self.patch_size) // stride + 1
        if n_patches_h * n_patches_w < 2:  # pragma: no cover
            raise RuntimeError(
                f"NIQE requires more than 1 patch to compute covariance, but got only {n_patches_h * n_patches_w} patches "
                f"for batch of shape {x_net.shape} with patch_size={self.patch_size} and patch_overlap={self.patch_overlap}. "
            )
        if C == 3:
            luminance_weights = torch.tensor(
                [0.29893602, 0.58704307, 0.11402090],
                dtype=x_net.dtype,
                device=x_net.device,
            ).view(
                1, 3, 1, 1
            )  # this matches https://github.com/mattools/matlab-image-class/blob/master/src/%40Image/rgb2gray.m
            x_net = F.conv2d(x_net, luminance_weights)
        if x_net.shape[1] != 1:  # pragma: no cover
            raise RuntimeError(
                f"NIQE only operates on single channel images. 3 channel (RGB) gets converted to relative luminance, but got {C}-channel input"
            )
        if self.round:
            x_net = x_net.round()
        block_hnum = math.floor(H / self.patch_size)
        block_wnum = math.floor(W / self.patch_size)
        x_net = x_net[
            :, :, : block_hnum * self.patch_size, : block_wnum * self.patch_size
        ]

        n = self.niqe(x_net).float()
        return n.unsqueeze(0) if n.dim() == 0 else n

    def create_weights(
        self,
        dataset: torch.utils.data.Dataset,
        sharpness_threshold: float = 0.75,
        save_path: str | Path | None = None,
    ):
        r"""
        Fit NIQE model parameters (mu_prisparam, cov_prisparam) from a dataset of 'pristine' images,
        following the original MATLAB pipeline with two scales and sharpness-based patch selection.
        ``patch_size``, ``patch_overlap``, and ``denominator`` used are those passed at init (unless modified post-init by user).

        ``dataset`` should yield a (C, H, W) ``Tensor``, where C=1 and C=3 are allowed. If C=3, RGB is assumed and will be converted
        to greyscale using 0.299*R + 0.587*G + 0.114*B.

        :param torch.utils.data.Dataset dataset: for each item, should yield a Tensor representing a
            distortion-free (pristine) image.
        :param float sharpness_threshold: only patches whose sharpness is at least
            ``sharpness_threshold`` of the per-image peak sharpness (measured from σ at scale 1) are kept.
        :param str save_path: Path to which weights are to be saved. Must have ``.pt`` extension. If not passed, weights are returned without saving.

        :return: (mu_prisparam, cov_prisparam) as self.dtype on self.device. Also updates self.mu_p, self.cov_p.
        """

        device = self.device
        dtype = torch.float32
        kernel = self._gen_gauss_kernel().to(device=device, dtype=dtype)

        all_feats = []

        for i, x in enumerate(dataset):

            if x.ndim == 2:
                x = x.unsqueeze(0)
            if x.ndim == 3 and x.shape[0] in (1, 3):
                pass
            else:
                raise RuntimeError(
                    f"Unsupported input shape {tuple(x.shape)}, expecting (C, H, W) with C in set(1,3)"
                )

            x = x.to(device=device, dtype=dtype).unsqueeze(0)

            if x.shape[1] == 3:
                luminance_weights = torch.tensor(
                    [0.29893602, 0.58704307, 0.11402090], dtype=x.dtype, device=x.device
                ).view(1, 3, 1, 1)
                x = F.conv2d(x, luminance_weights)

            if self.round:
                x = x.round()

            _, _, H, W = x.shape
            if H < self.patch_size or W < self.patch_size:
                print(
                    f"Sample {i} / {len(dataset)}: Too small H or Width, not included for weight creation."
                )
                continue  # too small -> should we raise a warning here?
            block_hnum = math.floor(H / self.patch_size)
            block_wnum = math.floor(W / self.patch_size)
            x = x[:, :, : block_hnum * self.patch_size, : block_wnum * self.patch_size]

            feats_scales = []
            sharpness = None

            x_scale = x
            for scale in range(1, self.n_scales + 1):
                mu = conv2d(x_scale, kernel, "replicate")
                mu_sq = mu * mu
                sigma = torch.sqrt(
                    torch.abs(conv2d(x_scale * x_scale, kernel, "replicate") - mu_sq)
                )
                structdis = (x_scale - mu) / (sigma + self.denominator)

                k = max(1, self.patch_size // scale)
                ov = self.patch_overlap // scale
                stride = max(1, k - ov)
                feats = self._patch_features(structdis, k, stride)  # (1, L, 18)
                feats_scales.append(feats)

                if scale == 1:
                    U = F.unfold(
                        sigma, kernel_size=(k, k), stride=(stride, stride)
                    )  # (1, k*k, L)
                    sharpness = U.mean(dim=1).squeeze(0)  # (L,)

                if scale < self.n_scales:
                    x_scale = imresize_matlab(
                        x_scale,
                        scale=0.5,
                        kernel="cubic",
                        antialiasing=True,
                        padding_type="reflect",
                    )

            feats_2scales = torch.cat(feats_scales, dim=2).squeeze(0)  # (L,36)

            if sharpness is None or sharpness.numel() == 0:
                continue
            th = sharpness_threshold * sharpness.max()
            keep_idx = (sharpness > th).nonzero(as_tuple=False).flatten()
            if keep_idx.numel() == 0:
                continue
            feats_kept = feats_2scales.index_select(0, keep_idx)
            feats_kept = feats_kept[torch.isfinite(feats_kept).all(dim=1)]
            if feats_kept.numel() == 0:
                continue
            all_feats.append(feats_kept)

        if not all_feats:
            raise RuntimeError(
                "No patches collected. Consider lowering sharpness_threshold or checking dataset."
            )

        prisparam = torch.cat(all_feats, dim=0).to(device=device, dtype=dtype)  # (N,36)

        mu = prisparam.double().mean(dim=0)  # (36,)
        xc = prisparam.double() - mu.unsqueeze(0)
        denom = max(1, prisparam.shape[0] - 1)
        cov = (xc.t() @ xc) / denom  # (36,36)

        self.mu_p = mu.to(dtype=self.dtype)
        self.cov_p = cov.to(dtype=self.dtype)

        if save_path is not None:
            self.mu_p.requires_grad_(False)
            self.cov_p.requires_grad_(False)
            torch.save({"mu": self.mu_p.cpu(), "cov": self.cov_p.cpu()}, save_path)

        return self.mu_p, self.cov_p


class BlurStrength(Metric):
    r"""
    No-reference blur strength metric for batched images.

    Returns a value in (0, 1) for each image in the batch, where 0 indicates a very sharp image and 1 indicates a very blurry image.

    The metric has been introduced in :cite:t:`crete2007blur`.

    :param int h_size: size of the uniform blur filter. Default: 11.
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2`` normalizes by :math:`{\ell}_2` spatial norm, ``min_max`` normalizes by min and max of each input.
    :param bool check_input_range: if True, ``pyiqa`` will raise error if inputs aren't in the appropriate range ``[0, 1]``.
    :param int, tuple[int], None center_crop: If not `None` (default), center crop the tensor(s) before computing the metrics.
        If an `int` is provided, the cropping is applied equally on all spatial dimensions (by default, all dimensions except the first two).
        If `tuple` of `int`, cropping is performed over the last `len(center_crop)` dimensions. If positive values are provided, a standard center crop is applied.
        If negative (or zero) values are passed, cropping will be done by removing `center_crop` pixels from the borders (useful when tensors vary in size across the dataset).

    |sep|

    :Example:

    >>> from deepinv.loss.metric import BlurStrength
    >>> m = BlurStrength()
    >>> x_net = torch.randn(2, 3, 16, 16)  # batch of 2 RGB images
    >>> m(x_net).shape
    torch.Size([2])

    """

    def __init__(self, h_size: int = 11, **kwargs):
        super().__init__(**kwargs)
        self.h_size = h_size
        self.lower_better = True

    def metric(self, x_net: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Compute blur strength metric for a batch of images.

        :param x_net: (B, C, ...) input tensors with C=1 or 3 channels. The spatial dimensions can be 1D, 2D, or higher.
        :return: (B,) tensor of blur strength values in (0,1) for each image in the batch.
        """
        assert x_net.shape[1] in [1, 3], "Input must have 1 or 3 channels."

        x = x_net

        if x.shape[1] == 3:  # RGB to grayscale
            x = 0.2989 * x[:, [0]] + 0.5870 * x[:, [1]] + 0.1140 * x[:, [2]]

        spatial = x.shape[2:]
        n_spatial = len(spatial)

        # crop
        slices = (slice(None), slice(None)) + tuple(slice(2, s - 1) for s in spatial)

        # Compute metric for each spatial axis
        results = []

        # spatial axes start at dim=2
        for ax in range(2, 2 + n_spatial):
            # 1D uniform blur
            filt = self.uniform_filter1d(x, self.h_size, axis=ax)

            # Sobel derivatives
            sharp = torch.abs(self.sobel1d(x, axis=ax))
            blur = torch.abs(self.sobel1d(filt, axis=ax))

            # clamp/sharpness difference
            t = torch.clamp(sharp - blur, min=0)

            # sums over all except batch dimension
            m1 = sharp[slices].sum(dim=list(range(1, sharp.ndim)))
            m2 = t[slices].sum(dim=list(range(1, t.ndim)))

            # per-image blur per-axis
            axis_blur = torch.abs(m1 - m2) / (m1 + 1e-12)
            results.append(axis_blur)

        results = torch.stack(results, dim=1)  # (B, n_spatial)
        return results.max(dim=1).values  # (B,)

    @staticmethod
    def uniform_filter1d(x: torch.Tensor, size: int, axis: int) -> torch.Tensor:
        r"""
        Batched 1D uniform filter along an arbitrary axis.

        :param torch.Tensor x: input tensor of shape `(B, C, ...)`
        :param int size: size of filter
        :param int axis: axis along which to compute filter
        :return: filtered tensor of shape `(B, C, ...)`
        """
        pad = size // 2
        kernel = torch.ones(1, 1, size, device=x.device, dtype=x.dtype) / size

        # move axis to last dim
        x_perm = x.transpose(axis, -1)
        orig_shape = x_perm.shape

        # flatten spatial dims except last
        x_flat = x_perm.reshape(-1, 1, orig_shape[-1])

        x_flat = F.pad(x_flat, (pad, pad), mode="reflect")
        out = F.conv1d(x_flat, kernel)

        out = out.reshape(orig_shape)
        out = out.transpose(axis, -1)
        return out

    @staticmethod
    def sobel1d(x: torch.Tensor, axis: int) -> torch.Tensor:
        r"""
        Batched 1D Sobel derivative along an arbitrary axis.

        :param torch.Tensor x: `(B, C, ...)`
        :param int axis: axis along which to compute sobel derivative along.
        :return: :class:`torch.Tensor` of shape `(B, C, ...)`
        """
        kernel = torch.tensor([[-1.0, 0.0, 1.0]], device=x.device, dtype=x.dtype)
        pad = 1

        # move target axis to last dim
        x_perm = x.transpose(axis, -1)
        orig_shape = x_perm.shape

        # flatten all leading dims
        x_flat = x_perm.reshape(-1, 1, orig_shape[-1])

        x_pad = F.pad(x_flat, (pad, pad), mode="reflect")
        out = F.conv1d(x_pad, kernel.unsqueeze(0))

        out = out.reshape(orig_shape)
        out = out.transpose(axis, -1)
        return out


class SharpnessIndex(Metric):
    r"""
    No-reference sharpness index metric for 2D images.

    Measures how sharp an image is, defined as

    .. math::

            \text{SI}(x) = -\log \Phi \left( \frac{\mathbb{E}_{\omega} \{ \text{TV}(\omega * x)\} - \text{TV}(x)  }{\sqrt{\mathbb{V}_{\omega} \{ \text{TV}(\omega * x) \} } } \right)


    where :math:`\Phi` is the CDF of a standard Gaussian distribution, :math:`\text{TV}` is the total variation,
    and :math:`\omega \sim \mathcal{N}(0, I)` is a Gaussian white noise distribution.

    Higher values indicate sharper images.

    The metric is used to introduced by :cite:t:`blanchet2012sharpness`.
    We use the fast implementation presented by :cite:t:`leclaire2015sharpness`.

    Adapted from MATLAB implementation in https://helios2.mi.parisdescartes.fr/~moisan/sharpness/.


    Default mode computing the periodic component and dequantizing should be used, unless you want to work on very
    specific images that are naturally periodic or not quantized (see :cite:t:`leclaire2015sharpness`).

    :param bool periodic_component: if `True` (default), compute the periodic component of the image before computing the metric.
    :param bool dequantize: if `True` (default), perform image dequantization by (1/2, 1/2) translation in Fourier domain before computing the metric.
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2`` normalizes by :math:`\ell_2` spatial norm, ``min_max`` normalizes by min and max of each input.
    :param bool check_input_range: if True, ``pyiqa`` will raise error if inputs aren't in the appropriate range ``[0, 1]``.
    :param int, tuple[int], None center_crop: If not `None` (default), center crop the tensor(s) before computing the metrics.
        If an `int` is provided, the cropping is applied equally on all spatial dimensions (by default, all dimensions except the first two).
        If `tuple` of `int`, cropping is performed over the last `len(center_crop)` dimensions. If positive values are provided, a standard center crop is applied.
        If negative (or zero) values are passed, cropping will be done by removing `center_crop` pixels from the borders (useful when tensors vary in size across the dataset).

    |sep|

    :Example:

    >>> from deepinv.loss.metric import SharpnessIndex
    >>> m = SharpnessIndex()
    >>> x_net = torch.randn(2, 3, 16, 16)  # batch of 2 RGB images
    >>> m(x_net).shape
    torch.Size([2])

    """

    def __init__(
        self, periodic_component: bool = True, dequantize: bool = True, **kwargs
    ) -> torch.Tensor:
        super().__init__(**kwargs)
        self.lower_better = False
        self.periodic_component = periodic_component
        self.dequantize = dequantize

        if not self.periodic_component and not self.dequantize:
            raise ValueError(
                "At least one of periodic_component or dequantize must be True."
            )

    def metric(self, x_net, *args, **kwargs):
        """
        Compute sharpness index metric for a batch of images.

        :param x_net: (B, C, H, W) input tensors with C=1 or 3 channels.
        :return: (B,) tensor of sharpness index values for each image in the batch
        """
        if len(x_net.shape) != 4:
            raise ValueError(
                "Sharpness index metric only supports 2D images of size (B, C, H, W)."
            )

        B, C, H, W = x_net.shape

        # preprocessing modes
        if self.periodic_component:
            x_net = self.per_decomp(x_net)
        if self.dequantize:
            x_net = self.dequant(x_net)

        gx = torch.roll(x_net, shifts=-1, dims=3) - x_net  # (B,C,H,W)
        gy = torch.roll(x_net, shifts=-1, dims=2) - x_net

        tv = (gx.abs() + gy.abs()).sum(dim=(2, 3))  # (B,C)

        fu = torch.fft.fft2(x_net)  # (B,C,H,W) complex

        # frequency grids
        p = torch.arange(W, device=x_net.device).reshape(1, 1, 1, W) * (
            2 * torch.pi / W
        )
        q = torch.arange(H, device=x_net.device).reshape(1, 1, H, 1) * (
            2 * torch.pi / H
        )

        # fgx2 = real(4 * fu * sin(P/2) * conj(fu))
        sin_p = torch.sin(p / 2)
        sin_q = torch.sin(q / 2)

        fgx2 = fu * sin_p
        fgx2 = 4 * (
            fgx2.real**2 + fgx2.imag**2
        )  # |4*fu*sin|^2 but matches MATLAB’s real(4*z*conj(z))

        fgy2 = fu * sin_q
        fgy2 = 4 * (fgy2.real**2 + fgy2.imag**2)

        # sums
        fgxx2 = fgx2.pow(2).sum(dim=(2, 3))  # (B,C)
        fgyy2 = fgy2.pow(2).sum(dim=(2, 3))
        fgxy2 = (fgx2 * fgy2).sum(dim=(2, 3))

        # simplified variance
        axx = (gx * gx).sum(dim=(2, 3))  # (B,C)
        ayy = (gy * gy).sum(dim=(2, 3))
        axy = torch.sqrt(axx * ayy)

        vara = torch.zeros_like(axx)

        mask = axx > 0
        vara = vara + torch.where(mask, fgxx2 / axx.clamp(min=1e-12), 0.0)

        mask = ayy > 0
        vara = vara + torch.where(mask, fgyy2 / ayy.clamp(min=1e-12), 0.0)

        mask = axy > 0
        vara = vara + torch.where(mask, 2 * fgxy2 / axy.clamp(min=1e-12), 0.0)

        vara = vara / (torch.pi * W * H)

        scale = math.sqrt(2 * W * H / torch.pi)
        t = ((torch.sqrt(axx) + torch.sqrt(ayy)) * scale - tv) / torch.sqrt(
            vara.clamp(min=1e-12)
        )

        s = torch.zeros_like(t)
        positive = vara > 0
        ts = t[positive] / math.sqrt(2)
        s_pos = -self.logerfc(ts) / math.log(10) + math.log10(2)
        s[positive] = s_pos
        return s.mean(dim=1)  # (B,)

    @staticmethod
    def per_decomp(u: torch.Tensor) -> torch.Tensor:
        r"""
        Periodic + smooth decomposition of a 2D image.

        Adapted from MATLAB implementation in https://helios2.mi.parisdescartes.fr/~moisan/sharpness/.

        :param torch.Tensor u: (B, C, H, W) tensor
        :return: p: periodic component minus smooth component (B, C, H, W)
        """
        B, C, H, W = u.shape
        u = u.double()

        v = torch.zeros_like(u)

        # temp differences for broadcasting
        u_top = u[..., 0, :]  # (B,C,W)
        u_bottom = u[..., H - 1, :]
        u_left = u[..., :, 0]  # (B,C,H)
        u_right = u[..., :, W - 1]

        v[..., 0, :] += u_top - u_bottom

        v[..., H - 1, :] -= u_top - u_bottom

        v[..., :, 0] += u_left - u_right

        v[..., :, W - 1] -= u_left - u_right

        # frequency grids (fx, fy)
        X = torch.arange(W, dtype=torch.float64, device=u.device).reshape(1, 1, 1, W)
        Y = torch.arange(H, dtype=torch.float64, device=u.device).reshape(1, 1, H, 1)

        fx = torch.cos(2 * torch.pi * (X) / W)  # (1,1,1,W) broadcasted
        fy = torch.cos(2 * torch.pi * (Y) / H)  # (1,1,H,1)

        # denominator = 2 - fx - fy
        denom = 2.0 - fx - fy

        denom[..., 0, 0] = 2.0

        # compute smooth part: s = real(ifft2( fft2(v) * 0.5 ./ denom ))
        fv = torch.fft.fft2(v)
        s = torch.fft.ifft2(fv * (0.5 / denom))
        s = s.real

        # periodic part
        p = u - s
        return p

    @staticmethod
    def dequant(u: torch.Tensor) -> torch.Tensor:
        r"""
        Image dequantization via (1/2, 1/2) translation in Fourier domain.

        Adapted from MATLAB implementation in https://helios2.mi.parisdescartes.fr/~moisan/sharpness/.

        :param torch.Tensor u: (B, C, H, W) tensor
        :return: (:class:torch.Tensor) dequantized image (B, C, H, W)
        """
        B, C, H, W = u.shape
        u = u.double()

        # Compute mx, my exactly as in MATLAB
        mx = W // 2
        my = H // 2

        # Build Tx and Ty (complex exponential phase shift)

        # index arrays
        x = torch.arange(mx, mx + W, device=u.device)
        y = torch.arange(my, my + H, device=u.device)

        x_mod = (x % W) - mx  # (W,)
        y_mod = (y % H) - my  # (H,)

        Tx = torch.exp(-1j * math.pi / W * x_mod)  # (W,) complex
        Ty = torch.exp(-1j * math.pi / H * y_mod)  # (H,) complex

        # Outer product Ty' * Tx → shape (H, W)
        shift = Ty[:, None] * Tx[None, :]  # (H, W)

        # Apply Fourier-domain phase shift
        fu = torch.fft.fft2(u)
        fv = fu * shift  # broadcasting over (B,C)
        v = torch.fft.ifft2(fv).real
        return v

    @staticmethod
    def logerfc(x: torch.Tensor) -> torch.Tensor:
        r"""
        Compute `log(erfc(x))` with asymptotic expansion for large `x`.

        Adapted from MATLAB implementation in https://helios2.mi.parisdescartes.fr/~moisan/sharpness/.

        :param torch.Tensor x: `(B, C, H, W)` tensor
        :return: `(B,)` tensor of logarithmic value of `x`
        """

        x = x.double()
        y = torch.empty_like(x)

        # mask for large x (asymptotic approximation)
        ind = x > 20

        # if x > 20  → use asymptotic expansion
        if ind.any():
            X = x[ind]
            z = X.pow(-2)
            s = torch.ones_like(X)

            # MATLAB loop: for k = 8:-1:1
            for k in range(8, 0, -1):
                s = 1 - (k - 0.5) * z * s

            y[ind] = -0.5 * math.log(math.pi) - X**2 + torch.log(s / X)

        # if x ≤ 20  → directly log(erfc(x))
        if (~ind).any():
            y[~ind] = torch.log(torch.erfc(x[~ind]))

        return y
