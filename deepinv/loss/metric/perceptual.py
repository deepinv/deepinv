from __future__ import annotations
import math, sys, io
from pathlib import Path
import torch
import torch.nn.functional as F

from deepinv.loss.metric.metric import Metric
from deepinv.physics.functional.convolution import conv2d
from deepinv.physics.functional.imresize import imresize_matlab
from deepinv.models.utils import load_state_dict_from_url, get_weights_url
from deepinv.datasets.base import batch_as_dict


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
    :param bool check_input_range: if True, raise error if inputs aren't in the appropriate range ``[0, 1]``.
    :param int, tuple[int], None center_crop: If not `None` (default), center crop the tensor(s) before computing the metrics.
        If an `int` is provided, the cropping is applied equally on all spatial dimensions (by default, all dimensions except the first two).
        If `tuple` of `int`, cropping is performed over the last `len(center_crop)` dimensions. If positive values are provided, a standard center crop is applied.
        If negative (or zero) values are passed, cropping will be done by removing `center_crop` pixels from the borders (useful when tensors vary in size across the dataset).
    :param str, torch.device device: LPIPS net device.
    """

    def __init__(
        self,
        net_type: str = "alex",
        device: torch.device | str = None,
        check_input_range: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        from torchmetrics.functional.image.lpips import _lpips_update, _NoTrainLpips

        if net_type not in ["vgg", "vgg16", "alex", "squeeze"]:  # pragma: no cover
            raise ValueError(
                f"net_type must be one of (vgg, alex, squeeze), got {net_type}"
            )

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

        self.check_input_range = check_input_range

        self.lower_better = True

    def metric(
        self, x_net: torch.Tensor, x: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        if x_net.ndim != 4 or x.ndim != 4:
            raise ValueError(
                f"LPIPS metric requires 4D input (B, C, H, W), but got shapes {x_net.shape}, {x.shape}."
            )

        if not (x_net.shape[1] == x.shape[1] == 3):
            raise ValueError(
                f"LPIPS metric only supports 3-channel input, but got channels for x_net, x as {x_net.shape[1]}, {x.shape[1]}."
            )

        if self.check_input_range:
            min_val, max_val = torch.aminmax(torch.cat([x_net, x], dim=0))
            if not ((min_val >= 0.0) & (max_val <= 1.0)):
                raise ValueError(
                    "LPIPS metric requires x_net and x to be between 0 and 1. To supress this error, set check_input_range to False at lpips init."
                )

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

        ``dataset`` should yield a (C, H, W) ``Tensor`` or a dictionary with key ``"x"`` containing such image, where C=1 and C=3 are allowed. If C=3, RGB is assumed and will be converted
        to greyscale using 0.299*R + 0.587*G + 0.114*B.

        :param torch.utils.data.Dataset dataset: for each item, should yield a Tensor representing a
            distortion-free (pristine) image or a dictionary with key ``"x"`` containing such image. Should have finite __len__ and indexable __getitem__.
        :param float sharpness_threshold: only patches whose sharpness is at least
            ``sharpness_threshold`` of the per-image peak sharpness (measured from σ at scale 1) are kept.
        :param str save_path: Path to which weights are to be saved. Must have ``.pt`` extension. If not passed, weights are returned without saving.

        :return: (mu_prisparam, cov_prisparam) as self.dtype on self.device. Also updates self.mu_p, self.cov_p.
        """
        with torch.no_grad():
            device = self.device
            dtype = torch.float32
            kernel = self._gen_gauss_kernel().to(device=device, dtype=dtype)

            all_feats = []

            for i, batch in zip(range(len(dataset)), dataset, strict=False):
                batch = batch_as_dict(batch)
                x = batch["x"]

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
                        [0.29893602, 0.58704307, 0.11402090],
                        dtype=x.dtype,
                        device=x.device,
                    ).view(1, 3, 1, 1)
                    x = F.conv2d(x, luminance_weights)

                if self.round:
                    x = x.round()

                _, _, H, W = x.shape
                if H < self.patch_size or W < self.patch_size:
                    print(
                        f"Sample {i} / {len(dataset)}: Too small H or Width, not included for weight creation."
                    )
                    continue
                block_hnum = math.floor(H / self.patch_size)
                block_wnum = math.floor(W / self.patch_size)
                x = x[
                    :, :, : block_hnum * self.patch_size, : block_wnum * self.patch_size
                ]

                feats_scales = []
                sharpness = None

                x_scale = x
                for scale in range(1, self.n_scales + 1):
                    mu = conv2d(x_scale, kernel, "replicate")
                    mu_sq = mu * mu
                    sigma = torch.sqrt(
                        torch.abs(
                            conv2d(x_scale * x_scale, kernel, "replicate") - mu_sq
                        )
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

            prisparam = torch.cat(all_feats, dim=0).to(
                device=device, dtype=dtype
            )  # (N,36)

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


class BRISQUE(Metric):
    r"""
    Blind/Referenceless Image Spatial QUality Evaluator (BRISQUE) metric.

    Calculates the BRISQUE score :math:`\text{BRISQUE}(\hat{x})` where :math:`\hat{x}=\inverse{y}`.
    It is a no-reference image quality metric introduced by :footcite:t:`mittal2012no`,
    which quantifies how far an image departs from the natural scene statistics of
    pristine natural images. Lower is better, with scores roughly in :math:`[0, 100]`.

    BRISQUE works with images of 1 or 3 channels. If the image has 3 channels,
    it is assumed to be RGB and converted to relative luminance, then, at two scales, the mean
    subtracted contrast normalized (MSCN) coefficients

    .. math::

        \hat{x}_{ij} = \frac{x_{ij} - \mu_{ij}}{\sigma_{ij} + 1}

    are computed with a :math:`7\times 7` Gaussian window. A generalized Gaussian is fitted
    to the MSCN coefficients and asymmetric generalized Gaussians are fitted to their four
    neighbouring products, yielding 36 features which are mapped to a quality score by a
    support vector regressor pre-trained on the `LIVE IQA dataset <https://live.ece.utexas.edu/research/quality/subjective.htm>`_.

    This is a PyTorch translation of the implementation
    (https://github.com/dsoellinger/blind_image_quality_toolbox). The pre-trained support vector
    regressor is the one released with the original MATLAB implementation, and weights
    were downloaded from https://github.com/dsoellinger/blind_image_quality_toolbox/blob/master/%2Bbrisque/allmodel.

    .. note::

        The features and the regressor were fitted on images in the :math:`[0, 255]` range.
        Inputs are internally rescaled from ``[0, max_pixel]`` to :math:`[0, 255]`,
        so make sure ``max_pixel`` matches the intensity scale of your data.

    .. note::

        By default, no reduction is performed in the batch dimension.

    :param str, pathlib.Path, None weights_path: path to the support vector regressor weights.
        If ``'download'`` (default), the weights released with the original implementation are downloaded.
    :param float max_pixel: maximum pixel value of the input images, used to rescale them to
        the :math:`[0, 255]` range expected by the regressor. Default: 1.
    :param torch.device, str device: device on which the regressor weights are stored. Default: ``'cpu'``.
    :param torch.dtype dtype: dtype used for the feature computation (the regressor is always evaluated
        in ``float64``). Default: ``torch.float32``.
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2`` normalizes by :math:`\ell_2` spatial norm, ``min_max`` normalizes by min and max of each input.
    :param int, tuple[int], None center_crop: If not `None` (default), center crop the tensor(s) before computing the metrics.
        If an `int` is provided, the cropping is applied equally on all spatial dimensions (by default, all dimensions except the first two).
        If `tuple` of `int`, cropping is performed over the last `len(center_crop)` dimensions. If positive values are provided, a standard center crop is applied.
        If negative (or zero) values are passed, cropping will be done by removing `center_crop` pixels from the borders (useful when tensors vary in size across the dataset).

    |sep|

    :Example:

    >>> from deepinv.loss.metric import BRISQUE
    >>> m = BRISQUE()
    >>> x_net = torch.rand(2, 3, 32, 32)  # batch of 2 RGB images in [0, 1]
    >>> m(x_net).shape
    torch.Size([2])

    """

    # min and max of each of the 36 features, used to scale them to [-1, 1] before
    # feeding them to the support vector regressor (from the original implementation).
    feature_range = (
        (0.338, 10),
        (0.017204, 0.806612),
        (0.236, 1.642),
        (-0.123884, 0.20293),
        (0.000155, 0.712298),
        (0.001122, 0.470257),
        (0.244, 1.641),
        (-0.123586, 0.179083),
        (0.000152, 0.710456),
        (0.000975, 0.470984),
        (0.249, 1.555),
        (-0.135687, 0.100858),
        (0.000174, 0.684173),
        (0.000913, 0.534174),
        (0.258, 1.561),
        (-0.143408, 0.100486),
        (0.000179, 0.685696),
        (0.000888, 0.536508),
        (0.471, 3.264),
        (0.012809, 0.703171),
        (0.218, 1.046),
        (-0.094876, 0.187459),
        (1.5e-05, 0.442057),
        (0.001272, 0.40803),
        (0.222, 1.042),
        (-0.115772, 0.162604),
        (1.6e-05, 0.444362),
        (0.001374, 0.40243),
        (0.227, 0.996),
        (-0.117188, 0.098323),
        (3e-05, 0.531903),
        (0.001122, 0.369589),
        (0.228, 0.99),
        (-0.12243, 0.098658),
        (2.8e-05, 0.530092),
        (0.001118, 0.370399),
    )

    def __init__(
        self,
        weights_path: str | Path | None = "download",
        max_pixel: float = 1.0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lower_better = True
        self.max_pixel = max_pixel
        self.dtype = dtype

        if str(weights_path) == "download":
            params = load_state_dict_from_url(
                get_weights_url("demo", "brisque_weights.pt"),
                map_location=lambda storage, loc: storage,
                file_name="brisque_weights.pt",
                weights_only=True,
            )
        else:
            params = torch.load(weights_path, weights_only=True)

        # epsilon-SVR with RBF kernel, evaluated in float64 for reproducibility
        self.register_buffer(
            "support_vectors",
            params["support_vectors"].to(device=device, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "dual_coef",
            params["dual_coef"].to(device=device, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "svr_rho",
            params["rho"].to(device=device, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "svr_gamma",
            params["gamma"].to(device=device, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "feature_min",
            torch.tensor([r[0] for r in self.feature_range], device=device).to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "feature_max",
            torch.tensor([r[1] for r in self.feature_range], device=device).to(dtype),
            persistent=False,
        )

        # candidate shape parameters of the (asymmetric) generalized Gaussian and the
        # associated moment ratios, tabulated once and looked up by the fits below
        gam = 0.2 + 1e-3 * torch.arange(9801, device=device, dtype=torch.float64)
        self.register_buffer("gamma_grid", gam.to(dtype), persistent=False)
        self.register_buffer(
            "ggd_ratio",
            (
                self._gamma_fn(1.0 / gam)
                * self._gamma_fn(3.0 / gam)
                / self._gamma_fn(2.0 / gam) ** 2
            ).to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "aggd_ratio",
            (
                self._gamma_fn(2.0 / gam) ** 2
                / (self._gamma_fn(1.0 / gam) * self._gamma_fn(3.0 / gam))
            ).to(dtype),
            persistent=False,
        )

    def _gauss_kernel(self, device: torch.device) -> torch.Tensor:
        r"""Separable :math:`7\times 7` Gaussian window with :math:`\sigma=7/6`, as in the original implementation."""
        sigma = 7 / 6
        ax = torch.arange(-3, 4, device=device, dtype=self.dtype)
        k = torch.exp(-(ax**2) / (2 * sigma * sigma))
        k = k / k.sum()
        return torch.outer(k, k).view(1, 1, 7, 7)

    @staticmethod
    def _gamma_fn(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(torch.lgamma(x))

    def estimate_ggd_param(
        self, vecs: torch.Tensor, eps: float = 1e-12
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Fit a generalized Gaussian distribution to each row by moment matching.

        :param torch.Tensor vecs: `(B, N)` tensor of samples.
        :param float eps: stabilizer used in the denominators.
        :return: tuple of `(B,)` tensors with the shape parameter and the standard deviation.
        """
        sigma_sq = (vecs**2).mean(dim=1)
        mean_abs = vecs.abs().mean(dim=1)
        rho = sigma_sq / torch.clamp(mean_abs**2, min=eps)

        idx = (rho.unsqueeze(1) - self.ggd_ratio.unsqueeze(0)).abs().argmin(dim=1)
        return self.gamma_grid[idx], sigma_sq.sqrt()

    def estimate_aggd_param(
        self, vecs: torch.Tensor, eps: float = 1e-12
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Fit an asymmetric generalized Gaussian distribution to each row by moment matching.

        :param torch.Tensor vecs: `(B, N)` tensor of samples.
        :param float eps: stabilizer used in the denominators.
        :return: tuple of `(B,)` tensors with the shape parameter, the left and the right standard deviations.
        """
        neg, pos = vecs < 0, vecs > 0
        left_std = (
            (vecs**2 * neg).sum(dim=1) / torch.clamp(neg.sum(dim=1), min=1)
        ).sqrt()
        right_std = (
            (vecs**2 * pos).sum(dim=1) / torch.clamp(pos.sum(dim=1), min=1)
        ).sqrt()

        gamma_hat = left_std / torch.clamp(right_std, min=eps)
        rhat = vecs.abs().mean(dim=1) ** 2 / torch.clamp((vecs**2).mean(dim=1), min=eps)
        rhat_norm = (
            rhat
            * (gamma_hat**3 + 1)
            * (gamma_hat + 1)
            / torch.clamp((gamma_hat**2 + 1) ** 2, min=eps)
        )

        idx = ((self.aggd_ratio.unsqueeze(0) - rhat_norm.unsqueeze(1)) ** 2).argmin(
            dim=1
        )
        return self.gamma_grid[idx], left_std, right_std

    def features(self, x_net: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the 36 natural scene statistics features of BRISQUE.

        :param torch.Tensor x_net: `(B, 1, H, W)` single-channel images in the `[0, 255]` range.
        :return: `(B, 36)` tensor of features.
        """
        B = x_net.shape[0]
        kernel = self._gauss_kernel(x_net.device)
        feats = []

        for scale in range(2):
            # local mean and standard deviation, zero padded as in the original implementation
            mu = F.conv2d(F.pad(x_net, (3, 3, 3, 3)), kernel)
            sigma = F.conv2d(F.pad(x_net * x_net, (3, 3, 3, 3)), kernel)
            sigma = (sigma - mu * mu).abs().sqrt()
            structdis = (x_net - mu) / (sigma + 1)

            alpha, overallstd = self.estimate_ggd_param(structdis.reshape(B, -1))
            feats += [alpha, overallstd**2]

            for dr, dc in ((0, 1), (1, 0), (1, 1), (-1, 1)):
                shifted = torch.roll(structdis, shifts=(dr, dc), dims=(2, 3))
                pair = (structdis * shifted).reshape(B, -1)
                alpha, left_std, right_std = self.estimate_aggd_param(pair)
                const = (self._gamma_fn(1 / alpha) / self._gamma_fn(3 / alpha)).sqrt()
                mean_param = (
                    (right_std - left_std)
                    * (self._gamma_fn(2 / alpha) / self._gamma_fn(1 / alpha))
                    * const
                )
                feats += [alpha, mean_param, left_std**2, right_std**2]

            if scale == 0:
                # nearest-neighbour downsampling by a factor 2, as in the original implementation
                x_net = x_net[:, :, ::2, ::2]

        return torch.stack(feats, dim=1)

    def metric(self, x_net: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Compute the BRISQUE score for a batch of images.

        :param torch.Tensor x_net: `(B, C, H, W)` input tensors with C=1 or 3 channels.
        :return: `(B,)` tensor of BRISQUE scores.
        """
        if x_net.ndim != 4:
            raise ValueError(
                f"BRISQUE expects batched, 2D data of shape (B, C, H, W), but got tensor with {x_net.ndim} dimensions (shape: {x_net.shape})."
            )

        C = x_net.shape[1]
        if C not in (1, 3):
            raise ValueError(
                f"BRISQUE only operates on 1- or 3-channel images, but got {C}-channel input."
            )

        x_net = x_net.to(self.dtype)

        if C == 3:  # RGB to relative luminance
            x_net = (
                0.2989 * x_net[:, [0]] + 0.5870 * x_net[:, [1]] + 0.1140 * x_net[:, [2]]
            )

        # the features were fitted on images in the [0, 255] range
        x_net = x_net * (255.0 / self.max_pixel)

        feats = self.features(x_net)
        scaled = -1 + 2 * (feats - self.feature_min) / (
            self.feature_max - self.feature_min
        )

        # epsilon-SVR decision function with RBF kernel
        scaled = scaled.to(torch.float64)
        sq_dist = (
            torch.cdist(
                scaled,
                self.support_vectors,
                compute_mode="donot_use_mm_for_euclid_dist",
            )
            ** 2
        )
        return (
            torch.exp(-self.svr_gamma * sq_dist) @ self.dual_coef - self.svr_rho
        ).float()


class _DepthwiseSeparable(torch.nn.Module):
    r"""Depthwise separable block of MobileNetV1, matching ``tf.keras.applications.mobilenet``,
    see https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNet.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.stride = stride
        self.dw = torch.nn.Conv2d(
            in_channels,
            in_channels,
            3,
            stride=stride,
            padding=1 if stride == 1 else 0,
            groups=in_channels,
            bias=False,
        )
        self.dw_bn = torch.nn.BatchNorm2d(in_channels, eps=1e-3)
        self.pw = torch.nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pw_bn = torch.nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 2:
            # Keras pads asymmetrically before a 'valid' strided convolution
            x = F.pad(x, (0, 1, 0, 1))
        x = F.relu6(self.dw_bn(self.dw(x)))
        return F.relu6(self.pw_bn(self.pw(x)))


class _MobileNetV1(torch.nn.Module):
    r"""
    MobileNetV1 (width multiplier 1) with a NIMA classification head :cite:p:`howard2017mobilenets`:

    Layer-for-layer equivalent of ``tf.keras.applications.mobilenet.MobileNet`` with
    ``include_top=False, pooling='avg'`` followed by a dense softmax layer, so that the
    Keras weights released by idealo can be loaded directly.

    :param int n_classes: number of score bins of the head. Default: 10.
    """

    # (out_channels, stride) of the 13 depthwise separable blocks
    blocks_config = (
        (64, 1),
        (128, 2),
        (128, 1),
        (256, 2),
        (256, 1),
        (512, 2),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (1024, 2),
        (1024, 1),
    )

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(32, eps=1e-3)

        in_channels = 32
        blocks = []
        for out_channels, stride in self.blocks_config:
            blocks.append(_DepthwiseSeparable(in_channels, out_channels, stride))
            in_channels = out_channels
        self.blocks = torch.nn.ModuleList(blocks)

        # the dropout of the original model is inactive at evaluation time
        self.head = torch.nn.Linear(in_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu6(self.bn1(self.conv1(F.pad(x, (0, 1, 0, 1)))))
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=(2, 3))  # global average pooling
        return self.head(x).softmax(dim=1)


class NIMA(Metric):
    r"""
    Neural Image Assessment (NIMA) metric.

    Calculates the NIMA score :math:`\text{NIMA}(\hat{x})` where :math:`\hat{x}=\inverse{y}`.
    It is a no-reference image quality metric introduced by :footcite:t:`talebi2018nima`,
    which predicts the distribution of human opinion scores an image would receive.

    A convolutional network outputs a probability :math:`p_i` for each of the 10 score bins,
    and the metric returns the mean opinion score

    .. math::

        \text{NIMA}(\hat{x}) = \sum_{i=1}^{10} i \, p_i \in [1, 10],

    where higher is better. Use :func:`distribution <deepinv.loss.metric.NIMA.distribution>`
    to obtain the full predicted distribution, whose spread indicates how much raters would disagree.

    Two pre-trained heads are available, selected with ``variant``:

    - ``'aesthetic'`` (default), trained on the AVA dataset :cite:p:`murray2012ava`, which rates the aesthetic appeal of an image;
    - ``'technical'``, trained on the TID2013 dataset :cite:p:`ponomarenko2015image`, which rates the amount of distortion in an image

    This is adapted from the ``image-quality-assessment`` implementation in
    (https://github.com/idealo/image-quality-assessment), which we gratefully acknowledge.
    The MobileNet backbone and both heads use their released weights, converted to PyTorch.


    .. warning::

        The network expects :math:`224\times 224` inputs, so images are bilinearly resized before
        being scored, without preserving the aspect ratio, as in the original implementation.

    .. note::

        Single-channel images are replicated over three channels, as the network expects RGB input.

    .. note::

        By default, no reduction is performed in the batch dimension.

    :param str variant: which pre-trained head to use, either `'aesthetic'` or `'technical'`. Default: ``'aesthetic'``.
    :param str, pathlib.Path, None weights_path: path to the network weights. If ``'download'`` (default),
        the weights of the chosen ``variant`` are downloaded.
    :param float max_pixel: maximum pixel value of the input images, used to rescale them to the
        :math:`[-1, 1]` range expected by the network. Default: 1.
    :param torch.device, str device: device on which the network is stored. Default: ``'cpu'``.
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2`` normalizes by :math:`\ell_2` spatial norm, ``min_max`` normalizes by min and max of each input.
    :param int, tuple[int], None center_crop: If not `None` (default), center crop the tensor(s) before computing the metrics.
        If an `int` is provided, the cropping is applied equally on all spatial dimensions (by default, all dimensions except the first two).
        If `tuple` of `int`, cropping is performed over the last `len(center_crop)` dimensions. If positive values are provided, a standard center crop is applied.
        If negative (or zero) values are passed, cropping will be done by removing `center_crop` pixels from the borders (useful when tensors vary in size across the dataset).

    |sep|

    :Example:

    >>> from deepinv.loss.metric import NIMA
    >>> m = NIMA()
    >>> x_net = torch.rand(2, 3, 64, 64)  # batch of 2 RGB images in [0, 1]
    >>> m(x_net).shape
    torch.Size([2])

    """

    def __init__(
        self,
        variant: str = "aesthetic",
        weights_path: str | Path | None = "download",
        max_pixel: float = 1.0,
        device: str | torch.device = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lower_better = False
        self.max_pixel = max_pixel

        variants = ("aesthetic", "technical")
        if variant not in variants:
            raise ValueError(f"variant must be one of {variants}, but got {variant}.")
        self.variant = variant

        if weights_path == "download":
            file_name = f"nima_mobilenet_{variant}_weights.pt"
            params = load_state_dict_from_url(
                get_weights_url("demo", file_name),
                map_location=lambda storage, loc: storage,
                file_name=file_name,
                weights_only=True,
            )
        else:
            params = torch.load(weights_path, weights_only=True)

        self.net = _MobileNetV1()
        self.net.load_state_dict(params)
        self.net.eval().requires_grad_(False)
        self.net.to(device)

        self.register_buffer(
            "score_bins",
            torch.arange(1, 11, device=device, dtype=torch.float32),
            persistent=False,
        )

    def distribution(self, x_net: torch.Tensor) -> torch.Tensor:
        r"""
        Predict the distribution of human opinion scores of a batch of images.

        Resizes to the network input size and rescale to :math:`[-1, 1]`.

        :param torch.Tensor x_net: `(B, C, H, W)` input tensors with C=1 or 3 channels.
        :return: `(B, 10)` tensor of probabilities, where entry :math:`i` is the predicted
            probability that a rater would give the image a score of :math:`i+1`.
        """
        if x_net.ndim != 4:
            raise ValueError(
                f"NIMA expects batched, 2D data of shape (B, C, H, W), but got tensor with {x_net.ndim} dimensions (shape: {x_net.shape})."
            )

        C = x_net.shape[1]
        if C not in (1, 3):
            raise ValueError(
                f"NIMA only operates on 1- or 3-channel images, but got {C}-channel input."
            )
        if C == 1:
            x_net = x_net.expand(-1, 3, -1, -1)

        if x_net.shape[-2:] != (224, 224):
            x_net = F.interpolate(
                x_net, size=(224, 224), mode="bilinear", align_corners=False
            )

        x_net = 2 * x_net / self.max_pixel - 1
        return self.net(x_net)

    def metric(self, x_net: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Compute the mean opinion score of a batch of images.

        :param torch.Tensor x_net: `(B, C, H, W)` input tensors with C=1 or 3 channels.
        :return: `(B,)` tensor of NIMA scores, between 1 and 10, higher is better.
        """
        return self.distribution(x_net) @ self.score_bins


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
        if x_net.shape[1] not in [1, 3]:  # pragma: no cover
            raise ValueError("Input must have 1 or 3 channels.")

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

    def metric(self, x_net: torch.Tensor, *args, **kwargs) -> torch.Tensor:
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
