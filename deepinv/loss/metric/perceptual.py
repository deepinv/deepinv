from __future__ import annotations
import torch
import sys, io
import math
import torch
import torch.nn.functional as F
import torch.nn.functional as F

from deepinv.loss.metric.metric import Metric, import_pyiqa
from deepinv.physics.functional.convolution import conv2d

import io, requests, math


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

    Calculates the no-reference metric :math:`\text{NIQE}(\hat{x})` where :math:`\hat{x}=\inverse{y}`.

    This metric was introduced by :cite:t:`saad2012blind`, and
    relies on a natural scene statistics model of video DCT coefficients, as well as a temporal model of motion coherency.

    Lower values indicate better perceptual quality.

    It is a no-reference image quality metric that estimates the quality of images.

    .. note::

        By default, no reduction is performed in the batch dimension.


    :Example:

    ::

        from deepinv.utils import load_example
        from deepinv.loss.metric import NIQE
        m = NIQE()
        (...)
        x_net = load_example("celeba_example.jpg", img_size=128)
        m(x_net)
        tensor([...])

    :param bool round_tensor: whether to round the input. The original NIQE implementation used rounding and requires input to be range [0, 255]. Do not set round_tensor if incoming tensors will be in [0,1] style ranges. Defaults to False.
    :param torch.device, str device: device to use for the metric computation. Default: 'cpu'.
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2`` normalizes by :math:`\ell_2` spatial norm, ``min_max`` normalizes by min and max of each input.
    :param bool check_input_range: if True, ``pyiqa`` will raise error if inputs aren't in the appropriate range ``[0, 1]``.
    :param int, tuple[int], None center_crop: If not `None` (default), center crop the tensor(s) before computing the metrics.
        If an `int` is provided, the cropping is applied equally on all spatial dimensions (by default, all dimensions except the first two).
        If `tuple` of `int`, cropping is performed over the last `len(center_crop)` dimensions. If positive values are provided, a standard center crop is applied.
        If negative (or zero) values are passed, cropping will be done by removing `center_crop` pixels from the borders (useful when tensors vary in size across the dataset).
    """

    def __init__(
        self,
        round_tensor: bool = False,
        device: str | torch.device = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.round = round_tensor
        self.lower_better = True
        self.patch_size = 96
        self.device = device
        self.n_scales = 2
        self.patch_overlap = 0

        # resp = requests.get(
        #     "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/niqe_modelparameters.mat",
        #     timeout=2.5,
        # )
        try:
            from scipy.io import loadmat
        except:  # pragma: no cover
            raise ImportError("NIQE requires scipy. Please install it")
        resp = requests.get(
            "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/niqe_matlab_params.mat",
            timeout=2.5,
        )
        resp.raise_for_status()

        params = loadmat(io.BytesIO(resp.content))

        self.mu_p = (
            torch.from_numpy(params["mu_prisparam"])
            .to(dtype=torch.float32, device=device)
            .squeeze(0)
        )

        self.cov_p = torch.from_numpy(params["cov_prisparam"]).to(
            dtype=torch.float32, device=device
        )

    def estimate_aggd_param(self, vecs: torch.Tensor, eps: float = 1e-12):
        v = vecs
        neg_mask = v < 0
        pos_mask = v > 0

        cnt_neg = neg_mask.sum(dim=1).clamp_min(1)
        cnt_pos = pos_mask.sum(dim=1).clamp_min(1)

        left_ms = ((v * v) * neg_mask).sum(dim=1) / cnt_neg
        right_ms = ((v * v) * pos_mask).sum(dim=1) / cnt_pos

        leftstd = left_ms.clamp_min(eps).sqrt()
        rightstd = right_ms.clamp_min(eps).sqrt()

        gammahat = leftstd / rightstd.clamp_min(eps)
        rhat = (v.abs().mean(dim=1) ** 2) / (v.pow(2).mean(dim=1).clamp_min(eps))

        gam = torch.arange(
            0.2, 10.0 + 1e-9, 0.001, device=v.device, dtype=v.dtype
        )  # (G,)
        r_gam = (self._gamma(2.0 / gam) ** 2) / (
            self._gamma(1.0 / gam) * self._gamma(3.0 / gam)
        )  # (G,)

        rhatnorm = (rhat * (gammahat**3 + 1.0) * (gammahat + 1.0)) / (
            (gammahat**2 + 1.0) ** 2
        ).clamp_min(eps)

        diff = (r_gam.unsqueeze(0) - rhatnorm.unsqueeze(1)).pow(2)
        idx = diff.argmin(dim=1)
        alpha = gam[idx]  # (M,)

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
        B, _, H, W = structdis.shape
        base = F.unfold(
            structdis, kernel_size=(k, k), stride=(stride, stride)
        )  # (B, k*k, L)
        BKL = base.shape
        L = BKL[-1]
        base = base.transpose(1, 2).contiguous().view(B * L, k * k)

        a0, bl0, br0 = self.estimate_aggd_param(base)
        feat_cols = [a0, (bl0 + br0) * 0.5]

        shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in shifts:
            shifted = torch.roll(structdis, shifts=(dr, dc), dims=(2, 3))
            pair = structdis * shifted
            U = F.unfold(pair, kernel_size=(k, k), stride=(stride, stride))
            U = U.transpose(1, 2).contiguous().view(B * L, k * k)

            a, bl, br = self.estimate_aggd_param(U)
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
            structdis = (x_net - mu) / (sigma + 1)
            k = max(1, self.patch_size // scale)
            ov = self.patch_overlap // scale
            strd = max(1, k - ov)

            feats = self._patch_features(structdis, k, strd)  # (B, L, 18)
            all_feats.append(feats)

            if scale < self.n_scales:
                newH = x_net.shape[2] // 2
                newW = x_net.shape[3] // 2
                x_net = F.interpolate(
                    x_net,
                    size=(newH, newW),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )

        X = torch.cat(all_feats, dim=2)  # (B, L, 36)
        mu_d = X.mean(dim=1)  # (B, 36)
        Xc = X - mu_d.unsqueeze(1)  # (B, L, 36)
        denom = (X.shape[1] - 1) if X.shape[1] > 1 else 1
        cov_d = torch.einsum("blf,blg->bfg", Xc, Xc) / denom  # (B, 36, 36)

        cov_p = self.cov_p.expand_as(cov_d)  # (B,36,36)
        mu_p = self.mu_p  # (36,)
        invcov = torch.linalg.pinv(
            0.5 * (cov_d.to(torch.float64) + cov_p.to(torch.float64))
        ).to(
            torch.float32
        )  # (B,36,36)
        diff = (mu_p.unsqueeze(0) - mu_d).unsqueeze(1)  # (B,1,36)
        score = torch.sqrt((diff @ invcov @ diff.transpose(1, 2)).squeeze())
        return score

    def _gen_gauss_kernel(self):
        # sigma per original code: 7/6, window size 7
        sigma = 7 / 6
        radius = 3
        ax = torch.arange(-radius, radius + 1, device=self.device, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
        kernel /= kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

    def _gamma(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(torch.lgamma(x))

    def metric(self, x_net: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """We base ourselves on the original Matlab code (available at http://live.ece.utexas.edu/research/quality/niqe_release.zip), but allow some exceptions:

        (i) Originally, the image was converted to float64. For efficiency & kernel reasons, we work in float32.
        (ii)

        """
        if x_net.ndim != 4:  # pragma: no cover
            raise RuntimeError(
                f"NIQE expects batched, 2D data, but got tensor with {x_net.ndim} dimensions (shape: {x_net.shape})"
            )

        _, C, H, W = x_net.shape

        if H < self.patch_size or W < self.patch_size:  # pragma: no cover
            raise RuntimeError(
                f"NIQE requires images to have height and width larger than or equal to its patch size {self.patch_size}, but got batch of shape {x_net.shape}"
            )

        if C == 3:
            luminance_weights = torch.tensor(
                [0.2126, 0.7152, 0.0722], dtype=x_net.dtype, device=x_net.device
            ).view(1, 3, 1, 1)
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
