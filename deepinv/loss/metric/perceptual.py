from __future__ import annotations
import torch
import torch.nn.functional as F
from scipy.io import loadmat

from deepinv.loss.metric.metric import Metric, import_pyiqa
from deepinv.physics.functional.convolution import conv2d
from deepinv.datasets.utils import download_archive

import io
import requests


class LPIPS(Metric):
    r"""
    Learned Perceptual Image Patch Similarity (LPIPS) metric.

    Calculates the LPIPS :math:`\text{LPIPS}(\hat{x},x)` where :math:`\hat{x}=\inverse{y}`.

    Computes the perceptual similarity between two images, based on a pre-trained deep neural network.
    Uses implementation from `pyiqa <https://pypi.org/project/pyiqa/>`_.

    .. note::

        By default, no reduction is performed in the batch dimension.

    :Example:

    >>> from deepinv.utils.demo import load_example
    >>> from deepinv.loss.metric import LPIPS
    >>> m = LPIPS() # doctest: +IGNORE_RESULT
    >>> x = load_example("celeba_example.jpg", img_size=128)
    >>> x_net = x - 0.01
    >>> m(x_net, x) # doctest: +ELLIPSIS
    tensor([...])

    :param str device: device to use for the metric computation. Default: 'cpu'.
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2``normalizes by L2 spatial norm, ``min_max`` normalizes by min and max of each input.
    :param bool check_input_range: if True, ``pyiqa`` will raise error if inputs aren't in the appropriate range ``[0, 1]``.
    :param bool as_loss: if True, returns LPIPS as a loss. Default: False.
    """

    def __init__(self, device="cpu", check_input_range=False, as_loss=False, **kwargs):
        super().__init__(**kwargs)
        pyiqa = import_pyiqa()
        self.lpips = pyiqa.create_metric(
            "lpips", check_input_range=check_input_range, device=device, as_loss=as_loss
        ).to(device)
        self.lower_better = self.lpips.lower_better

    def metric(self, x_net, x, *args, **kwargs):
        return self.lpips(x_net, x).squeeze(-1)


class NIQE(Metric):
    r"""
    Natural Image Quality Evaluator (NIQE) metric.

    Calculates the NIQE :math:`\text{NIQE}(\hat{x})` where :math:`\hat{x}=\inverse{y}`.
    It is a no-reference image quality metric that estimates the quality of images.

    .. note::

        By default, no reduction is performed in the batch dimension.


    :Example:

    >>> from deepinv.utils.demo import load_example
    >>> from deepinv.loss.metric import NIQE
    >>> m = NIQE() # doctest: +IGNORE_RESULT
    (...)
    >>> x_net = load_example("celeba_example.jpg", img_size=128)
    >>> m(x_net) # doctest: +ELLIPSIS
    tensor([...])

    :param bool round_tensor: whether to round the input. The original NIQE implementation used rounding and requires input to be range [0, 255]. Do not set round_tensor if incoming tensors will be in [0,1] style ranges. Defaults to False.
    :param torch.device, str device: device to use for the metric computation. Default: 'cpu'.
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2``normalizes by L2 spatial norm, ``min_max`` normalizes by min and max of each input.
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

        # resp = requests.get(
        #     "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/niqe_modelparameters.mat",
        #     timeout=2.5,
        # )

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
            ov = getattr(self, "patch_overlap", 0) // scale
            strd = max(1, k - ov)

            feats = self._patch_features(structdis, k, strd)  # (B, L, 18)
            all_feats.append(feats)

            if scale < self.n_scales:
                newH = x_net.shape[2] // 2
                newW = x_net.shape[3] // 2
                x_net = F.interpolate(
                    x_net, size=(newH, newW), mode="bilinear", align_corners=False
                )

        X = torch.cat(all_feats, dim=2)  # (B, L, 36)
        mu_d = X.mean(dim=1)  # (B, 36)
        Xc = X - mu_d.unsqueeze(1)  # (B, L, 36)
        denom = (X.shape[1] - 1) if X.shape[1] > 1 else 1
        cov_d = torch.einsum("blf,blg->bfg", Xc, Xc) / denom  # (B, 36, 36)

        cov_p = self.cov_p.expand_as(cov_d)  # (B,36,36)
        mu_p = self.mu_p  # (36,)
        invcov = torch.linalg.pinv(0.5 * (cov_d + cov_p))  # (B,36,36)
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

        b, c, h, w = x_net.shape
        if c == 3:
            luminance_weights = torch.tensor(
                [0.2126, 0.7152, 0.0722], dtype=x_net.dtype, device=x_net.device
            ).view(1, 3, 1, 1)
            x_net = F.conv2d(x_net, luminance_weights)
        if x_net.shape[1] != 1:
            raise RuntimeError(
                f"NIQE only operates on single channel images. 3 channel (RGB) gets converted to relative luminance, but got {c}-channel input"
            )
        if self.round:
            x_net = x_net.round()

        n = self.niqe(x_net).float()
        return n.unsqueeze(0) if n.dim() == 0 else n
