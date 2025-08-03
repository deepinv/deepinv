import torch
import torch.nn as nn
from .base import Denoiser
from typing import Union

try:
    import ptwt
    import pywt
except:  # pragma: no cover
    ptwt = ImportError("The ptwt package is not installed.")  # pragma: no cover
    # No need to pywt, which is a dependency of ptwt


class WaveletDenoiser(Denoiser):
    r"""
    Orthogonal Wavelet denoising with the :math:`\ell_1` norm.


    This denoiser is defined as the solution to the optimization problem:

    .. math::

        \underset{x}{\arg\min} \;  \|x-y\|^2 + \gamma \|\Psi x\|_n


    where :math:`\Psi` is an orthonormal wavelet transform, :math:`\lambda>0` is a hyperparameter, and where
    :math:`\|\cdot\|_n` is either the :math:`\ell_1` norm (``non_linearity="soft"``) or
    the :math:`\ell_0` norm (``non_linearity="hard"``). A variant of the :math:`\ell_0` norm is also available
    (``non_linearity="topk"``), where the thresholding is done by keeping the :math:`k` largest coefficients
    in each wavelet subband and setting the others to zero.

    The solution is available in closed-form, thus the denoiser is cheap to compute.

    .. warning::

        This model requires Pytorch Wavelets (``ptwt``) to be installed. It can be installed with
        ``pip install ptwt``.

    :param int level: decomposition level of the wavelet transform
    :param str wv: mother wavelet (follows the `PyWavelets convention
        <https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html>`_) (default: "db8")
    :param str non_linearity: ``"soft"``, ``"hard"`` or ``"topk"`` thresholding (default: ``"soft"``).
        If ``"topk"``, only the top-k wavelet coefficients are kept.
    :param str mode: padding mode for the wavelet transform (default: "zero").
    :param int wvdim: dimension of the wavelet transform (either 2 or 3) (default: 2).
    :param str device: cpu or gpu

    """

    def __init__(
        self,
        level: int = 3,
        wv: str = "db8",
        device: torch.device = "cpu",
        non_linearity: str = "soft",
        mode: str = "zero",
        wvdim: int = 2,
    ):
        if isinstance(ptwt, ImportError):
            raise ImportError(
                "pytorch_wavelets is needed to use the WaveletDenoiser class. "
                "It should be installed with `pip install ptwt`."
            ) from ptwt
        super().__init__()
        self.level = level
        self.wv = wv
        self.device = device
        self.non_linearity = non_linearity
        self.dimension = wvdim
        self.mode = mode

    def dwt(self, x):
        r"""
        Applies the wavelet decomposition.
        """
        if self.dimension == 2:
            dec = ptwt.wavedec2(
                x, pywt.Wavelet(self.wv), mode=self.mode, level=self.level
            )
        elif self.dimension == 3:
            dec = ptwt.wavedec3(
                x, pywt.Wavelet(self.wv), mode=self.mode, level=self.level
            )
        dec = [list(t) if isinstance(t, tuple) else t for t in dec]
        return dec

    def flatten_coeffs(self, dec):
        r"""
        Flattens the wavelet coefficients and returns them in a single torch vector of shape (n_coeffs,).
        """
        if self.dimension == 2:
            flat = torch.hstack(
                [dec[0].flatten()]
                + [decl.flatten() for l in range(1, len(dec)) for decl in dec[l]]
            )
        elif self.dimension == 3:
            flat = torch.hstack(
                [dec[0].flatten()]
                + [dec[l][key].flatten() for l in range(1, len(dec)) for key in dec[l]]
            )
        return flat

    @staticmethod
    def psi(x, wavelet="db2", level=2, dimension=2, mode="zero"):
        r"""
        Returns a flattened list containing the wavelet coefficients.

        :param torch.Tensor x: input image.
        :param str wavelet: mother wavelet.
        :param int level: decomposition level.
        :param int dimension: dimension of the wavelet transform (either 2 or 3).
        """
        if dimension == 2:
            dec = ptwt.wavedec2(x, pywt.Wavelet(wavelet), mode=mode, level=level)
            dec = list(dec)
            vec = [decl.flatten(1, -1) for l in range(1, len(dec)) for decl in dec[l]]
        elif dimension == 3:
            dec = ptwt.wavedec3(x, pywt.Wavelet(wavelet), mode=mode, level=level)
            dec = list(dec)
            vec = [
                dec[l][key].flatten(1, -1) for l in range(1, len(dec)) for key in dec[l]
            ]
        return vec

    def iwt(self, coeffs):
        r"""
        Applies the wavelet recomposition.
        """

        coeffs = self._list_to_tuple(coeffs)
        if self.dimension == 2:
            rec = ptwt.waverec2(coeffs, pywt.Wavelet(self.wv))
        elif self.dimension == 3:
            rec = ptwt.waverec3(coeffs, pywt.Wavelet(self.wv))
        return rec

    def prox_l1(self, x, ths=0.1):
        r"""
        Soft thresholding of the wavelet coefficients.

        :param torch.Tensor x: wavelet coefficients.
        :param float, torch.Tensor ths: threshold.
        """
        ths = self._expand_ths_as(ths, x)
        return torch.maximum(torch.tensor(0.0), x - abs(ths)) + torch.minimum(
            torch.tensor(0.0), x + abs(ths)
        )

    @staticmethod
    def _expand_ths_as(
        ths: Union[float, torch.Tensor], x: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Expand the threshold to the same shape as the input tensor.
        """
        if isinstance(ths, (float, int)):
            return float(ths)
        elif isinstance(ths, torch.Tensor):
            ths = ths.squeeze()
            return ths.view(-1, *([1] * (x.ndim - 1))).to(x.device)
        else:
            raise ValueError(f"Invalid threshold type: {type(ths)}")

    def prox_l0(
        self, x: torch.Tensor, ths: Union[float, torch.Tensor] = 0.1
    ) -> torch.Tensor:
        r"""
        Hard thresholding of the wavelet coefficients.

        :param torch.Tensor x: wavelet coefficients of shape (B, C, H, W) or (B, C, D, H, W).
        :param float, torch.Tensor ths: threshold of shape (B,) or scalar. If scalar, same threshold is used for all elements in batch.
        """

        out = x.clone()
        ths = self._expand_ths_as(ths, out)
        out[out.abs() < ths] = 0
        return out

    def hard_threshold_topk(self, x, ths=0.1):
        r"""
        Hard thresholding of the wavelet coefficients by keeping only the top-k coefficients and setting the others to
        0.

        :param torch.Tensor x: wavelet coefficients.
        :param float, int ths: top k coefficients to keep. If ``float``, it is interpreted as a proportion of the total
            number of coefficients. If ``int``, it is interpreted as the number of coefficients to keep.
        """
        if isinstance(ths, (float, int)):
            k = int(ths * x.shape[-3] * x.shape[-2] * x.shape[-1])
        elif isinstance(ths, torch.Tensor):
            k = ths.squeeze().view(-1).expand(x.size(0)).to(x.device, torch.int32)
        else:
            raise ValueError(
                f"Invalid threshold type: {type(ths)}. Expected float, int or torch.Tensor."
            )

        # Reshape arrays to 2D and initialize output to 0
        x_flat = x.reshape(x.shape[0], -1)
        out = torch.zeros_like(x_flat)

        # Convert the flattened indices to the original indices of x
        if isinstance(k, int):
            topk_indices_flat = torch.topk(abs(x_flat), k, dim=-1)[1]
            batch_indices = (
                torch.arange(x.shape[0], device=x.device).unsqueeze(1).expand(-1, k)
            )
            topk_indices = torch.stack([batch_indices, topk_indices_flat], dim=-1)

            # Set output's top-k elements to values from original x
            out[tuple(topk_indices.view(-1, 2).t())] = x_flat[
                tuple(topk_indices.view(-1, 2).t())
            ]
            return torch.reshape(out, x.shape)
        else:
            # For each batch, keep the top-k coefficients
            for i in range(x.shape[0]):
                topk_indices = torch.topk(abs(x_flat[i]), k[i].item(), dim=-1)[1]
                out[i, topk_indices] = x_flat[i, topk_indices]
            return torch.reshape(out, x.shape)

    def thresold_func(self, x, ths):
        r""" "
        Apply thresholding to the wavelet coefficients.
        """
        if self.non_linearity == "soft":
            y = self.prox_l1(x, ths)
        elif self.non_linearity == "hard":
            y = self.prox_l0(x, ths)
        elif self.non_linearity == "topk":
            y = self.hard_threshold_topk(x, ths)
        return y

    def thresold_2D(self, coeffs, ths):
        r"""
        Thresholds coefficients of the 2D wavelet transform.
        """
        for level in range(1, self.level + 1):
            ths_cur = self.reshape_ths(ths, level)
            for c in range(3):
                coeffs[level][c] = self.thresold_func(coeffs[level][c], ths_cur[c])
        return coeffs

    def threshold_3D(self, coeffs, ths):
        r"""
        Thresholds coefficients of the 3D wavelet transform.
        """
        for level in range(1, self.level + 1):
            ths_cur = self.reshape_ths(ths, level)
            for c, key in enumerate(["aad", "ada", "daa", "add", "dad", "dda", "ddd"]):
                coeffs[level][key] = self.thresold_func(coeffs[level][key], ths_cur[c])
        return coeffs

    def threshold_ND(self, coeffs, ths):
        r"""
        Apply thresholding to the wavelet coefficients of arbitrary dimension.
        """
        if self.dimension == 2:
            coeffs = self.thresold_2D(coeffs, ths)
        elif self.dimension == 3:
            coeffs = self.threshold_3D(coeffs, ths)
        else:
            raise ValueError("Only 2D and 3D wavelet transforms are supported")

        return coeffs

    def pad_input(self, x):
        r"""
        Pad the input to make it compatible with the wavelet transform.
        """
        if self.dimension == 2:
            h, w = x.size()[-2:]
            padding_bottom = h % 2
            padding_right = w % 2
            p = (padding_bottom, padding_right)
            x = torch.nn.functional.pad(
                x, (0, padding_right, 0, padding_bottom), mode="replicate"
            )
        elif self.dimension == 3:
            d, h, w = x.size()[-3:]
            padding_depth = d % 2
            padding_bottom = h % 2
            padding_right = w % 2
            p = (padding_depth, padding_bottom, padding_right)
            x = torch.nn.functional.pad(
                x,
                (0, padding_right, 0, padding_bottom, 0, padding_depth),
                mode="replicate",
            )
        return x, p

    def crop_output(self, x, padding):
        r"""
        Crop the output to make it compatible with the wavelet transform.
        """
        d, h, w = x.size()[-3:]
        if len(padding) == 2:
            out = x[..., : h - padding[0], : w - padding[1]]
        elif len(padding) == 3:
            out = x[..., : d - padding[0], : h - padding[1], : w - padding[2]]
        return out

    def reshape_ths(self, ths, level):
        r"""
        Reshape the thresholding parameter in the appropriate format, i.e. either:
         - a list of 3 elements, or
         - a tensor of 3 elements.

        Since the approximation coefficients are not thresholded, we do not need to provide a thresholding parameter,
        ths has shape (n_levels-1, 3).
        """

        # Return tensor of shape (B, n_level - 1, numel)
        numel = 3 if self.dimension == 2 else 7
        if not torch.is_tensor(ths):
            if isinstance(ths, (int, float)):
                ths_cur = [ths] * numel
            elif len(ths) == 1:
                ths_cur = [ths[0]] * numel
            else:
                ths_cur = ths[level]
                if len(ths_cur) == 1:
                    ths_cur = [ths_cur[0]] * numel
        else:
            if ths.ndim == 0 or ths.ndim == 1:  # a tensor of shape 0 or (B,)
                return self._reshape_ths_one_dim(ths, level)
            elif ths.ndim == 2:  # (B, n_levels-1)
                return self._reshape_ths_two_dim(ths, level)
            elif ths.ndim == 3:
                # (B, n_levels-1, numel) or (B, n_levels-1, 1)
                ths_cur = self._reshape_ths_three_dim(ths, level)
            else:
                raise ValueError(
                    f"Expected tensor of 0, 1, 2 or 3 dimensions. Got tensor of {ths.ndim} dimensions"
                )

        return ths_cur

    def _reshape_ths_one_dim(self, ths, level):
        numel = 3 if self.dimension == 2 else 7
        return [ths] * numel

    def _reshape_ths_two_dim(self, ths, level):
        numel = 3 if self.dimension == 2 else 7
        if ths.size(1) == 1:
            return [ths[:, 0]] * numel
        else:
            assert ths.size(1) == self.level
            return [ths[:, level - 2]] * numel

    def _reshape_ths_three_dim(self, ths, level):
        numel = 3 if self.dimension == 2 else 7
        if ths.size(1) == 1:
            ths = ths.expand(-1, self.level, -1)
        assert (
            ths.size(1) == self.level
        ), f"Expected tensor of shape (B, {self.level}, {numel}), got {ths.shape}"
        if ths.size(-1) == numel:
            return ths.permute(2, 0, 1)[..., level - 2]
        elif ths.size(-1) == 1:
            return self._reshape_ths_two_dim(ths[..., 0], level)
        else:
            raise ValueError(
                f"Expected tensor of shape (B, {self.level}, {numel}), got {ths.shape}"
            )

    @staticmethod
    def _list_to_tuple(obj):
        r"""
        Helper function to convert lists to tuples recursively.
        This is used to ensure that the wavelet coefficients are in the correct format for the inverse wavelet transform.
        """
        if isinstance(obj, (list, tuple)):
            return tuple(WaveletDenoiser._list_to_tuple(item) for item in obj)
        return obj

    def forward(self, x, ths=0.1, **kwargs):
        r"""
        Run the model on a noisy image.

        :param torch.Tensor x: noisy image.
        :param int, float, torch.Tensor ths: thresholding parameter :math:`\gamma`.
            If `ths` is a tensor, it should be of shape
            ``(B,)`` (same coefficent for all levels), ``(B, n_levels-1)`` (one coefficient per level),
            or ``(B, n_levels-1, 3)`` (one coefficient per subband and per level). `B` should be the same as the batch size of the input or `1`.
            If ``non_linearity`` equals ``"soft"`` or ``"hard"``, ``ths`` serves as a (soft or hard)
            thresholding parameter for the wavelet coefficients. If ``non_linearity`` equals ``"topk"``,
            ``ths`` can indicate the number of wavelet coefficients
            that are kept (if ``int``) or the proportion of coefficients that are kept (if ``float``).

        """
        # Pad data
        x, padding = self.pad_input(x)

        # Apply wavelet transform
        coeffs = self.dwt(x)

        # Threshold coefficients (we do not threshold the approximation coefficients)
        coeffs = self.threshold_ND(coeffs, ths)

        # Inverse wavelet transform
        y = self.iwt(coeffs)

        # Crop data
        y = self.crop_output(y, padding)
        return y


class WaveletDictDenoiser(Denoiser):
    r"""
    Overcomplete Wavelet denoising with the :math:`\ell_1` norm.

    This denoiser is defined as the solution to the optimization problem:

    .. math::

        \underset{x}{\arg\min} \;  \|x-y\|^2 + \lambda \|\Psi x\|_n

    where :math:`\Psi` is an overcomplete wavelet transform, composed of 2 or more wavelets, i.e.,
    :math:`\Psi=[\Psi_1,\Psi_2,\dots,\Psi_L]`, :math:`\lambda>0` is a hyperparameter, and where
    :math:`\|\cdot\|_n` is either the :math:`\ell_1` norm (``non_linearity="soft"``),
    the :math:`\ell_0` norm (``non_linearity="hard"``) or a variant of the :math:`\ell_0` norm
    (``non_linearity="topk"``) where only the top-k coefficients are kept; see :class:`deepinv.models.WaveletDenoiser` for
    more details.

    The solution is not available in closed-form, thus the denoiser runs an optimization algorithm for each test image.

    .. warning::

        This model requires Pytorch Wavelets (``ptwt``) to be installed. It can be installed with
        ``pip install ptwt``.

    :param int level: decomposition level of the wavelet transform.
    :param list[str] wv: list of mother wavelets. The names of the wavelets can be found in `here
        <https://wavelets.pybytes.com/>`_. (default: ["db8", "db4"]).
    :param str device: cpu or gpu.
    :param int max_iter: number of iterations of the optimization algorithm (default: 10).
    :param str non_linearity: "soft", "hard" or "topk" thresholding (default: "soft")
    """

    def __init__(
        self,
        level=3,
        list_wv=["db8", "db4"],
        max_iter=10,
        non_linearity="soft",
        wvdim=2,
        device="cpu",
    ):
        super().__init__()
        self.level = level
        self.list_wv = list_wv
        self.list_prox = nn.ModuleList(
            [
                WaveletDenoiser(
                    level=level,
                    wv=wv,
                    non_linearity=non_linearity,
                    wvdim=wvdim,
                    device=device,
                )
                for wv in list_wv
            ]
        )
        self.max_iter = max_iter

    def forward(self, y, ths=0.1, **kwargs):
        r"""
        Run the model on a noisy image.

        :param torch.Tensor y: noisy image.
        :param float, torch.Tensor ths: noise level.
        """
        z_p = y.repeat(len(self.list_prox), *([1] * (len(y.shape))))
        p_p = torch.zeros_like(z_p)
        x = p_p.clone()
        for it in range(self.max_iter):
            x_prev = x.clone()
            for p in range(len(self.list_prox)):
                p_p[p, ...] = self.list_prox[p](z_p[p, ...], ths)
            x = torch.mean(p_p.clone(), axis=0)
            for p in range(len(self.list_prox)):
                z_p[p, ...] = x + z_p[p, ...].clone() - p_p[p, ...]
            rel_crit = torch.linalg.norm((x - x_prev).flatten()) / torch.linalg.norm(
                x.flatten() + 1e-6
            )
            if rel_crit < 1e-3:
                break
        return x

    def psi(self, x, **kwargs):
        r"""
        Returns a flattened list containing the wavelet coefficients for each wavelet.
        """
        vec = []
        for p in self.list_prox:
            vec += p.psi(x, wavelet=p.wv, level=p.level, dimension=p.dimension)
        return vec
