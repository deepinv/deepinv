import torch
import torch.nn as nn
from .base import Denoiser

try:
    import ptwt
    import pywt
except:
    ptwt = ImportError("The ptwt package is not installed.")
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
    :param str device: cpu or gpu
    :param str non_linearity: ``"soft"``, ``"hard"`` or ``"topk"`` thresholding (default: ``"soft"``).
        If ``"topk"``, only the top-k wavelet coefficients are kept.
    """

    def __init__(self, level=3, wv="db8", device="cpu", non_linearity="soft", wvdim=2):
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

    def dwt(self, x):
        r"""
        Applies the wavelet decomposition.
        """
        if self.dimension == 2:
            dec = ptwt.wavedec2(x, pywt.Wavelet(self.wv), mode="zero", level=self.level)
        elif self.dimension == 3:
            dec = ptwt.wavedec3(x, pywt.Wavelet(self.wv), mode="zero", level=self.level)
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
    def psi(x, wavelet="db2", level=2, dimension=2):
        r"""
        Returns a flattened list containing the wavelet coefficients.

        :param torch.Tensor x: input image.
        :param str wavelet: mother wavelet.
        :param int level: decomposition level.
        :param int dimension: dimension of the wavelet transform (either 2 or 3).
        """
        if dimension == 2:
            dec = ptwt.wavedec2(x, pywt.Wavelet(wavelet), mode="zero", level=level)
            dec = [list(t) if isinstance(t, tuple) else t for t in dec]
            vec = [decl.flatten() for l in range(1, len(dec)) for decl in dec[l]]
        elif dimension == 3:
            dec = ptwt.wavedec3(x, pywt.Wavelet(wavelet), mode="zero", level=level)
            dec = [list(t) if isinstance(t, tuple) else t for t in dec]
            vec = [dec[l][key].flatten() for l in range(1, len(dec)) for key in dec[l]]
        return vec

    def iwt(self, coeffs):
        r"""
        Applies the wavelet recomposition.
        """
        coeffs = [tuple(t) if isinstance(t, list) else t for t in coeffs]
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
        return torch.maximum(
            torch.tensor([0], device=x.device).type(x.dtype), x - abs(ths)
        ) + torch.minimum(
            torch.tensor([0], device=x.device).type(x.dtype), x + abs(ths)
        )

    def prox_l0(self, x, ths=0.1):
        r"""
        Hard thresholding of the wavelet coefficients.

        :param torch.Tensor x: wavelet coefficients.
        :param float, torch.Tensor ths: threshold.
        """
        if isinstance(ths, float):
            ths_map = ths
        else:
            ths_map = ths.repeat(
                1, 1, 1, x.shape[-2], x.shape[-1]
            )  # Reshaping to image wavelet shape
        out = x.clone()
        out[abs(out) < ths_map] = 0
        return out

    def hard_threshold_topk(self, x, ths=0.1):
        r"""
        Hard thresholding of the wavelet coefficients by keeping only the top-k coefficients and setting the others to
        0.

        :param torch.Tensor x: wavelet coefficients.
        :param float, int ths: top k coefficients to keep. If ``float``, it is interpreted as a proportion of the total
            number of coefficients. If ``int``, it is interpreted as the number of coefficients to keep.
        """
        if isinstance(ths, float):
            k = int(ths * x.shape[-3] * x.shape[-2] * x.shape[-1])
        else:
            k = int(ths)

        # Reshape arrays to 2D and initialize output to 0
        x_flat = x.reshape(x.shape[0], -1)
        out = torch.zeros_like(x_flat)

        topk_indices_flat = torch.topk(abs(x_flat), k, dim=-1)[1]

        # Convert the flattened indices to the original indices of x
        batch_indices = (
            torch.arange(x.shape[0], device=x.device).unsqueeze(1).repeat(1, k)
        )
        topk_indices = torch.stack([batch_indices, topk_indices_flat], dim=-1)

        # Set output's top-k elements to values from original x
        out[tuple(topk_indices.view(-1, 2).t())] = x_flat[
            tuple(topk_indices.view(-1, 2).t())
        ]
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
                coeffs[level][key] = self.prox_l1(coeffs[level][key], ths_cur[c])
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
            x = torch.nn.ReplicationPad2d((0, p[0], 0, p[1]))(x)
        elif self.dimension == 3:
            d, h, w = x.size()[-3:]
            padding_depth = d % 2
            padding_bottom = h % 2
            padding_right = w % 2
            p = (padding_depth, padding_bottom, padding_right)
            x = torch.nn.ReplicationPad3d((0, p[0], 0, p[1], 0, p[2]))(x)
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
        numel = 3 if self.dimension == 2 else 7
        if not torch.is_tensor(ths):
            if isinstance(ths, int) or isinstance(ths, float):
                ths_cur = [ths] * numel
            elif len(ths) == 1:
                ths_cur = [ths[0]] * numel
            else:
                ths_cur = ths[level]
                if len(ths_cur) == 1:
                    ths_cur = [ths_cur[0]] * numel
        else:
            if len(ths.shape) == 1:  # Needs to reshape to shape (n_levels-1, 3)
                ths_cur = ths.squeeze().repeat(numel)
            else:
                ths_cur = ths[level - 2]

        return ths_cur

    def forward(self, x, ths=0.1, **kwargs):
        r"""
        Run the model on a noisy image.

        :param torch.Tensor x: noisy image.
        :param int, float, torch.Tensor ths: thresholding parameter :math:`\gamma`.
            If `ths` is a tensor, it should be of shape
            ``(1, )`` (same coefficent for all levels), ``(n_levels-1, )`` (one coefficient per level),
            or ``(n_levels-1, 3)`` (one coefficient per subband and per level).
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
