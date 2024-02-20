import torch
import torch.nn as nn

try:
    import pywt
except:
    pywt = ImportError("The pywt package is not installed.")

try:
    import ptwt
except:
    pytorch_wavelets = ImportError("The ptwt package is not installed.")


class WaveletPrior(nn.Module):
    r"""
    Wavelet denoising with the :math:`\ell_1` norm.


    This denoiser is defined as the solution to the optimization problem:

    .. math::

        \underset{x}{\arg\min} \;  \|x-y\|^2 + \lambda \|\Psi x\|_n

    where :math:`\Psi` is an orthonormal wavelet transform, :math:`\lambda>0` is a hyperparameter, and where
    :math:`\|\cdot\|_n` is either the :math:`\ell_1` norm (``non_linearity="soft"``) or
    the :math:`\ell_0` norm (``non_linearity="hard"``). A variant of the :math:`\ell_0` norm is also available
    (``non_linearity="topk"``), where the thresholding is done by keeping the :math:`k` largest coefficients
    in each wavelet subband and setting the others to zero.

    The solution is available in closed-form, thus the denoiser is cheap to compute.

    :param int level: decomposition level of the wavelet transform
    :param str wv: mother wavelet (follows the `PyWavelets convention
        <https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html>`_) (default: "db8")
    :param str device: cpu or gpu
    :param str non_linearity: ``"soft"``, ``"hard"`` or ``"topk"`` thresholding (default: ``"soft"``).
        If ``"topk"``, only the top-k wavelet coefficients are kept.
    """

    def __init__(self, level=3, wv="db8", device="cpu", non_linearity="soft", wvdim=2):
        super().__init__()
        self.level = level
        self.wv = wv
        self.device = device
        self.non_linearity = non_linearity
        self.dimension = wvdim

    def get_ths_map(self, ths):
        if isinstance(ths, float) or isinstance(ths, int):
            ths_map = ths
        elif len(ths.shape) == 0 or ths.shape[0] == 1:
            ths_map = ths.to(self.device)
        else:
            ths_map = (
                ths.unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .to(self.device)
            )
        return ths_map

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
        ths_map = self.get_ths_map(ths)
        return torch.maximum(
            torch.tensor([0], device=x.device).type(x.dtype), x - ths_map
        ) + torch.minimum(torch.tensor([0], device=x.device).type(x.dtype), x + ths_map)

    def prox_l0(self, x, ths=0.1):
        r"""
        Hard thresholding of the wavelet coefficients.

        :param torch.Tensor x: wavelet coefficients.
        :param float, torch.Tensor ths: threshold.
        """
        if isinstance(ths, float):
            ths_map = ths
        else:
            ths_map = self.get_ths_map(ths)
            ths_map = ths_map.repeat(
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
            k = int(ths * x.shape[-2] * x.shape[-1])
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
        h, w = x.size()[-2:]
        padding_bottom = h % 2
        padding_right = w % 2
        x = torch.nn.ReplicationPad2d((0, padding_right, 0, padding_bottom))(x)
        return x, (padding_bottom, padding_right)

    def crop_output(self, x, padding):
        r"""
        Crop the output to make it compatible with the wavelet transform.
        """
        padding_bottom, padding_right = padding
        h, w = x.size()[-2:]
        return x[..., : h - padding_bottom, : w - padding_right]

    def reshape_ths(self, ths, level):
        r"""
        Reshape the thresholding parameter in the appropriate format, i.e. a list of 3 elements.
        """
        if not torch.is_tensor(ths):
            if isinstance(ths, int) or isinstance(ths, float):
                ths_cur = [ths] * 3
            elif len(ths) == 1:
                ths_cur = [ths[0]] * 3
            else:
                ths_cur = ths[level]
                if (ths_cur) == 1:
                    ths_cur = [ths_cur[0]] * 3
        else:
            if len(ths.shape) == 1:  # Needs to reshape to shape (n_levels, 3)
                ths_cur = ths.unsqueeze(0).repeat(self.level, 3)
            else:
                ths_cur = ths

        return ths_cur

    def forward(self, x, ths=0.1):
        r"""
        Run the model on a noisy image.

        :param torch.Tensor x: noisy image.
        :param int, float, torch.Tensor ths: thresholding parameter.
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
        # for level in range(1, self.level + 1):
        #     ths_cur = self.reshape_ths(ths, level)
        #
        #     for c in range(3):
        #         coeffs[level][c] = self.thresold_func(coeffs[level][c], ths_cur[c])
        coeffs = self.threshold_ND(coeffs, ths)

        # Inverse wavelet transform
        y = self.iwt(coeffs)

        # Crop data
        # y = y[..., :h, :w]
        y = self.crop_output(y, padding)
        return y


class WaveletDict(nn.Module):
    r"""
    Overcomplete Wavelet denoising with the :math:`\ell_1` norm.

    This denoiser is defined as the solution to the optimization problem:

    .. math::

        \underset{x}{\arg\min} \;  \|x-y\|^2 + \lambda \|\Psi x\|_n

    where :math:`\Psi` is an overcomplete wavelet transform, composed of 2 or more wavelets, i.e.,
    :math:`\Psi=[\Psi_1,\Psi_2,\dots,\Psi_L]`, :math:`\lambda>0` is a hyperparameter, and where
    :math:`\|\cdot\|_n` is either the :math:`\ell_1` norm (``non_linearity="soft"``),
    the :math:`\ell_0` norm (``non_linearity="hard"``) or a variant of the :math:`\ell_0` norm
    (``non_linearity="topk"``) where only the top-k coefficients are kept; see :meth:`deepinv.models.WaveletPrior` for
    more details.

    The solution is not available in closed-form, thus the denoiser runs an optimization algorithm for each test image.

    :param int level: decomposition level of the wavelet transform.
    :param list[str] wv: list of mother wavelets. The names of the wavelets can be found in `here
        <https://wavelets.pybytes.com/>`_. (default: ["db8", "db4"]).
    :param str device: cpu or gpu.
    :param int max_iter: number of iterations of the optimization algorithm (default: 10).
    :param str non_linearity: "soft", "hard" or "topk" thresholding (default: "soft")
    """

    def __init__(
        self, level=3, list_wv=["db8", "db4"], max_iter=10, non_linearity="soft"
    ):
        super().__init__()
        self.level = level
        self.list_prox = nn.ModuleList(
            [
                WaveletPrior(level=level, wv=wv, non_linearity=non_linearity)
                for wv in list_wv
            ]
        )
        self.max_iter = max_iter

    def forward(self, y, ths=0.1):
        r"""
        Run the model on a noisy image.

        :param torch.Tensor y: noisy image.
        :param float, torch.Tensor ths: noise level.
        """
        z_p = y.repeat(len(self.list_prox), 1, 1, 1, 1)
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
