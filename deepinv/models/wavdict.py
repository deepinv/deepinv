import torch
import torch.nn as nn

try:
    import pytorch_wavelets
except:
    pytorch_wavelets = ImportError("The pytorch_wavelets package is not installed.")


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

    def __init__(self, level=3, wv="db8", device="cpu", non_linearity="soft"):
        if isinstance(pytorch_wavelets, ImportError):
            raise ImportError(
                "pytorch_wavelets is needed to use the WaveletPrior class. "
                "It should be installed with `pip install "
                "git+https://github.com/fbcotter/pytorch_wavelets.git`"
            ) from pytorch_wavelets
        super().__init__()
        self.level = level
        self.dwt = pytorch_wavelets.DWTForward(J=self.level, wave=wv).to(device)
        self.iwt = pytorch_wavelets.DWTInverse(wave=wv).to(device)
        self.device = device
        self.non_linearity = non_linearity

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
        x_flat = x.view(x.shape[0], -1)
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
        h, w = x.size()[-2:]
        padding_bottom = h % 2
        padding_right = w % 2
        x = torch.nn.ReplicationPad2d((0, padding_right, 0, padding_bottom))(x)

        coeffs = self.dwt(x)
        for l in range(self.level):
            ths_cur = (
                ths
                if (
                    isinstance(ths, int)
                    or isinstance(ths, float)
                    or len(ths.shape) == 0
                    or ths.shape[0] == 1
                )
                else ths[l]
            )
            if self.non_linearity == "soft":
                coeffs[1][l] = self.prox_l1(coeffs[1][l], ths_cur)
            elif self.non_linearity == "hard":
                coeffs[1][l] = self.prox_l0(coeffs[1][l], ths_cur)
            elif self.non_linearity == "topk":
                coeffs[1][l] = self.hard_threshold_topk(coeffs[1][l], ths_cur)
        y = self.iwt(coeffs)

        y = y[..., :h, :w]
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
