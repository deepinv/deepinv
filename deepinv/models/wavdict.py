import torch
import torch.nn as nn


class WaveletPrior(nn.Module):
    r"""
    Wavelet denoising with the :math:`\ell_1` norm.


    This denoiser is defined as the solution to the optimization problem:

    .. math::

        \underset{x}{\arg\min} \;  \|x-y\|^2 + \lambda \|\Psi x\|_n

    where :math:`\Psi` is an orthonormal wavelet transform, :math:`\lambda>0` is a hyperparameter, and where
    :math:`\|\cdot\|_n` is either the :math:`\ell_1` norm (``non_linearity="soft"``) or
    the :math:`\ell_0` norm (``non_linearity="hard"``).

    The solution is available in closed-form, thus the denoiser is cheap to compute.

    :param int level: decomposition level of the wavelet transform
    :param str wv: mother wavelet (follows the `PyWavelets convention
        <https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html>`_) (default: "db8")
    :param str device: cpu or gpu
    :param str non_linearity: "soft" or "hard" thresholding (default: "soft")
    """

    def __init__(self, level=3, wv="db8", device="cpu", non_linearity="soft"):
        super().__init__()
        self.level = level
        try:
            from pytorch_wavelets import DWTForward, DWTInverse
        except ImportError as e:
            print(
                "pywavelets is needed to use the WaveletPrior class. "
                "It should be installed with `pip install"
                "git+https://github.com/fbcotter/pytorch_wavelets.git`"
            )
            raise e
        self.dwt = DWTForward(J=self.level, wave=wv).to(device)
        self.iwt = DWTInverse(wave=wv).to(device)
        self.device = device
        self.non_linearity = non_linearity

    def get_ths_map(self, ths):
        if isinstance(ths, float):
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
        ths_map = self.get_ths_map(ths)
        return torch.maximum(
            torch.tensor([0], device=x.device).type(x.dtype), x - ths_map
        ) + torch.minimum(torch.tensor([0], device=x.device).type(x.dtype), x + ths_map)

    def prox_l0(self, x, ths=0.1):
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

    def forward(self, x, ths=0.1):
        h, w = x.size()[-2:]
        padding_bottom = h % 2
        padding_right = w % 2
        x = torch.nn.ReplicationPad2d((0, padding_right, 0, padding_bottom))(x)

        coeffs = self.dwt(x)
        for l in range(self.level):
            ths_cur = (
                ths
                if (isinstance(ths, float) or len(ths.shape) == 0 or ths.shape[0] == 1)
                else ths[l]
            )
            if self.non_linearity == "soft":
                coeffs[1][l] = self.prox_l1(coeffs[1][l], ths_cur)
            elif self.non_linearity == "hard":
                coeffs[1][l] = self.prox_l0(coeffs[1][l], ths_cur)
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
    :math:`\|\cdot\|_n` is either the :math:`\ell_1` norm (``non_linearity="soft"``) or
    the :math:`\ell_0` norm (``non_linearity="hard"``).

    The solution is not available in closed-form, thus the denoiser runs an optimization algorithm for each test image.

    :param int level: decomposition level of the wavelet transform.
    :param list[str] wv: list of mother wavelets. The names of the wavelets can be found in `here
        <https://wavelets.pybytes.com/>`_. (default: ["db8", "db4"]).
    :param str device: cpu or gpu.
    :param int max_iter: number of iterations of the optimization algorithm (default: 10).
    :param str non_linearity: "soft" or "hard" thresholding (default: "soft")
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
