from __future__ import annotations
from torch import Tensor
from deepinv.models import DRUNet
from deepinv.models.base import Denoiser
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.optim.optimizers import create_iterator
from deepinv.optim import BaseOptim
import torch


def get_DPIR_params(noise_level_img, device="cpu"):
    r"""
    Default parameters for the DPIR Plug-and-Play algorithm.

    :param float noise_level_img: Noise level of the input image.
    :return: tuple(list with denoiser noise level per iteration, list with stepsize per iteration, iterations).
    """
    max_iter = 8
    s1 = 49.0 / 255.0
    s2 = noise_level_img
    sigma_denoiser = torch.logspace(
        torch.log10(torch.tensor(s1, dtype=torch.float32)),
        torch.log10(torch.tensor(s2, dtype=torch.float32)),
        steps=max_iter,
        dtype=torch.float32,
        device=device,
    )

    stepsize = (sigma_denoiser / max(0.01, noise_level_img)) ** 2
    lamb = 1 / 0.23

    return sigma_denoiser, lamb * stepsize, max_iter


class DPIR(BaseOptim):
    r"""
    Deep Plug-and-Play (DPIR) algorithm for image restoration.

    The method is based on half-quadratic splitting (HQS) and a PnP prior with a pretrained denoiser :class:`deepinv.models.DRUNet`.
    The optimization is stopped early and the noise level for the denoiser is adapted at each iteration.
    See :ref:`sphx_glr_auto_examples_plug-and-play_demo_PnP_DPIR_deblur.py` for more details on the implementation,
    and how to adapt it to your specific problem.

    This method uses a standard :math:`\ell_2` data fidelity term.

    The DPIR method is described in :footcite:t:`zhang2021plug`.

    :param float, torch.Tensor sigma: Standard deviation of the measurement noise, which controls the choice of the
        rest of the hyperparameters of the algorithm. Default is ``0.1``.
    :param deepinv.models.Denoiser denoiser: optional denoiser. If `None`, use a pretrained denoiser :class:`deepinv.models.DRUNet`.
    :param str, torch.device device: Device to run the algorithm, either "cpu" or "cuda". Default is "cuda".


    """

    def __init__(
        self,
        sigma: float | Tensor = 0.1,
        denoiser: Denoiser = None,
        device="cuda",
    ):
        prior = PnP(
            denoiser=(
                DRUNet(pretrained="download", device=device)
                if denoiser is None
                else denoiser
            )
        )
        sigma_denoiser, stepsize, max_iter = get_DPIR_params(sigma, device=device)
        params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser}
        super(DPIR, self).__init__(
            create_iterator("HQS", prior=prior, F_fn=None, g_first=False),
            max_iter=max_iter,
            data_fidelity=L2(),
            prior=prior,
            early_stop=False,
            params_algo=params_algo,
        )
