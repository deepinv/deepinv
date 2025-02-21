from deepinv.models import DRUNet
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.optim.optimizers import create_iterator
from deepinv.optim import BaseOptim
import numpy as np


def get_DPIR_params(noise_level_img):
    r"""
    Default parameters for the DPIR Plug-and-Play algorithm.

    :param float noise_level_img: Noise level of the input image.
    :return: tuple(list with denoiser noise level per iteration, list with stepsize per iteration, iterations).
    """
    max_iter = 8
    s1 = 49.0 / 255.0
    s2 = noise_level_img
    sigma_denoiser = np.logspace(np.log10(s1), np.log10(s2), max_iter).astype(
        np.float32
    )
    stepsize = (sigma_denoiser / max(0.01, noise_level_img)) ** 2
    lamb = 1 / 0.23
    return list(sigma_denoiser), list(lamb * stepsize), max_iter


class DPIR(BaseOptim):
    r"""
    Deep Plug-and-Play (DPIR) algorithm for image restoration.

    The method is based on half-quadratic splitting (HQS) and a PnP prior with a pretrained denoiser :class:`deepinv.models.DRUNet`.
    The optimization is stopped early and the noise level for the denoiser is adapted at each iteration.
    See :ref:`sphx_glr_auto_examples_plug-and-play_demo_PnP_DPIR_deblur.py` for more details on the implementation,
    and how to adapt it to your specific problem.

    This method uses a standard :math:`\ell_2` data fidelity term.

    The DPIR method is described in Zhang, K., Zuo, W., Gu, S., & Zhang, L. (2017). "Learning deep CNN denoiser prior for image restoration"
    In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3929-3938).

    :param float sigma: Standard deviation of the measurement noise, which controls the choice of the
        rest of the hyperparameters of the algorithm. Default is ``0.1``.
    :param str, torch.device device: Device to run the algorithm, either "cpu" or "cuda". Default is "cuda".
    """

    def __init__(self, sigma=0.1, device="cuda"):
        prior = PnP(denoiser=DRUNet(pretrained="download", device=device))
        sigma_denoiser, stepsize, max_iter = get_DPIR_params(sigma)
        params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser}
        super(DPIR, self).__init__(
            create_iterator("HQS", prior=prior, F_fn=None, g_first=False),
            max_iter=max_iter,
            data_fidelity=L2(),
            prior=prior,
            early_stop=False,
            params_algo=params_algo,
        )
