import torch
from deepinv.optim.epll import EPLL
from deepinv.physics import Denoising, GaussianNoise
from .base import Denoiser


class EPLLDenoiser(Denoiser):
    r"""
    Expected Patch Log Likelihood denoising method.

    Denoising method based on the minimization problem

    .. math::

        \underset{x}{\arg\min} \, \|y-x\|^2 - \sum_i \log p(P_ix)

    where the first term is a standard L2 data-fidelity, and the second term represents a patch prior via
    Gaussian mixture models, where :math:`P_i` is a patch operator that extracts the ith (overlapping) patch from the image.

    :param None, deepinv.optim.utils.GaussianMixtureModel GMM: Gaussian mixture defining the distribution on the patch space.
        ``None`` creates a GMM with n_components components of dimension accordingly to the arguments patch_size and channels.
    :param int n_components: number of components of the generated GMM if GMM is ``None``.
    :param str, None pretrained: Path to pretrained weights of the GMM with file ending ``.pt``. None for no pretrained weights,
        ``"download"`` for pretrained weights on the BSDS500 dataset, ``"GMM_lodopab_small"`` for the weights from the limited-angle CT example.
        See :ref:`pretrained-weights <pretrained-weights>` for more details.
    :param int patch_size: patch size.
    :param int channels: number of color channels (e.g. 1 for gray-valued images and 3 for RGB images)
    :param str device: defines device (``cpu`` or ``cuda``)
    """

    def __init__(
        self,
        GMM=None,
        n_components=200,
        pretrained="download",
        patch_size=6,
        channels=1,
        device="cpu",
    ):
        super(EPLLDenoiser, self).__init__()
        self.PatchGMM = EPLL(
            GMM, n_components, pretrained, patch_size, channels, device
        )
        self.denoising_operator = Denoising(GaussianNoise(0))

    def forward(self, x, sigma, betas=None, batch_size=-1):
        r"""
        Denoising method based on the minimization problem.

        :param torch.Tensor y: noisy image. Shape: batch size x ...
        :param deepinv.physics.LinearPhysics physics: Forward linear operator.
        :param list[float] betas: parameters from the half-quadratic splitting. ``None`` uses
            the standard choice ``[1,4,8,16,32]/sigma_sq``
        :param int batch_size: batching the patch estimations for large images. No effect on the output,
            but a small value reduces the memory consumption
            and might increase the computation time. ``-1`` for considering all patches at once.
        """
        return self.PatchGMM(
            x,
            x_init=x.clone(),
            sigma=sigma,
            physics=self.denoising_operator,
            batch_size=batch_size,
            betas=betas,
        )
