import numpy as np

import torch


class EquivariantDenoiser(torch.nn.Module):
    r"""
    Turns the input denoiser into an equivariant denoiser with respect to geometric transforms.

    Recall that a denoiser is equivariant with respect to a group of transformations if it commutes with the action of
    the group. More precisely, let :math:`\mathcal{G}` be a group of transformations :math:`\{T_g\}_{g\in \mathcal{G}}`
    and :math:`\denoisername` a denoiser. Then, :math:`\denoisername` is equivariant with respect to :math:`\mathcal{G}`
    if :math:`\denoisername(T_g(x)) = T_g(\denoisername(x))` for any image :math:`x` and any :math:`g\in \mathcal{G}`.

    The denoiser can be turned into an equivariant denoiser by averaging over the group of transforms, i.e.

    .. math::
        \operatorname{D}^{\text{eq}}_{\sigma}(x) = \frac{1}{|\mathcal{G}|}\sum_{g\in \mathcal{G}} T_g^{-1}(\operatorname{D}_{\sigma}(T_g(x))).

    Otherwise, as proposed in https://arxiv.org/abs/2312.01831, a Monte Carlo approximation can be obtained by
    sampling :math:`g \sim \mathcal{G}` at random and applying

    .. math::
        \operatorname{D}^{\text{MC}}_{\sigma}(x) = T_g^{-1}(\operatorname{D}_{\sigma}(T_g(x))).


    :param callable denoiser: Denoiser :math:`\operatorname{D}_{\sigma}`.
    :param str transform: type of geometric transformation. Can be either 'rotations', 'flips' or 'rotoflips'.
        If 'rotations', the group of transformations contains the 4 rotations by multiples of 90 degrees; if 'flips',
        the group of transformations contains the 2 horizontal and vertical flips; if 'rotoflips', the group of
        transformations contains the 8 rotations and flips.
    :param bool random: if True, the denoiser is applied to a randomly transformed version of the input image.
        If False, the denoiser is applied to the average of all the transformed images, turning the denoiser into an
        equivariant denoiser with respect to the chosen group of transformations. Otherwise, it is a Monte-Carlo
        approximation of an equivariant denoiser.
    """

    def __init__(self, denoiser, transform="rotations", random=True):
        super().__init__()
        self.denoiser = denoiser
        self.rotations = True if "rot" in transform else False
        self.flips = True if "flip" in transform else False
        self.random = random

    def forward(self, x, sigma):
        r"""
        Applies the denoiser to the input image with the appropriate transformation.

        :param torch.Tensor x: input image.
        :param float sigma: noise level.
        :return: denoised image.
        """
        return denoise_rotate(
            self.denoiser,
            x,
            sigma,
            rotations=self.rotations,
            flips=self.flips,
            random=self.random,
        )


def denoise_rotate(
    denoiser,
    image,
    sigma,
    rotations=True,
    flips=False,
    random=True,
):
    r"""
    Applies a geometric transform (rotations and/or flips) to the input image, denoises it with the denoiser and
    transform back the result. The output is either the average of all the transformed images (if random=False) or a
    randomly transformed version of the denoised image (if random=True).

    :param callable denoiser: Denoiser :math:`\operatorname{D}_{\sigma}`.
    :param torch.Tensor image: input image.
    :param float sigma: noise level.
    :param bool rotations: if True, rotations are applied to the input image.
    :param bool flips: if True, flips are applied to the input image.
    :param bool random: if True, the denoiser is applied to a randomly transformed version of the input image.
    :return: denoised image.
    """
    if random:
        if rotations:
            idx = np.random.randint(8) if flips else np.random.randint(4)
        elif flips:
            idx = np.random.choice([4, 6])
        denoised = denoise_rotate_flip_fn(denoiser, image, sigma, idx)
    else:
        if rotations:
            list_idx = list(range(8)) if flips else list(range(4))
        elif flips:
            list_idx = [4, 6]
        denoised = torch.zeros_like(image)
        for idx in list_idx:
            denoised = denoised + denoise_rotate_flip_fn(denoiser, image, sigma, idx)
        denoised = denoised / len(list_idx)
    return denoised


def denoise_rotate_flip_fn(denoiser, x, sigma_den, idx):
    if idx == 0:
        out = denoiser(x, sigma_den)
    elif idx == 1:
        out = rot3(denoiser(rot1(x), sigma_den))
    elif idx == 2:
        out = rot2(denoiser(rot2(x), sigma_den))
    elif idx == 3:
        out = rot1(denoiser(rot3(x), sigma_den))
    elif idx == 4:
        out = hflip(denoiser(hflip(x), sigma_den))
    elif idx == 5:
        out = hflip(rot3(denoiser(rot1(hflip(x)), sigma_den)))
    elif idx == 6:
        out = hflip(rot2(denoiser(rot2(hflip(x)), sigma_den)))
    elif idx == 7:
        out = hflip(rot1(denoiser(rot3(hflip(x)), sigma_den)))
    return out


def hflip(x):
    return torch.flip(x, dims=[-1])


def rot1(x):
    return torch.rot90(x, k=1, dims=[-2, -1])


def rot2(x):
    return torch.rot90(x, k=2, dims=[-2, -1])


def rot3(x):
    return torch.rot90(x, k=3, dims=[-2, -1])
