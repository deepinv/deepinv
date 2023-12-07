import numpy as np

import torch


class EquivariantDenoiser(torch.nn.Module):
    r"""
    Applies geometric equivariant transforms to a denoiser.

    The transformations can be either averaged (turning the denoiser into an equivariant denoiser) or applied randomly.

    :param callable denoiser: Denoiser :math:`\operatorname{D}_{\sigma}`.
    :param bool rand_rot: if True, the denoiser is applied to a randomly rotated version of the input image.
    :param bool mean_rot: if True, the denoiser is applied to several randomly rotated version of the input image and the result is averaged.
    :param bool rand_translations: if True, the denoiser is applied to a randomly translated version of the input image.
    """

    def __init__(self, denoiser, rand_rot=False, mean_rot=False, rand_translations=False):
        super().__init__()
        self.denoiser = denoiser
        self.rand_rot = rand_rot
        self.mean_rot = mean_rot
        self.rand_translations = rand_translations

    def forward(self, x, sigma, retain_grad=False):
        r"""
        Applies the denoiser to the input image with the appropriate transformation.

        :param torch.Tensor x: input image.
        :param float sigma: noise level.
        :param bool retain_grad: if True, the gradient of the denoiser is retained.
        :return: denoised image.
        """
        return denoise_rotate(self.denoiser, x, sigma, rand_rot=self.rand_rot, mean_rot=self.mean_rot,
                              rand_translations=self.rand_translations, retain_grad=retain_grad)


def denoise_rotate(denoiser, image, sigma, rand_rot=False, mean_rot=False, rand_translations=False, retain_grad=False):
    r'''
    Applies rotations to the input image, denoises it with the denoiser and rotates back the result.

    :param callable denoiser: Denoiser :math:`\operatorname{D}_{\sigma}`.
    :param torch.Tensor image: input image.
    :param float sigma: noise level.
    :param bool rand_rot: if True, the denoiser is applied to a randomly rotated version of the input image.
    :param bool mean_rot: if True, the denoiser is applied to several randomly rotated version of the input image and the result is averaged.
    :param bool rand_translations: if True, the denoiser is applied to a randomly translated version of the input image.
    :return: denoised image.
    '''
    if rand_rot:

        k = np.random.choice([0, 1, 2, 3])
        denoised = denoise_rotate_fn(denoiser, image, sigma, rot_idx=k, retain_grad=retain_grad)

    elif mean_rot:

        denoised = torch.zeros_like(image)
        for k in range(4):
            denoised = denoised + denoise_rotate_fn(denoiser, image, sigma, rot_idx=k, retain_grad=retain_grad)
        denoised = denoised/4.

    elif rand_translations:
        k = np.random.choice([0, 1, 2, 3])
        flip = np.random.choice([0, 1, 2])

        if flip > 0:
            image = torch.flip(image, dims=[-flip])

        x_shift = np.random.choice(list(range(-64, 64)))
        y_shift = np.random.choice(list(range(-64, 64)))
        denoised = denoise_rotate_translate_fn(denoiser, image, sigma, rot_idx=k, translate_shift=(x_shift, y_shift))

        if flip>0:
            denoised = torch.flip(denoised, dims=[-flip])

    else:
        denoised = denoise_rotate_fn(denoiser, image, sigma, rot_idx=0, retain_grad=retain_grad)

    return denoised


def denoise_rotate_fn(denoiser, image, sigma, rot_idx=0, retain_grad=False):
    r'''
    Function applying the appropriate geometric transform, denoising and inverse transform to the input image.

    :param callable denoiser: Denoiser :math:`\operatorname{D}_{\sigma}`.
    :param torch.Tensor image: input image.
    :param float sigma: noise level.
    :param int rot_idx: index of the rotation (1 corresponds to 90 degrees rotation, 2 to 180 degrees rotation, etc.)
    :param bool retain_grad: if True, the gradient of the denoiser is retained.
    :return: denoised image.
    '''
    image = torch.rot90(image, k=rot_idx, dims=[-2, -1])
    if not retain_grad:
        with torch.no_grad():
            if denoiser.__class__.__name__ == 'DiffUNet':
                image_pad, p = get_padding(image)
                denoised_pad = denoiser(image_pad, sigma)
                if p[0] == 0 and p[2] == 0:
                    denoised = denoised_pad
                else:
                    denoised = denoised_pad[..., p[2]:-p[3], p[0]:-p[1]]
            else:
                denoised = denoiser(image, sigma)
    else:
        if denoiser.__class__.__name__ == 'DiffUNet':
            image_pad, p = get_padding(image)
            denoised_pad = denoiser(image_pad, sigma)
            if p[0] == 0 and p[2] == 0:
                denoised = denoised_pad
            else:
                denoised = denoised_pad[..., p[2]:-p[3], p[0]:-p[1]]
        else:
            denoised = denoiser(image, sigma)
    denoised = torch.rot90(denoised, k=-rot_idx, dims=[-2, -1])
    return denoised


def denoise_rotate_translate_fn(denoiser, image, sigma, rot_idx=0, translate_shift=(0, 0)):
    r'''
    Function applying the appropriate geometric transform, denoising and inverse transform to the input image.

    :param callable denoiser: Denoiser :math:`\operatorname{D}_{\sigma}`.
    :param torch.Tensor image: input image.
    :param float sigma: noise level.
    :param int rot_idx: index of the rotation (1 corresponds to 90 degrees rotation, 2 to 180 degrees rotation, etc.)
    :param tuple translate_shift: tuple of integers corresponding to the translation to apply to the image.
    :return: denoised image.
    '''
    image = torch.roll(image, shifts=translate_shift, dims=(-2, -1))
    image = torch.rot90(image, k=rot_idx, dims=[-2, -1])
    with torch.no_grad():
        denoised = denoiser(image, sigma)
    denoised = torch.rot90(denoised, k=-rot_idx, dims=[-2, -1])
    denoised = torch.roll(denoised, shifts=(-translate_shift[0], -translate_shift[1]), dims=(-2, -1))
    return denoised


def get_padding(im):
    r'''
    Function padding the input image to the nearest power of 2.

    :param torch.Tensor im: input image.
    :return: padded image and padding.
    '''
    s_1 = int(2 ** (np.ceil(np.log10(im.shape[-2]) / np.log10(2))))
    s_2 = int(2 ** (np.ceil(np.log10(im.shape[-1]) / np.log10(2))))

    wpad_1 = (s_1 - im.shape[-2]) // 2
    wpad_2 = (s_1 - im.shape[-2] + 1) // 2
    hpad_1 = (s_2 - im.shape[-1]) // 2
    hpad_2 = (s_2 - im.shape[-1] + 1) // 2

    p = (hpad_1, hpad_2, wpad_1, wpad_2)

    im_pad = torch.nn.functional.pad(im, p, 'circular')

    return im_pad, p
