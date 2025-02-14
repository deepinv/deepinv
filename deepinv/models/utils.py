import torch
import numpy as np


def tensor2array(img):
    img = img.cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    return img


def array2tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1)


def get_weights_url(model_name, file_name):
    return (
        "https://huggingface.co/deepinv/"
        + model_name
        + "/resolve/main/"
        + file_name
        + "?download=true"
    )


def test_pad(model, L, modulo=16):
    """
    Pads the image to fit the model's expected image size.

    Code borrowed from Kai Zhang https://github.com/cszn/DPIR/tree/master/models
    """
    h, w = L.size()[-2:]
    padding_bottom = int(np.ceil(h / modulo) * modulo - h)
    padding_right = int(np.ceil(w / modulo) * modulo - w)
    L = torch.nn.ReplicationPad2d((0, padding_right, 0, padding_bottom))(L)
    E = model(L)
    E = E[..., :h, :w]
    return E


def test_onesplit(model, L, refield=32, sf=1):
    """
    Changes the size of the image to fit the model's expected image size.

    Code borrowed from Kai Zhang https://github.com/cszn/DPIR/tree/master/models.

    :param model: model.
    :param L: input Low-quality image.
    :param refield: effective receptive field of the network, 32 is enough.
    :param sf: scale factor for super-resolution, otherwise 1.
    """
    h, w = L.size()[-2:]
    top = slice(0, (h // 2 // refield + 1) * refield)
    bottom = slice(h - (h // 2 // refield + 1) * refield, h)
    left = slice(0, (w // 2 // refield + 1) * refield)
    right = slice(w - (w // 2 // refield + 1) * refield, w)
    Ls = [
        L[..., top, left],
        L[..., top, right],
        L[..., bottom, left],
        L[..., bottom, right],
    ]
    Es = [model(Ls[i]) for i in range(4)]
    b, c = Es[0].size()[:2]
    E = torch.zeros(b, c, sf * h, sf * w).type_as(L)
    E[..., : h // 2 * sf, : w // 2 * sf] = Es[0][..., : h // 2 * sf, : w // 2 * sf]
    E[..., : h // 2 * sf, w // 2 * sf : w * sf] = Es[1][
        ..., : h // 2 * sf, (-w + w // 2) * sf :
    ]
    E[..., h // 2 * sf : h * sf, : w // 2 * sf] = Es[2][
        ..., (-h + h // 2) * sf :, : w // 2 * sf
    ]
    E[..., h // 2 * sf : h * sf, w // 2 * sf : w * sf] = Es[3][
        ..., (-h + h // 2) * sf :, (-w + w // 2) * sf :
    ]
    return E
