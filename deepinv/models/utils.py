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
