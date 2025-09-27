r"""A PyTorch dataset of 20 Normal Brains based on Brainweb"""

from deepinv.datasets.base import ImageDataset
import brainweb_dl as bwdl
import torch
import numpy as np


mri_2D = torch.Tensor(np.flipud(bwdl.get_mri(4, "T1")[80, ...]).astype(np.complex64))


class BrainWeb(ImageDataset):
    def __init__():
        return
