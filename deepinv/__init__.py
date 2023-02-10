from .__about__ import *
import torch


__all__ = [
    "__title__",
    "__summary__",
    "__url__",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]

from .diffops import models, physics, transform
__all__ += ['models']

from deepinv import optim
__all__ += ['optim']

from deepinv import loss
__all__ += ['loss']

from deepinv.diffops import models
__all__ += ['iterative']

from deepinv.diffops.models import iterative
__all__ += ['iterative']

from deepinv.diffops import physics
__all__ += ['physics']

from deepinv import datasets
__all__ += ['datasets']

__all__ += ['transform']

from deepinv.diffops import noise
__all__ += ['noise']

from deepinv import sampling
__all__ += ['sampling']

from deepinv.loss import loss as metric
__all__ += ['metric']

from deepinv.training_utils import train, test

# GLOBAL PROPERTY
dtype = torch.float

def get_freer_gpu():
    import os
    import numpy as np
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    idx = np.argmax(memory_available)
    print(f'Selected GPU {idx} with {np.max(memory_available)} MB free memory ')
    return idx

if torch.cuda.is_available():
    free_gpu_id = get_freer_gpu()
    device = torch.device(f'cuda:{free_gpu_id}')
else:
    device = 'cpu'

