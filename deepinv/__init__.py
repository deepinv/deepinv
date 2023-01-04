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


try:
    from .diffops import models, physics, transform

    __all__ += ['models']
except ImportError:
    pass

try:
    from deepinv import loss
    __all__ += ['loss']
except ImportError:
    print('Warning: couldnt import loss subpackage')
    pass


try:
    from deepinv.diffops import models
    __all__ += ['iterative']
except ImportError:
    print('Warning: couldnt import models subpackage')
    pass

try:
    from deepinv.diffops.models import iterative
    __all__ += ['iterative']
except ImportError:
    print('Warning: couldnt import iterative subpackage')
    pass

try:
    from deepinv.diffops import physics
    __all__ += ['physics']
except ImportError:
    print('Warning: couldnt import physics subpackage')
    pass

try:
    from deepinv import datasets
    __all__ += ['datasets']
except ImportError:
    print('Warning: couldnt import datasets subpackage')
    pass

try:
    __all__ += ['transform']
except ImportError:
    print('Warning: couldnt import transform subpackage')
    pass


try:
    from deepinv.diffops import noise
    __all__ += ['noise']
except ImportError:
    pass

try:
    from torch import optim
    __all__ += ['optim']
except ImportError:
    pass

try:
    from deepinv.loss import loss as metric
    __all__ += ['metric']
except ImportError:
    pass

try:
    from deepinv.training_utils import train, test
    __all__ += ['metric']
except ImportError:
    pass


# GLOBAL PROPERTY
dtype = torch.float
device = torch.device(f'cuda:0')

