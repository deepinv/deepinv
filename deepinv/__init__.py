from .__about__ import *
import torch

__all__ = [
    "__title__",
    "__summary__",
    "__url__",
    "__version__",
    "__author__",
    "__license__",
]

from deepinv import models

__all__ += ["models"]

from deepinv import optim

__all__ += ["optim"]

from deepinv import loss

__all__ += ["loss"]

from deepinv import utils

__all__ += ["utils"]

from deepinv import models

__all__ += ["iterative"]

from deepinv import physics

__all__ += ["physics"]

from deepinv import datasets

__all__ += ["datasets"]

from deepinv import transform

__all__ += ["transform"]

from deepinv import sampling

__all__ += ["sampling"]

from deepinv.loss import metric

__all__ += ["metric"]

from deepinv import unfolded

__all__ += ["unfolded"]

from deepinv import training

__all__ += ["training"]

from deepinv.training import train, test, Trainer


# GLOBAL PROPERTY
dtype = torch.float
