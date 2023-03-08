import torch
import torch.nn as nn
from .optim_iterator import OptimIterator

class ADMM(OptimIterator):

    def __init__(self, **kwargs):
        '''
        TODO: MATTHIEU
        '''
        super(ADMM, self).__init__(**kwargs)

    def forward(self, x):
        pass

