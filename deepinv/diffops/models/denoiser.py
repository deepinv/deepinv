import copy
import torch
import torch.nn as nn

import numpy as np


models = {}

# TAKEN FROM https://github.com/jaewon-lee-b/lte/blob/main/models/models.py
def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(model_spec, args=None):
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    model = models[model_spec['name']](**model_args)
    return model

class Denoiser(nn.Module):
    def __init__(self, model_spec=None):
        '''
        TODO: write description
        '''
        super(Denoiser, self).__init__()
        self.model = make(model_spec)
        
    def forward(self, x, sigma):
        return self.model(x, sigma)