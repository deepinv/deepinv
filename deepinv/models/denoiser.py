import copy
import torch
import torch.nn as nn


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
    '''
    Plug-and-Play (PnP) / Regularization bu Denoising (RED) algorithms for Image Restoration.
    Consists in replacing prox_g or grad_g with a denoiser.
    '''

    def __init__(self, model_spec=None, init=None, stepsize=1., sigma_denoiser=0.05, max_iter=None,
                 device=torch.device('cpu')):
        super(Denoiser, self).__init__()

        self.denoiser = make(model_spec)
        self.max_iter = max_iter
        self.device = device  # Is this really needed?
        self.init = init

        if isinstance(sigma_denoiser, float):
            self.sigma_denoiser = [sigma_denoiser] * self.max_iter
        elif isinstance(sigma_denoiser, list):
            assert len(sigma_denoiser) == self.max_iter
            self.sigma_denoiser = sigma_denoiser
        else:
            raise ValueError('sigma_denoiser must be either float or a list of length max_iter')

        if isinstance(stepsize, float):
            self.stepsize = [stepsize] * max_iter  # Should be a list
        elif isinstance(stepsize, list):
            assert len(stepsize) == self.max_iter
            self.stepsize = stepsize
        else:
            raise ValueError('stepsize must be either float or a list of length max_iter')


class ProxDenoiser(Denoiser):
    def __init__(self, *args, **kwargs):
        super(ProxDenoiser, self).__init__(*args, **kwargs)

    def forward(self, x, sigma, it):
        if isinstance(self.denoiser, list) or isinstance(self.denoiser, nn.ModuleList):
            out = self.denoiser[it](x, sigma)
        else:
            out = self.denoiser(x, sigma)
        return out


class ScoreDenoiser(Denoiser):
    def __init__(self, *args, **kwargs):
        super(ScoreDenoiser, self).__init__(*args, **kwargs)

    def forward(self, x, sigma, it):
        if isinstance(self.denoiser, list) or isinstance(self.denoiser, nn.ModuleList):
            out = x - self.denoiser[it](x, sigma)
        else:
            out = x - self.denoiser(x, sigma)
        return out

