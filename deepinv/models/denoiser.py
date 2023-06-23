import copy
import torch
import torch.nn as nn


def online_weights_path():
    return "https://mycore.core-cloud.net/index.php/s/9EzDqcJxQUJKYul/download?path=%2Fweights&files="


models = {}


# TAKEN FROM https://github.com/jaewon-lee-b/lte/blob/main/models/models.py
def register(name):
    def decorator(cls):
        models[name] = cls
        return cls

    return decorator


def make(model_spec, args=None):
    if args is not None:
        model_args = copy.deepcopy(model_spec["args"])
        model_args.update(args)
    else:
        model_args = {}
    model = models[model_spec["name"]](**model_args)
    return model


class Denoiser(nn.Module):
    r"""
    Builds a (possibly pretrained) denoiser.

    The input should be a dictionary containing the inputs to a denoiser in :ref:`models`.

    For example:

    ::

        # Load the DnCNN denoiser with weights trained using the Lipschitz constraints.
        model_spec = {
        "name": "dncnn",
        "args": {
            "device": dinv.device,
            "in_channels": 3,
            "out_channels": 3,
            "pretrained": "download_lipschitz",
            },
        }

        model = Denoiser(model_spec)

    :param dict model_spec: a dictionary containing the necessary information for generating the model.
    """

    def __init__(self, model_spec=None, denoiser=None):
        super(Denoiser, self).__init__()
        if denoiser is not None:
            self.denoiser = denoiser
        elif model_spec is not None:
            self.denoiser = make(model_spec)
        else:
            raise ValueError("Either denoiser or model_spec must be provided.")

    def forward(self, x, sigma):
        r""" """
        return self.denoiser(x, sigma)
