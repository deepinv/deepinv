import torch
from .dnnlib.util import open_url
import pickle
import torch.nn as nn
import os
import sys


class ModelWrapper(nn.Module):
    def __init__(self, edm_model: nn.Module):
        super().__init__()
        self.edm_model = edm_model

    def forward(self, x: torch.Tensor, t: float):
        if isinstance(t, float):
            t = torch.tensor([t] * x.size(0), device=x.device)
        return self.edm_model.forward(x, noise_labels=t, class_labels=None)


current_path = os.path.dirname(os.path.abspath(__file__))


def load_model(model_name: str = "edm-afhqv2-64x64-uncond-ve.pkl") -> nn.Module:
    r"""
    model_name (str): one of the following
        edm-afhqv2-64x64-uncond-ve.pkl
        edm-afhqv2-64x64-uncond-ve.pkl
        edm-afhqv2-64x64-uncond-vp.pkl
        edm-cifar10-32x32-cond-ve.pkl
        edm-cifar10-32x32-cond-vp.pkl
        edm-cifar10-32x32-uncond-ve.pkl
        edm-cifar10-32x32-uncond-vp.pkl
        edm-ffhq-64x64-uncond-ve.pkl
        edm-ffhq-64x64-uncond-vp.pkl
        edm-imagenet-64x64-cond-adm.pkl
    """
    os.chdir(current_path)
    sys.path.append(current_path)
    network_pkl = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/" + str(model_name)
    with open_url(network_pkl) as f:
        net = pickle.load(f)["ema"]
        print(f"Sucessfully loaded the model {model_name}")
        print(
            "Number of parameters: ",
            sum(p.numel() for p in net.model.parameters()),
        )

    os.chdir("/".join(current_path.split("/")[:-1]))
    return ModelWrapper(net.model)
