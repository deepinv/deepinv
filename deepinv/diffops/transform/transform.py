import torch
import torchvision
import deepinv.diffops.transform.shift
import deepinv.diffops.transform.rotate

def __call__(name='shift', ntrans=1):
    if name.lower()=='shift':
        return
