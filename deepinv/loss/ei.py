import torch
from torch.autograd import Variable
from utils.logger import AverageMeter, ProgressMeter
from utils.metric import cal_psnr, cal_mse



import math
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from torch import autograd as autograd


# --------------------------------------------
# EI loss
# --------------------------------------------
class EILoss(nn.Module):
    def __init__(self, transform, physics, ei_loss_weight=1.0, metric=torch.nn.MSELoss()):
        """
        Equivariant imaging loss
        https://github.com/edongdongchen/EI
        https://https://arxiv.org/pdf/2103.14756.pdf
        Args:
            ei_loss_weight (int):
        """
        super(EILoss, self).__init__()
        self.ei_loss_weight = ei_loss_weight
        self.metric = metric

        self.T = lambda x: transform.apply(x)
        self.A = lambda x: physics.A(x)

    def forward(self, y, model):
        x1 = model(y)
        x2 = self.T(x1)
        x3 = model(self.A(x2))


        loss_mc = self.metric(self.A(x1), y) # convert loss_mc (estimation to mse(f(y),x)) into PSNR: psnr = 20 * torch.log10(max_pixel / torch.sqrt(loss_mc))
        loss_ei = self.ei_loss_weight * self.metric(x3, x2) # x2 is varying...
        return loss_mc + loss_ei
