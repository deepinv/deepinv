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
    def __init__(self, transform, physics, ei_loss_weight=1.0, metric=torch.nn.MSELoss()): #todo: (metric: mse, L1, L2)
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


        loss_mc = self.metric(self.A(x1), y)
        loss_ei = self.ei_loss_weight * self.metric(x3, x2)
        return loss_mc + loss_ei

# REI Loss

# --------------------------------------------
# MC loss
# --------------------------------------------
class MCLoss(nn.Module):
    def __init__(self, mc_loss_weight=1, metric=torch.nn.MSELoss()):
        """
        measurement (or data) consistency loss
        Args:
            mc_loss_weight (int):
        """
        super(EILoss, self).__init__()
        self.mc_loss_weight = mc_loss_weight

    def forward(self, y, model, physics):
        x1 = model(y)
        return self.mc_loss_weight * torch.nn.MSELoss()(physics.A(x1), y)

# --------------------------------------------
# Supversided loss
# --------------------------------------------
class SupLoss(nn.Module):
    def __init__(self, sup_loss_weight=1):
        """
        supervised (paired GT x and meas. y) loss
        Args:
            sup_loss_weight (int):
        """
        super(EILoss, self).__init__()
        self.sup_loss_weight = sup_loss_weight

    def forward(self, x, y, model):
        x1 = model(y)
        return self.sup_loss_weight * torch.nn.MSELoss()(x1, x)


#todo: REQ loss, SURE_alone loss (with noise distribution),
#todo: define an individual noise module (Gaussian, Possion, MPG)


# --------------------------------------------
# TV loss
# --------------------------------------------
class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        """
        Total variation loss
        https://github.com/jxgu1016/Total_Variation_Loss.pytorch
        Args:
            tv_loss_weight (int):
        """
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


# --------------------------------------------
# Charbonnier loss
# --------------------------------------------
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-9):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + self.eps))
        return loss



def r1_penalty(real_pred, real_img):
    """R1 regularization for discriminator. The core idea is to
        penalize the gradient on real data alone: when the
        generator distribution produces the true data distribution
        and the discriminator is equal to 0 on the data manifold, the
        gradient penalty ensures that the discriminator cannot create
        a non-zero gradient orthogonal to the data manifold without
        suffering a loss in the GAN game.
        Ref:
        Eq. 9 in Which training methods for GANs do actually converge.
        """
    grad_real = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3])
    grad = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (
        path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()


def gradient_penalty_loss(discriminator, real_data, fake_data, weight=None):
    """Calculate gradient penalty for wgan-gp.
    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        weight (Tensor): Weight tensor. Default: None.
    Returns:
        Tensor: A tensor for gradient penalty.
    """

    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

    # interpolate between real_data and fake_data
    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if weight is not None:
        gradients = gradients * weight

    gradients_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    if weight is not None:
        gradients_penalty /= torch.mean(weight)

    return gradients_penalty