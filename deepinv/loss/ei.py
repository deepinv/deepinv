import torch
import torch.nn as nn
from torch.autograd import Variable

# --------------------------------------------
# EI loss
# --------------------------------------------
class EILoss(nn.Module):
    def __init__(self, transform, metric=torch.nn.MSELoss(), noise=True, weight=1.):
        '''
        Equivariant imaging self-supervised loss
        https://https://arxiv.org/pdf/2103.14756.pdf
        https://https://arxiv.org/pdf/2111.12855.pdf
        :param transform: Transform to generate the virtually augmented measurement. It can be any torch-differentiable
         function (deepinv.diffops.transform, torchvision.transforms, torch.nn.Modules, etc.)
        :param metric: Metric used to compute the error between the reconstructed augmented measurement and the reference image.
        :param noise: Apply noise (of physics) in the virtually augmented measurement.
        :param weight: Weight of the loss.
        '''
        super(EILoss, self).__init__()
        self.name = 'ei'
        self.metric = metric
        self.weight = weight
        self.T = transform
        self.noise = noise

    def forward(self, x1, physics, f):
        x2 = self.T(x1)

        if self.noise:
            y = physics(x2)
        else:
            y = physics.A(x2)

        x3 = f(y, physics)

        loss_ei = self.weight*self.metric(x3, x2)
        return loss_ei


# --------------------------------------------
# Adversarial EI loss
# --------------------------------------------

class AdvEILoss(nn.Module):
    def __init__(self, transform, physics, loss_weight_adv=1.0, loss_weight_ei=1.0, metric_adv=torch.nn.MSELoss(), metric_ei=torch.nn.MSELoss()):
        """
        Equivariant imaging loss
        https://github.com/edongdongchen/Adversarial EI
        https://https://arxiv.org/pdf/2103.14756.pdf
        Args:
            adv_loss_weight (float)
            ei_loss_weight (float):
        """
        super(AdvEILoss, self).__init__()
        self.loss_weight_ei = loss_weight_ei
        self.loss_weight_adv = loss_weight_adv
        self.metric_adv = metric_adv
        self.metric_ei = metric_ei

        self.T = lambda x: transform(x)
        self.n_trans = transform.n_trans
        self.A = lambda x: physics.A(x)
        self.A_dagger = lambda y: physics.A_dagger(y)

    def forward(self, y0, discriminator, generator, optimizer_G, optimizer_D):

        x0 = Variable(self.A_dagger(y0))  # range input (pr)

        # Adversarial ground truths
        valid = torch.ones(y0.shape[0], *discriminator.output_shape).type(y0.dtype).to(y0.device)
        valid_ei = torch.ones(y0.shape[0] * self.n_trans, *discriminator.output_shape).type(y0.dtype).to(
            y0.device)
        fake_ei = torch.zeros(y0.shape[0] * self.n_trans, *discriminator.output_shape).type(y0.dtype).to(
            y0.device)

        valid = Variable(valid, requires_grad=False)
        valid_ei = Variable(valid_ei, requires_grad=False)
        fake_ei = Variable(fake_ei, requires_grad=False)
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images from range input
        x1 = generator(x0)
        y1 = self.A(x1)

        # TI: x2, x3
        x2 = self.T(x1)
        x3 = generator(self.A_dagger(self.A(x2)))

        # Loss measures generator's ability to forward consistency and TI
        loss_mc = self.metric_ei(y1, y0)
        loss_eq = self.metric_ei(x3, x2)

        # Loss measures generator's ability to fool the discriminator
        loss_g = self.metric_adv(discriminator(x2), valid_ei)

        loss_G = loss_mc + self.loss_weight_ei * loss_eq + self.loss_weight_adv * loss_g

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = self.metric_adv(discriminator(x1.detach()), valid)
        fake_loss = self.metric_adv(discriminator(x2.detach()), fake_ei)

        loss_D = 0.5 * self.adv_loss_weight * (real_loss + fake_loss)

        loss_D.backward()

        optimizer_D.step()

        return loss_G, loss_D