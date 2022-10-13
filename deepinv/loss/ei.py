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


        loss_mc = self.metric(self.A(x1), y) # convert loss_mc (estimation to mse(f(y),x)) into PSNR: psnr = 20 * torch.log10(max_pixel / torch.sqrt(loss_mc))
        loss_ei = self.ei_loss_weight * self.metric(x3, x2) # x2 is varying...
        return loss_mc + loss_ei




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

        self.T = lambda x: transform.apply(x)
        self.n_trans = transform.n_trans
        self.A = lambda x: physics.A(x)
        self.A_dagger = lambda y: physics.A_dagger(y)

    def forward(self, y0, discriminator, generator):

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

        losses_mc.update(loss_mc.item())
        losses_ei.update(loss_eq.item())

        # measure reconstruction accuracy/error and record loss
        psnr_model.update(cal_psnr(x1, x, max_pixel=1, complex=args.task in ['mri']))
        psnr_fbp.update(cal_psnr(x0, x, max_pixel=1, complex=args.task in ['mri']))
        mse_model.update(cal_mse(x1, x))
        mse_fbp.update(cal_mse(x0, x))

        losses_D.update(loss_D.item())
        losses_G.update(loss_G.item())
        losses_g.update(loss_g.item())








        x1 = model(y)
        x2 = self.T(x1)
        x3 = model(self.A(x2))


        loss_mc = self.metric(self.A(x1), y)
        loss_ei = self.ei_loss_weight * self.metric(x3, x2)
        return loss_mc + loss_ei



def closure_ei_adv(epoch, train_loader, generator, discriminator,
               physics, criterion, criterion_adversarial, optimizer_G, optimizer_D,
               transform, args):
    # TODO: in adv, criterion should be BCELoss? No

    # adversarial_criterion = torch.nn.BCELoss()
    # criterion = torch.nn.BCELoss()

    psnr_model = AverageMeter('psnr_net', ':2.2f')
    psnr_fbp = AverageMeter('psnr_fbp', ':2.2f')
    mse_model = AverageMeter('mse_net', ':.4e')
    mse_fbp = AverageMeter('mse_fbp', ':.4e')
    # losses = AverageMeter('loss', ':.4e')
    losses_mc = AverageMeter('loss_mc', ':.4e')
    losses_ei = AverageMeter('loss_eq', ':.4e')
    losses_g = AverageMeter('loss_g', ':.4e')
    losses_G = AverageMeter('loss_G', ':.4e')
    losses_D = AverageMeter('loss_D', ':.4e')

    meters = [psnr_model, psnr_fbp, mse_model, mse_fbp, losses_mc, losses_ei, losses_g, losses_G, losses_D]

    progress = ProgressMeter(args.epochs, meters, prefix=f"[net: {args.net_name}]\tEpoch")

    for i, x in enumerate(train_loader):
        x = x[0] if isinstance(x, list) else x
        x = x.type(args.dtype).to(args.device)

        y0 = physics.A(x)  # generate measurement input y

        x0 = Variable(physics.A_dagger(y0))  # range input (pr)

        # Adversarial ground truths
        valid = torch.ones(y0.shape[0], *discriminator.output_shape).type(args.dtype).to(args.device)
        valid_ei = torch.ones(y0.shape[0] * transform.n_trans, *discriminator.output_shape).type(args.dtype).to(args.device)
        fake_ei = torch.zeros(y0.shape[0] * transform.n_trans, *discriminator.output_shape).type(args.dtype).to(args.device)

        valid = Variable(valid, requires_grad=False)
        valid_ei = Variable(valid_ei, requires_grad=False)
        fake_ei = Variable(fake_ei, requires_grad=False)
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images from range input
        x1 = generator(x0)
        y1 = physics.A(x1)

        # TI: x2, x3
        x2 = transform.apply(x1)
        x3 = generator(physics.A_dagger(physics.A(x2)))

        # Loss measures generator's ability to forward consistency and TI
        loss_mc = criterion(y1, y0)
        loss_eq = criterion(x3, x2)

        # Loss measures generator's ability to fool the discriminator
        loss_g = criterion_adversarial(discriminator(x2), valid_ei)

        loss_G = loss_mc + args.alpha * loss_eq + args.beta * loss_g

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = criterion_adversarial(discriminator(x1.detach()), valid)
        fake_loss = criterion_adversarial(discriminator(x2.detach()), fake_ei)


        loss_D = 0.5 * args.beta * (real_loss + fake_loss)

        loss_D.backward()
        optimizer_D.step()


        losses_mc.update(loss_mc.item())
        losses_ei.update(loss_eq.item())

        # measure reconstruction accuracy/error and record loss
        psnr_model.update(cal_psnr(x1, x, max_pixel=1, complex=args.task in ['mri']))
        psnr_fbp.update(cal_psnr(x0, x, max_pixel=1, complex=args.task in ['mri']))
        mse_model.update(cal_mse(x1, x))
        mse_fbp.update(cal_mse(x0, x))

        losses_D.update(loss_D.item())
        losses_G.update(loss_G.item())
        losses_g.update(loss_g.item())


    progress.display(epoch + 1)

    return [meter.avg for meter in meters]

