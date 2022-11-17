import torch
import torch.nn as nn

def mc_div(x, y, f, tau):
    # y = f(x), avoids double computation
    # computes the divergence of f at x using a montecarlo approx.
    b = torch.randn_like(x)
    y2 = f(x + tau * b)
    div =  (b * (y2 - y)).flatten().mean()/tau
    return div

class SureMCLoss(nn.Module):
    def __init__(self, sigma, tau=0.01):
        super(SureMCLoss, self).__init__()
        self.name='suremc'
        self.sigma2=sigma** 2
        self.tau=tau

    # TODO: leave denoising as default
    def forward(self, y0, y1, physics, f):
        # compute loss_sure
        div = mc_div(y0, y1, lambda x: physics.A(f(x, physics)), self.tau)
        loss_sure = (y1 - y0).pow(2).flatten().mean() - self.sigma2\
                    + 2 * self.sigma2 * div
        return loss_sure


class SURE_Poisson_Loss(nn.Module):
    def __init__(self, gamma, tau, physics):
        super(SURE_Poisson_Loss, self).__init__()
        '''ONLY for Inpainting Task'''
        self.name = 'sure'
        # self.sure_loss_weight = sure_loss_weight
        self.gamma = gamma
        self.tau = tau
        self.physics = physics
        # self.A = lambda x: physics.A(x)
        # self.A_dagger=lambda y: physics.A_dagger(y)

    def forward(self, y0, y1, f):
        # generate a random vector b

        b = torch.rand_like(y0) > 0.5
        # b = torch.rand_like(self.physics.A_dagger(y0)) > 0.5

        b = (2 * b.int() - 1) * 1.0  # binary [-1, 1]
        b = self.physics.A(b * 1.0)
        if self.physics.name in ['mri', 'inpainting']:
            b = self.physics.A(b)

        y2 = self.physics.A(f(self.physics.A_dagger(y0 + self.tau * b)))

        # compute batch size K
        K = y0.shape[0]
        # compute n (dimension of x)
        n = y0.shape[-1] * y0.shape[-2] * y0.shape[-3]

        # compute m (dimension of y)
        if self.physics.name == 'mri':
            m = n / self.physics.acceleration  # dim(y)
        if self.physics.name == 'inpainting':
            m = n * (1 - self.physics.mask_rate)

        loss_sure = torch.sum((y1 - y0).pow(2)) / (K * m) \
                    - self.gamma * y0.sum() / (K * m) \
                    + 2 * self.gamma / (self.tau * K * m) * ((b * y0) * (y2 - y1)).sum()
        return loss_sure


class SURE_MixedPoissonGaussian_Loss(nn.Module):
    def __init__(self, sigma, gamma, tau, physics):
        super(SURE_MixedPoissonGaussian_Loss, self).__init__()
        '''This MPG is ONLY for CT task'''
        self.name = 'sure'
        # self.sure_loss_weight = sure_loss_weight
        self.sigma = sigma
        self.gamma = gamma
        self.tau = tau
        self.physics = physics
        # self.A = lambda x: physics.A(x)
        # self.A_dagger=lambda y: physics.A_dagger(y)

    def forward(self, meas0, meas1, model):
        # CT only, Model must be: model=f(fbp)
        sigma2 = self.sigma ** 2
        b1 = torch.randn_like(meas0)
        b2 = torch.rand_like(meas0) > 0.5
        b2 = (2 * b2.int() - 1) * 1.0  # binary [-1, 1]

        fbp_2 = self.physics.iradon(torch.log(self.physics.I0 / (meas0 + self.tau * b1)))
        fbp_2p = self.physics.iradon(torch.log(self.physics.I0 / (meas0 + self.tau * b2)))
        fbp_2n = self.physics.iradon(torch.log(self.physics.I0 / (meas0 - self.tau * b2)))

        meas2 = self.physics.A(model(fbp_2))
        meas2p = self.physics.A(model(fbp_2p))
        meas2n = self.physics.A(model(fbp_2n))

        K = meas0.shape[0]  # batch size
        m = meas0.shape[-1] * meas0.shape[-2] * meas0.shape[-3]  # dimension of y

        loss_A = torch.sum((meas1 - meas0).pow(2)) / (K * m) - sigma2
        loss_div1 = 2 / (self.tau * K * m) * ((b1 * (self.gamma * meas0 + sigma2)) * (meas2 - meas1)).sum()
        loss_div2 = 2 * sigma2 * self.gamma / (self.tau ** 2 * K * m) \
                    * (b2 * (meas2p + meas2n - 2 * meas1)).sum()

        loss_sure = loss_A + loss_div1 + loss_div2
        return loss_sure

# class SURELoss(nn.Module):
#     def __init__(self, sure_loss_weight):
#         super(SURELoss, self).__init__()
#         self.name='sure'
#         self.sure_loss_weight = sure_loss_weight
#
#     def gaussian(self, net, physics, x0, y0, y1, sigma, tau):
#         sigma2 = sigma ** 2
#         b = torch.randn_like(x0)
#         b = physics.A(b)
#
#         y2 = physics.A(net(physics.A_dagger(y0 + tau * b)))
#
#         # compute batch size K
#         K = y0.shape[0]
#         # compute n (dimension of x)
#         n = y0.shape[-1] * y0.shape[-2] * y0.shape[-3]
#
#         # compute m (dimension of y)
#         if physics.name == 'mri':
#             m = n / physics.acceleration  # dim(y)
#         if physics.name == 'inpainting':
#             m = n * (1 - physics.mask_rate)
#
#         # compute loss_sure
#         loss_sure = torch.sum((y1 - y0).pow(2)) / (K * m) - sigma2 \
#                     + (2 * sigma2 / (tau * m * K)) * (b * (y2 - y1)).sum()
#
#         return self.sure_loss_weight * loss_sure
#
#     def poisson(self, net, physics, x0, y0, y1, gamma, tau):
#         # generate a random vector b
#         b = torch.rand_like(y0) > 0.5
#         b = (2 * b.int() - 1) * 1.0  # binary [-1, 1]
#         b = physics.A(b * 1.0)
#         if physics.name in ['mri', 'inpainting']:
#             b = physics.A(b)
#
#         y2 = physics.A(net(physics.A_dagger(y0 + tau * b)))
#
#         # compute batch size K
#         K = y0.shape[0]
#         # compute n (dimension of x)
#         n = y0.shape[-1] * y0.shape[-2] * y0.shape[-3]
#
#         # compute m (dimension of y)
#         if physics.name == 'mri':
#             m = n / physics.acceleration  # dim(y)
#         if physics.name == 'inpainting':
#             m = n * (1 - physics.mask_rate)
#
#         loss_sure = torch.sum((y1 - y0).pow(2)) / (K * m) \
#                     - gamma * y0.sum() / (K * m) \
#                     + 2 * gamma / (tau * K * m) * ((b * y0) * (y2 - y1)).sum()
#
#         return self.sure_loss_weight * loss_sure
#
#     def mixed_poisson_gaussian(self, net, physics, x0, meas0, meas1, sigma, gamma, tau):
#         # CT only
#         sigma2 = sigma ** 2
#         b1 = torch.randn_like(meas0)
#         b2 = torch.rand_like(meas0) > 0.5
#         b2 = (2 * b2.int() - 1) * 1.0  # binary [-1, 1]
#
#         fbp_2 = physics.iradon(torch.log(physics.I0 / (meas0 + tau * b1)))
#         fbp_2p = physics.iradon(torch.log(physics.I0 / (meas0 + tau * b2)))
#         fbp_2n = physics.iradon(torch.log(physics.I0 / (meas0 - tau * b2)))
#
#         meas2 = physics.A(net(fbp_2))
#         meas2p = physics.A(net(fbp_2p))
#         meas2n = physics.A(net(fbp_2n))
#
#         K = meas0.shape[0]  # batch size
#         m = meas0.shape[-1] * meas0.shape[-2] * meas0.shape[-3]  # dimension of y
#
#         loss_A = torch.sum((meas1 - meas0).pow(2)) / (K * m) - sigma2
#         loss_div1 = 2 / (tau * K * m) * ((b1 * (gamma * meas0 + sigma2)) * (meas2 - meas1)).sum()
#         loss_div2 = 2 * sigma2 * gamma / (tau ** 2 * K * m) \
#                     * (b2 * (meas2p + meas2n - 2 * meas1)).sum()
#
#         loss_sure = loss_A + loss_div1 + loss_div2
#         return self.sure_loss_weight * loss_sure
#
#     def forward(self):
#         print("'Gaussian', 'Poisson' and 'MixedPoissonGaussian'")
#         return
# if __name__ == '__main__':
#     sure = SURELoss(tau=1e-3, sure_loss_weight=1e-2)