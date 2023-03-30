import torch
import torch.nn as nn

# --------------------------------------------
# Jacobian Spectral Norm loss
# --------------------------------------------
class JacobianSpectralNorm(nn.Module):
    r'''
        Jacobian spectral norm from Pesquet et al;

        Computes the spectral norm of Q = 2J-I where J is the denoising model.
        TODO description

    '''
    def __init__(self):
        super().__init__()

    def forward(self, y_in, x_hat, denoiser, sigma, interpolation=False, training=False):
        r'''
        BEWARE: reversed usage compared to Pesquet et al: it is now called as (y, x_hat) and not (x_hat, y) !

        TODO: explain parameters
        :param torch.tensor y_in:
        :param torch.tensor x_hat:
        :param denoiser:
        :param float sigma:
        :param bool interpolation:
        :param bool training:
        :return:

        '''

        # FOR REFERENCE
        # out_detached = out.detach().type(Tensor)
        # true_detached = data_true.detach().type(Tensor)
        #
        # tau = torch.rand(true_detached.shape[0], 1, 1, 1).type(Tensor)
        # out_detached = tau * out_detached + (1 - tau) * true_detached
        # out_detached.requires_grad_()
        #
        # out_reg = model(out_detached)
        #
        # out_net_reg = 2. * out_reg - out_detached
        # reg_loss = reg_fun(out_detached, out_net_reg)

        if interpolation:
            eta = torch.rand(y_in.size(0), 1, 1, 1, requires_grad=True).to(self.device)
            x = eta * y_in.detach() + (1 - eta) * x_hat.detach()
            x = x.to(self.device)
        else:
            x = y_in

        x.requires_grad_()
        x_hat, _ = denoiser(x, sigma)

        y = 2.*x_hat-y_in  # Beware notation : y_in = input, x_hat = output network

        u = torch.randn_like(x)
        u = u / torch.norm(u, p=2)

        z_old = torch.zeros(u.shape[0])

        for it in range(self.hparams.power_method_nb_step):

            w = torch.ones_like(y, requires_grad=True)  # Double backward trick. From https://gist.github.com/apaszke/c7257ac04cb8debb82221764f6d117ad
            v = torch.autograd.grad(torch.autograd.grad(y, x, w, create_graph=True), w, u, create_graph=training)[0]  # Ju

            v, = torch.autograd.grad(y, x, v, retain_graph=True, create_graph=True)  # vtJt

            z = torch.matmul(u.reshape(u.shape[0], 1, -1), v.reshape(v.shape[0], -1, 1)) / torch.matmul(
                u.reshape(u.shape[0], 1, -1), u.reshape(u.shape[0], -1, 1))

            if it > 0:
                rel_var = torch.norm(z - z_old)
                if rel_var < self.hparams.power_method_error_threshold:
                    break
            z_old = z.clone()

            u = v / torch.norm(v, p=2)  # Modified

            if self.eval:
                w.detach_()
                v.detach_()
                u.detach_()

        return z.view(-1)
