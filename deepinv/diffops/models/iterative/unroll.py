# import DeepInv
import torch
import torch.nn as nn

# from deepinv import models as models

class ArtifactRemoval(nn.Module):
    '''a.k.a. FBPConvNet (DNN is used to do denoising or artifact removal by having FBP as the input)'''
    def __init__(self, backbone_net, pinv=False):
        super(ArtifactRemoval, self).__init__()
        self.pinv = pinv
        self.backbone_net = backbone_net

    def forward(self, y, physics):
        return self.backbone_net(physics.A_adjoint(y)) if not self.pinv else self.backbone_net(physics.A_dagger(y))


class Unrolling(nn.Module):
    def __init__(self, backbone_net, mode='pgd', weight_tied=False, step_size=1.0, iterations=5, pinv=False):
        # todo: tied-> weight_tied; block_arch-> backbone_net; (remove config, just a fixed network/config) # done
        # todo: step_size (A.LipCons)
        # todo: time_step -> iterations # done

        # todo: env. requirements
        super(Unrolling, self).__init__()
        assert mode in ['lfb', 'pgd', 'gd', 'admm']

        device = next(backbone_net.parameters()).device
        self.mode = mode
        self.iterations = iterations
        self.register_parameter(name='step_size',
                                param=torch.nn.Parameter(torch.tensor(step_size, device=device),
                                requires_grad=True))
        self.pinv = pinv
        self.weight_tied = weight_tied

        if self.weight_tied:
            self.blocks = torch.nn.ModuleList([backbone_net])
        else:
            self.blocks = torch.nn.ModuleList([backbone_net[_].to(device) for _ in range(iterations)])


        #todo: MoDL conjugate GD
        #todo: self.cong_gd()
    # def ls_dc_gradient(self, x, y, physics):

    def forward(self, y, physics, x_init=None):
        # gradient of (least square) data consistency term
        device = y.device
        #dc_grad = lambda x0, y0: physics.A_adjoint(y0.to(physics.device) - physics.A(x0.to(physics.device)))
        dc_grad = lambda x0, y0: physics.A_adjoint(y0.to(device) - physics.A(x0.to(device)))

        if self.pinv:
            x = physics.A_dagger(y).to(device) if x_init is None else x_init.clone()
        else:
            x = physics.A_adjoint(y).to(device) if x_init is None else x_init.clone()

        input = x

        for t in range(self.iterations):
            r = 0 if len(self.blocks) == 1 else t
            if self.mode == 'lfb':  # learned forward backward
                x = self.blocks[r](torch.cat([x, dc_grad(x, y)], dim=1))
            if self.mode == 'pgd':  # proximal gradient descent
                if r == 0:
                    x = self.blocks[r](x + self.step_size * dc_grad(x, y))
                else:
                    x = self.blocks[r](x + self.step_size[r] * dc_grad(x, y))

            if self.mode == 'gd':  # gradient descent
                if r == 0:
                    x = self.blocks[r](x) + x + self.step_size * dc_grad(x, y)
                else:
                    x = self.blocks[r](x) + x + self.step_size[r] * dc_grad(x, y)

        if self.mode == 'lfb':
            x = x[:, :self.block_config['out_channels']]
            if self.block_config['residual']:
                x = x + input
        return x


# todo: Deep equilibrium? dongdong? # Done (Dec05): /deepinv/diffops/models/iterative/equilibrium.py