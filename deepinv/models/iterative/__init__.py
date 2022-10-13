import torch
import torch.nn as nn

from .unroll import Unrolling as unroll


# class Unrolling(nn.Module):
#     def __init__(self, mode, iterations, step_size, weight_tied, backbone_net, device):
#         # todo: tied-> weight_tied; block_arch-> backbone_net; (remove config, just a fixed network/config)
#         # todo: step_size (A.LipCons)
#         # todo: time_step -> iterations
#
#         # todo: env. requirements
#         super(Unrolling, self).__init__()
#         assert mode in ['lfb', 'pgd', 'gd', 'admm']
#
#         # self.name = f'unrolling-{mode}'
#         self.mode = mode
#         self.iterations = iterations
#         # self.step_size = step_size
#         self.register_parameter(name='step_size',
#                                 param=torch.nn.Parameter(torch.tensor(step_size),
#                                                          requires_grad=True))
#         # self.step_size = torch.autograd.Variable(torch.Tensor(step_size), requires_grad=True)
#
#         self.weight_tied = weight_tied
#
#         # block = models.__dict__[backbone_net]
#
#
#         if self.weight_tied:
#             self.blocks = nn.ModuleList([backbone_net.to(device)])
#         else:
#             self.blocks = nn.ModuleList([backbone_net[_].to(device) for _ in range(iterations)])
#
#         # gradient of (least square) data consistency term
#         self.dc_grad = lambda x,y,physics: physics.A_adjoint(y.to(physics.device) - physics.A(x.to(physics.device))).to(x.device)
#
#         #todo: MoDL conjugate GD
#         #todo: self.cong_gd()
#     # def ls_dc_gradient(self, x, y, physics):
#
#     def forward(self, y, physics, x_init=None):
#         x = physics.A_dagger(y).to(y.device) if x_init is None else x_init.clone()
#         input = x
#         for t in range(self.iterations):
#             t = 0 if len(self.blocks)==1 else t
#             if self.mode == 'lfb': # learned forward backward
#                 x = self.blocks[t](torch.cat([x, self.dc_grad(x, y, physics)], dim=1))
#             if self.mode == 'pgd': # proximal gradient descent
#                 x = self.blocks[t](x + self.step_size[t] * self.dc_grad(x, y, physics))
#             if self.mode == 'gd': # gradient descent
#                 x = self.blocks[t](x) + x + self.step_size[t] * self.dc_grad(x, y, physics)
#         if self.mode == 'lfb':
#             x = x[:,:self.block_config['out_channels']]
#             if self.block_config['residual']:
#                 x = x + input
#         return x
