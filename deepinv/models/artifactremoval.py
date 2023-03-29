# import DeepInv
import torch
import torch.nn as nn

# from deepinv import models as models

class ArtifactRemoval(nn.Module):
    '''a.k.a. FBPConvNet (DNN is used to do denoising or artifact removal by having FBP as the input)'''
    def __init__(self, backbone_net, pinv=False, ckpt_path=None, device=None):
        super(ArtifactRemoval, self).__init__()
        self.pinv = pinv
        self.backbone_net = backbone_net

        if ckpt_path is not None:
            self.backbone_net.load_state_dict(torch.load(ckpt_path), strict=True)
            self.backbone_net.eval()

        if type(self.backbone_net).__name__ == 'UNetRes':
            for _, v in self.backbone_net.named_parameters():
                v.requires_grad = False
            self.backbone_net = self.backbone_net.to(device)

    def forward(self, y, physics, **kwargs):
        print(type(self.backbone_net).__name__)
        y_in = physics.A_adjoint(y) if not self.pinv else physics.A_dagger(y)
        if type(self.backbone_net).__name__ == 'UNetRes':
            noise_level_map = torch.FloatTensor(y_in.size(0), 1, y_in.size(2), y_in.size(3)).fill_(kwargs['sigma']).to(y_in.dtype)
            y_in = torch.cat((y_in, noise_level_map), 1)
        return self.backbone_net(y_in)
