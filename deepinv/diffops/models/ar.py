import torch.nn as nn

class ArtifactRemoval(nn.Module):
    '''a.k.a. FBPConvNet (DNN is used to do denoising or artifact removal by having FBP as the input)'''
    def __init__(self, backbone_net, pinv=False):
        super(ArtifactRemoval, self).__init__()
        self.pinv = pinv
        self.backbone_net = backbone_net

    def forward(self, y, physics):
        return self.backbone_net(physics.A_adjoint(y)) if self.pinv else self.backbone_net(physics.A_dagger(y))