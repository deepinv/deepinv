from deepinv.physics.forward import DecomposablePhysics
import torch


class Inpainting(DecomposablePhysics):
    def __init__(self, tensor_size, mask=0.3, device='cuda:0', **kwargs):
        super().__init__(**kwargs)
        self.name = 'inpainting'
        self.tensor_size = tensor_size

        if isinstance(mask, torch.Tensor): # check if the user created mask
            self.mask = mask
        else: # otherwise create new random mask
            mask_rate = mask
            self.mask = torch.ones(tensor_size, device=device)
            self.mask[torch.rand_like(self.mask) > mask_rate] = 0

        self.mask = torch.nn.Parameter(self.mask.unsqueeze(0), requires_grad=False)
