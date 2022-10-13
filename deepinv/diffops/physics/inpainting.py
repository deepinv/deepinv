import os
import torch

class Inpainting():
    def __init__(self, img_heigth=512, img_width=512, mask_rate=0.3, online=False, mask_path=None, device='cuda:0'):
        self.name = 'inpainting'
        if online:
            self.mask = torch.ones(img_heigth, img_width, device=device)
            self.mask[torch.rand_like(self.mask) > 1 - mask_rate] = 0
            print('the online mask is created...')
        else:
            if mask_path is not None:
                if os.path.exists(mask_path):
                    mask = torch.load(mask_path)
                    self.mask = mask.to(device)
            else:
                mask_path = '/mask_{}x{}_{}.pt'.format(img_width, img_heigth, mask_rate)
                # mask_path = '/deepinv/diffops/physics/mask_{}x{}_{}.pt'.format(img_width, img_heigth, mask_rate)
                #mask_path = '/remote/rds/users/dchen2/DongdongChen_UoE/Code/tmp/pycharm_project_deepinv/deepinv/datasets/mask_{}x{}_{}.pt'.format(img_width, img_heigth, mask_rate)
                self.mask = torch.ones(img_heigth, img_width, device=device)
                self.mask[torch.rand_like(self.mask) > 1 - mask_rate] = 0

                import os
                print('1', os.getcwd()) # /remote/rds/users/dchen2/DongdongChen_UoE/Code/tmp/pycharm_project_deepinv/deepinv/datasets
                print('2', mask_path)

                torch.save(self.mask, mask_path)
                print(f'the mask is created and saved at {mask_path}')

    def A(self, x, new_mask=None):
        return torch.einsum('kl,ijkl->ijkl', self.mask if new_mask is None else new_mask, x)

    def A_dagger(self, x, new_mask=None):
        return torch.einsum('kl,ijkl->ijkl', self.mask if new_mask is None else new_mask, x)

    def A_adjoint(self, x):
        return self.A_dagger(x)

    def masking_y(self, y, new_mask=None):
        return y if new_mask is None else torch.einsum('kl,ijkl->ijkl', new_mask, y)
