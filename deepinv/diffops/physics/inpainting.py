import os
import torch

class GaussianNoise(torch.nn.Module): # parent class for forward models
    def __init__(self, std=.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        return x + torch.randn_like(x)*self.std


class Forward(torch.nn.Module): # parent class for forward models
    def __init__(self, A = lambda x: x, A_adjoint = lambda x: x, noise_model = GaussianNoise(std=0.2)):
        super().__init__()
        self.noise_model = noise_model
        self.forw = A
        self.adjoint = A_adjoint

    def __add__(self, other):
        A = lambda x: self.A(other.A(x))
        A_adjoint = lambda x: self.A_adjoint(other.A_adjoint(x))
        noise = self.noise_model
        return Forward(A, A_adjoint, noise)

    def forward(self, x): # degrades signal
        return self.noise(self.A(x))

    def A(self, x):
        return self.forw(x)

    def noise(self, x):
        return self.noise_model(x)

    def A_adjoint(self, x):
        return self.adjoint(x)

    def A_dagger(self, x): # degrades signal
        # USE Conjugate gradient here as default option
        return self.A_adjoint(x)


class Inpainting(Forward):
    def __init__(self, tensor_size, mask=0.3, save=False, device='cuda:0'):
        super().__init__()
        self.name = 'inpainting'
        self.tensor_size = tensor_size

        if isinstance(mask, torch.Tensor): # check if the user created mask
            self.mask = mask
        else: # otherwise create new random mask
            mask_rate = mask
            if not save:
                self.mask = torch.ones(tensor_size, device=device)
                self.mask[torch.rand_like(self.mask) > mask_rate] = 0
            else:
                root_path = './' #root_path = '/remote/rds/users/dchen2/DongdongChen_UoE/Code/tmp/pycharm_project_deepinv/deepinv/datasets/'
                mask_path = root_path + 'mask_{}x{}_{}.pt'.format(tensor_size[0], tensor_size[1], mask_rate)
                if os.path.exists(mask_path):
                    mask = torch.load(mask_path)
                    self.mask = mask.to(device)
                    print(f'the mask is loaded from:\n{mask_path}')
                else:
                    self.mask = torch.ones(tensor_size, device=device)
                    self.mask[torch.rand_like(self.mask) >  mask_rate] = 0
                    torch.save(self.mask, mask_path)
                    print(f'the mask is created and saved at:\n{mask_path}')

    def A(self, x):
        return self.mask * x

    def A_dagger(self, x):
        return self.mask * x

    def A_adjoint(self, x):
        return self.A_dagger(x)



# Below is dumpy


def get_group_inpainting_ops(mask_rate, img_heigth, img_width, device, G=1, division_mask=False, division_mask_rate=0.5):
    # print('1 division_mask=',division_mask)

    # print('222', device)

    if G > 1 and G<=100: # maximum G=100
        ipt_group=[]
        for g in range(G):
            if img_heigth == 256:

                mask_path = f'/home/dchen2/RDS/DongdongChen_UoE/Code/DATASET/inpainting_mask/mask_random{mask_rate}_G{100}_g{g}.pt'

            if img_heigth==128:
                mask_path = f'/home/dchen2/RDS/DongdongChen_UoE/Code/DATASET/inpainting_mask/mask_128x128_random{mask_rate}_G{100}_g{g}.pt'

            if img_heigth==64:
                mask_path = f'/home/dchen2/RDS/DongdongChen_UoE/Code/DATASET/inpainting_mask/mask_64x64_random{mask_rate}_G{100}_g{g}.pt'

            if not os.path.exists(mask_path):
            #     mask = torch.load(mask_path).to(device)
            # else:
            #     print(g)

                mask = torch.ones(img_heigth, img_width, device=device)
                mask[torch.rand_like(mask) > 1 - mask_rate] = 0
                torch.save(mask, mask_path)
            # print(g, ' division_mask=', division_mask)
                print(f'new...{mask_path}')

            ipt_group.append(Inpainting(img_heigth, img_width, mask_rate, device, False, g, G, division_mask, division_mask_rate))

        return ipt_group
    else:
        return Inpainting(img_heigth, img_width, mask_rate, device, False, 1, 1, division_mask, division_mask_rate)







class Inpainting_fixed():
    def __init__(self, img_heigth=512, img_width=512, mask_rate=0.3, device='cuda:0', online=False, g=None, G=None, division_mask=False, division_mask_rate=0.5):

        self.name = 'inpainting'
        # mask_path = './physics/mask_random{}.pt'.format(mask_rate)
        # print('333', device)

        # print('kk\t', img_width, img_heigth, device)

        if img_width==512:
            if g is not None or G is not None:
                G=100 # first G will be selected
                mask_path = f'/home/dchen2/RDS/DongdongChen_UoE/Code/DATASET/inpainting_mask/mask_512x512_random{mask_rate}_G{100}_g{g}.pt'
            else:
                mask_path = '/home/dchen2/RDS/DongdongChen_UoE/Code/DATASET/mask_512x512_random{}.pt'.format(mask_rate)


        if img_width==256:
            if g is not None or G is not None:
                G=100 # first G will be selected
                # mask_path = f'/home/dchen2/RDS/DongdongChen_UoE/Code/DATASET/inpainting_mask/mask_random{mask_rate}_G{100}_g{g}.pt'
                mask_path = f'/home/dchen2/RDS/DongdongChen_UoE/Code/DATASET/inpainting_mask/mask_256x256_random{mask_rate}_G{100}_g{g}.pt'
            else:
                # mask_path = '/home/dchen2/RDS/DongdongChen_UoE/Code/DATASET/mask_random{}.pt'.format(mask_rate)
                mask_path = '/home/dchen2/RDS/DongdongChen_UoE/Code/DATASET/mask_256x256_random{}.pt'.format(mask_rate)

        if img_width==128:
            if g is not None or G is not None:
                G=100 # first G will be selected
                mask_path = f'/home/dchen2/RDS/DongdongChen_UoE/Code/DATASET/inpainting_mask/mask_128x128_random{mask_rate}_G{100}_g{g}.pt'
            else:
                mask_path = '/home/dchen2/RDS/DongdongChen_UoE/Code/DATASET/mask_128x128_random{}.pt'.format(mask_rate)


        if img_width == 64:
            if g is not None or G is not None:
                G=100 # first G will be selected
                mask_path = f'/home/dchen2/RDS/DongdongChen_UoE/Code/DATASET/inpainting_mask/mask_64x64_random{mask_rate}_G{100}_g{g}.pt'
            else:
                mask_path = '/home/dchen2/RDS/DongdongChen_UoE/Code/DATASET/mask_64x64_random{}.pt'.format(mask_rate)



            # mask_path = '/home/dchen2/RDS/DongdongChen_UoE/Code/DATASET/mask_64x64_random{}.pt'.format(
            #     mask_rate)

        if not online:
            if os.path.exists(mask_path):
                # print('kk 1', device)
                # print('kk 2\t', mask_path)

                mask = torch.load(mask_path)
                # print('kk 3\t', mask.shape, mask.device)
                self.mask = mask.to(device)
                # print('kk 4\t self.mask', self.mask.shape, self.mask.device)
            else:
                self.mask = torch.ones(img_heigth, img_width, device=device)
                self.mask[torch.rand_like(self.mask) > 1 - mask_rate] = 0
                torch.save(self.mask, mask_path)
        else:
            self.mask = torch.ones(img_heigth, img_width, device=device)
            self.mask[torch.rand_like(self.mask) > 1 - mask_rate] = 0

        # print('2 division_mask=', division_mask)
        self.division_mask=division_mask
        self.division_mask_rate = division_mask_rate

        if division_mask:
            # # print('I am here...')
            # mask_left = torch.ones_like(self.mask) #256x256 all ones
            # mask_left[torch.rand_like(mask_left) >= division_mask_rate] = 0
            #
            # mask_right = torch.ones_like(self.mask) - mask_left
            # self.mask_left = mask_left
            # self.mask_right = mask_right
            #



            path_mask_left = f'/home/dchen2/RDS/DongdongChen_UoE/Code/DATASET/inpainting_mask/A2A_LEFT_mask_128x128_random{mask_rate}_G{100}_g{g}.pt'
            path_mask_right = f'/home/dchen2/RDS/DongdongChen_UoE/Code/DATASET/inpainting_mask/A2A_RIGHT_mask_128x128_random{mask_rate}_G{100}_g{g}.pt'

            if not (os.path.exists(path_mask_left) and os.path.exists(path_mask_right)):
                mask_left = torch.ones_like(self.mask)  # 256x256 all ones
                mask_left[torch.rand_like(mask_left) >= division_mask_rate] = 0

                mask_right = torch.ones_like(self.mask) - mask_left
                self.mask_left = mask_left
                self.mask_right = mask_right

                torch.save({'mask_left': mask_left}, path_mask_left)
                torch.save({'mask_right': mask_right}, path_mask_right)
                print('mask_left and mask_right are created and saved...')

            else:
                mask_left = torch.load(path_mask_left, map_location=device)['mask_left']
                mask_right = torch.load(path_mask_right, map_location=device)['mask_right']

                self.mask_left = mask_left.to(device)
                self.mask_right = mask_right.to(device)
                print('mask_left and mask_right are LOADED...')


    def A(self, x, new_mask=None):
        # print(x.shape, self.mask.shape)
        return torch.einsum('kl,ijkl->ijkl', self.mask if new_mask is None else new_mask, x)

    def A_dagger(self, x, new_mask=None):
        return torch.einsum('kl,ijkl->ijkl', self.mask if new_mask is None else new_mask, x)

    def A_adjoint(self, x):
        return self.A_dagger(x)

    def masking_y(self, y, new_mask=None):
        masked_y = torch.einsum('kl,ijkl->ijkl', new_mask, y)
        return masked_y

    def update_division_mask(self):
        if self.division_mask:
            # print('I am here...')
            mask_left = torch.ones_like(self.mask) #256x256 all ones
            mask_left[torch.rand_like(mask_left) >= self.division_mask_rate] = 0

            mask_right = torch.ones_like(self.mask) - mask_left
            self.mask_left = mask_left
            self.mask_right = mask_right
    #
    # def division_mask(self):
    #     mask_left = torch.ones_like(self.mask)
    #     mask_left[torch.rand_like(mask_left) >= 0.5] = 0
    #
    #     mask_right = torch.ones_like(self.mask) - mask_left
    #     self.mask_left = mask_left
    #     self.mask_right = mask_right


    # def A_left(self, x):
    #     return torch.einsum('kl,ijkl->ijkl', self.mask_left)
    #
    # def A_right(self, x):
    #     return torch.einsum('kl,ijkl->ijkl', self.mask_right)

# dumpy
class Inpainting_NN(torch.nn.Module):
    # def __init__(self):
    def __init__(self, img_heigth=512, img_width=512, mask_rate=0.3, device='cuda:0', online=False, g=None, G=None):
        super(Inpainting_NN, self).__init__()
        self.name = 'inpainting'
        # mask_path = './physics/mask_random{}.pt'.format(mask_rate)
        if img_width==256:
            if g is not None or G is not None:
                G=100 # first G will be selected
                mask_path = f'/home/dchen2/RDS/DongdongChen_UoE/Code/DATASET/inpainting_mask/mask_random{mask_rate}_G{100}_g{g}.pt'
            else:
                mask_path = '/home/dchen2/RDS/DongdongChen_UoE/Code/DATASET/mask_random{}.pt'.format(mask_rate)

        if img_width==128:
            if g is not None or G is not None:
                G=100 # first G will be selected
                mask_path = f'/home/dchen2/RDS/DongdongChen_UoE/Code/DATASET/inpainting_mask/mask_128x128_random{mask_rate}_G{100}_g{g}.pt'
            else:
                mask_path = '/home/dchen2/RDS/DongdongChen_UoE/Code/DATASET/mask_128x128_random{}.pt'.format(mask_rate)


        if img_width == 64:
            if g is not None or G is not None:
                G=100 # first G will be selected
                mask_path = f'/home/dchen2/RDS/DongdongChen_UoE/Code/DATASET/inpainting_mask/mask_64x64_random{mask_rate}_G{100}_g{g}.pt'
            else:
                mask_path = '/home/dchen2/RDS/DongdongChen_UoE/Code/DATASET/mask_64x64_random{}.pt'.format(mask_rate)



            # mask_path = '/home/dchen2/RDS/DongdongChen_UoE/Code/DATASET/mask_64x64_random{}.pt'.format(
            #     mask_rate)

        if not online:
            if os.path.exists(mask_path):
                self.mask = torch.load(mask_path).to(device)
            else:
                self.mask = torch.ones(img_heigth, img_width, device=device)
                self.mask[torch.rand_like(self.mask) > 1 - mask_rate] = 0
                torch.save(self.mask, mask_path)
        else:
            self.mask = torch.ones(img_heigth, img_width, device=device)
            self.mask[torch.rand_like(self.mask) > 1 - mask_rate] = 0

    def A(self, x):
        return torch.einsum('kl,ijkl->ijkl', self.mask, x)

    def A_dagger(self, x):
        return torch.einsum('kl,ijkl->ijkl', self.mask, x)

    def A_adjoint(self, x):
        return self.A_dagger(x)




if __name__ == '__main__':
    # opts = get_group_inpainting_ops(0.3, 256, 256, 'cuda:0', G=100)
    # opts = get_group_inpainting_ops(0.5, 256, 256, 'cuda:0', G=100)
    # opts = get_group_inpainting_ops(0.8, 256, 256, 'cuda:0', G=100)

    # physics = opts[0]

    import matplotlib.pyplot as plt
    from utils.plot import plot_imgs_light_no_args, torch2np

    # physics = Inpainting(img_heigth=64, img_width=64, mask_rate=0.3, device='cuda:1',
    #                      online=False, g=1, G=40, division_mask=True)


    physics = Inpainting(img_heigth=28, img_width=28, mask_rate=0.3, device='cuda:1',
                         online=False)

    # plt.subplot(1,4,1)
    plt.imshow(torch2np(physics.mask), cmap='gray')
    plt.title('mask')
    #
    # plt.subplot(1,4,1)
    # plt.imshow(torch2np(physics.mask), cmap='gray')
    # plt.title('mask')
    #
    # plt.subplot(1,4,2)
    # plt.imshow(torch2np(physics.mask_left), cmap='gray')
    # plt.title('mask_left')
    #
    # plt.subplot(1,4,3)
    # plt.imshow(torch2np(physics.mask_right), cmap='gray')
    # plt.title('mask_right')
    #
    # plt.subplot(1,4,4)
    # plt.imshow(torch2np(physics.mask_left+physics.mask_right), cmap='gray')
    # plt.title('mask_left+mask_right')

    plt.show()


