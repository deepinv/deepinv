import torch
from deepinv.diffops.physics.forward import Physics


class Decolorize(Physics):
    r'''
     Colorization forward operator
     Signals must be tensors with 3 colour (RGB) channels, i.e. [*,3,*,*]
     The measurements are grayscale images.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def A(self, x):
        y = x[:, 0, :, :] * 0.2989 + x[:, 1, :, :] * 0.5870 + x[:, 2, :, :] * 0.1140
        return y.unsqueeze(1)

    def A_adjoint(self, y):
        return torch.cat([y*0.2989, y*0.5870, y*0.1140], dim=1)



# test code
if __name__ == "__main__":
    device = 'cuda:0'

    import matplotlib.pyplot as plt
    import torchvision

    x = torchvision.io.read_image('../../../datasets/celeba/img_align_celeba/010214.jpg')
    x = x.unsqueeze(0).float()/256

    pix = 128
    factor = 3

    x = x[:, :, :pix, :pix].to(device)

    physics = Decolorize()

    y = physics(x)

    print(physics.adjointness_test(x))
    print(physics.power_method(x))
    xhat = physics.A_adjoint(y)

    plt.imshow(x.squeeze(0).permute(1, 2, 0).cpu().numpy())
    plt.show()
    plt.imshow(y.squeeze(0).permute(1, 2, 0).cpu().numpy())
    plt.show()
    plt.imshow(xhat.squeeze(0).permute(1, 2, 0).cpu().numpy())
    plt.show()