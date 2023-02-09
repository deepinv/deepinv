from torch.nn.functional import interpolate as interp
from torchvision.transforms.functional import rotate
import torchvision
import torch.nn.functional as F
import torch
from deepinv.diffops.physics.forward import Physics


def gaussian_blur(sigma=(1, 1), angle=0):
    s = max(sigma)
    c = int(s/0.3+1)
    k_size = 2*c+1

    delta = torch.arange(k_size)

    x, y = torch.meshgrid(delta, delta)
    x = x - c
    y = y - c
    filt = (x/sigma[0]).pow(2)
    filt += (y/sigma[1]).pow(2)
    filt = torch.exp(-filt/2.)

    filt = rotate(filt.unsqueeze(0).unsqueeze(0), angle,
                  interpolation=torchvision.transforms.InterpolationMode.BILINEAR)\
        .squeeze(0).squeeze(0)

    filt = filt/filt.flatten().sum()

    return filt.unsqueeze(0).unsqueeze(0)


class Downsampling(Physics):
    def __init__(self, img_size, factor=2, mode='nearest', antialias=False):
        super().__init__()
        self.mode = mode
        self.scale = factor
        self.imsize = img_size[-2:]
        self.antialias = antialias

    def A(self, x):
        s = interp(x, scale_factor=1/self.scale, mode=self.mode, antialias=self.antialias)
        return s

    def A_adjoint(self, y):
        s = interp(y, size=self.imsize, mode=self.mode, antialias=self.antialias)
        return s


def conv(x, filter, padding):
    b, c, h, w = x.shape

    if padding == 'same':
        h_out = h
        w_out = w
    else:
        h_out = int(h - filter.shape[2] + 1)
        w_out = int(w - filter.shape[3] + 1)

    if filter.shape[1] == 1:
        y = torch.zeros((b, c, h_out, w_out), device=x.device)
        for i in range(b):
            for j in range(c):
                y[i, j, :, :] = F.conv2d(x[i, j, :, :].unsqueeze(0).unsqueeze(1),
                                         filter, padding=padding).unsqueeze(1)
    else:
        y = F.conv2d(x, filter, padding=padding)

    return y


def conv_transpose(y, filter, padding):
    b, c, h, w = y.shape

    if padding == 'same':
        h_out = h
        w_out = w
        p = (int((filter.shape[2] - 1) / 2), int((filter.shape[2] - 1) / 2))
    else:
        h_out = int(h + filter.shape[2] - 1)
        w_out = int(w + filter.shape[3] - 1)
        p = 0

    x = torch.zeros((b, c, h_out, w_out), device=y.device)

    if filter.shape[1] == 1:
        for i in range(b):
            if filter.shape[0] > 1:
                f = filter[i, :, :, :].unsqueeze(0)
            else:
                f = filter

            for j in range(c):
                x[i, j, :, :] = F.conv_transpose2d(y[i, j, :, :].unsqueeze(0).unsqueeze(1),
                                                   f, padding=p).unsqueeze(1)
    else:
        x = F.conv_transpose2d(y, filter, padding=p)

    return x


class BlindBlur(Physics):
    def __init__(self, kernel_size=3, padding='same'):
        '''
        Blind blur operator

        The signal is described by a tuple (x,w) where the first element is the clean image, and the second element
        is the blurring kernel. The measurements y are a tensor representing the convolution of x and w.

        :param kernel_size: (int) maximum support size of the (unknown) blurring kernels
        :param padding:
        '''
        super().__init__()
        self.padding = padding

        if type(kernel_size) is not list or type(kernel_size) is not tuple:
            self.kernel_size = [kernel_size, kernel_size]

    def A(self, s):
        x = s[0]
        w = s[1]
        return conv(x, w, self.padding)

    def A_adjoint(self, y):
        x = y.clone()
        mid_h = int(self.kernel_size[0]/2)
        mid_w = int(self.kernel_size[1]/2)
        w = torch.zeros((y.shape[0], 1, self.kernel_size[0], self.kernel_size[1]))
        w[:, :, mid_h, mid_w] = 1.

        return x, w


class Blur(Physics):
    def __init__(self, filter=gaussian_blur(), padding='same', device='cpu'):
        '''

        Blur operator. Uses torch.conv2d for performing the convolutions

        :param filter: torch.Tensor of size (1, 1, H, W) or (1, C,H,W) containing the blur filter
        :param padding: if 'same' the blurred output has the same size as the image
        if 'valid' the blurred output is smaller than the image (no padding)
        :param device: cpu or cuda
        '''
        super().__init__()
        self.padding = padding
        self.device = device
        self.filter = filter.requires_grad_(False).to(device)

    def A(self, x):
        return conv(x, self.filter, self.padding)

    def A_adjoint(self, y):
        return conv_transpose(y, self.filter, self.padding)

# test code
if __name__ == "__main__":
    device = 'cuda:0'

    import matplotlib.pyplot as plt
    import deepinv as dinv

    x = torchvision.io.read_image('../../../datasets/celeba/img_align_celeba/010214.jpg')
    x = x.unsqueeze(0).float()/256
    x = x[:, :, :128, :128].to(device)

    w = torch.ones((1, 1, 5, 5), device=device)/25

    #physics = BlindBlur(kernel_size=5, padding='same')
    physics = Blur(filter=gaussian_blur(sigma=(5, .1)), padding='same', device=device)
    #physics = Downsampling(factor=8)
    physics.noise_model = dinv.physics.GaussianNoise(sigma=.1)

    #x = [x, w]
    y = physics(x)
    #xhat = physics.A_adjoint(y)

    #xhat = physics.A_dagger(y)
    xhat = physics.prox(y, torch.zeros_like(x), gamma=.1)

    #x = x[0]
    #xhat = xhat[0]

    plt.imshow(x.squeeze(0).permute(1, 2, 0).cpu().numpy())
    plt.show()
    plt.imshow(y.squeeze(0).permute(1, 2, 0).cpu().numpy())
    plt.show()
    plt.imshow(xhat.squeeze(0).permute(1, 2, 0).cpu().numpy())
    plt.show()
