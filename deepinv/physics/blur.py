from torchvision.transforms.functional import rotate
import torchvision
import torch.nn.functional as F
import torch
import numpy as np
import torch.fft as fft
from deepinv.physics.forward import Physics, LinearPhysics, DecomposablePhysics


def filter_fft(filter, img_size):
    ph = int((filter.shape[2] - 1) / 2)
    pw = int((filter.shape[3] - 1) / 2)

    filt2 = torch.zeros(filter.shape[:2] + img_size[-2:], device=filter.device)

    filt2[:, :filter.shape[1], :filter.shape[2], :filter.shape[3]] = filter
    filt2 = torch.roll(filt2, shifts=(-ph, -pw), dims=(2, 3))

    return fft.fft2(filt2)


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


def bilinear_filter(factor=2):
    x = np.arange(start=-factor + .5, stop=factor, step=1)/factor
    w = 1 - np.abs(x)
    w = np.outer(w, w)
    w = w/np.sum(w)
    return torch.Tensor(w).unsqueeze(0).unsqueeze(0)


def bicubic_filter(factor=2):
    x = np.arange(start=-2*factor + .5, stop=2*factor, step=1)/factor
    a = -.5
    x = np.abs(x)
    w = ((a + 2) * np.power(x, 3) - (a + 3) * np.power(x, 2) + 1) * (x <= 1)
    w += (a * np.power(x, 3) - 5 * a * np.power(x, 2) + 8 * a * x - 4 * a) * (x > 1) * (x < 2)
    w = np.outer(w, w)
    w = w/np.sum(w)
    return torch.Tensor(w).unsqueeze(0).unsqueeze(0)


# TODO: fix bilinear filter
class Downsampling(LinearPhysics):
    r'''
    Downsampling operator for super-resolution problems.

    It is defined as

    .. math::

        y = S (h*x)

    where :math:`h` is a low-pass filter and :math:`S` is a subsampling operator.

    :param img_size: (tuple of ints), size of the input image
    :param int factor: downsampling factor
    :param torch.tensor filter: custom filter.
    :param str mode: downsampling mode. Can be 'gauss', 'bilinear' or 'bicubic'
    :param int, tuple of ints sigma_gauss: standard deviation of the gaussian kernel. If int, the same value is used for
        both dimensions. If None, the standard deviation is set to the downsampling factor.
    :param str padding: options are ``'valid'``, ``'circular'``, ``'replicate'`` and ``'reflect'``.
        If ``padding='valid'`` the blurred output is smaller than the image (no padding)
        otherwise the blurred output has the same size as the image.

    '''

    def __init__(self, img_size, factor=2, filter=None, mode=None, sigma_gauss=None, device='cpu', padding='circular', **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        self.imsize = img_size
        self.padding = padding
        self.mode = mode
        if filter:
            self.filter = filter
        elif mode:
            if mode == 'gauss':
                if sigma_gauss is None:
                    sigma_gauss_x = factor
                    sigma_gauss_y = factor
                elif isinstance(sigma_gauss, int):
                    sigma_gauss_x = sigma_gauss
                    sigma_gauss_y = sigma_gauss
                elif isinstance(sigma_gauss, tuple):
                    sigma_gauss_x = sigma_gauss[0]
                    sigma_gauss_y = sigma_gauss[1]
                else:
                    raise Exception("sigma_gauss should be an int or a tuple of ints")
                self.filter = gaussian_blur(sigma=(sigma_gauss_x, sigma_gauss_y)).requires_grad_(False).to(device)
            elif mode == 'bilinear':
                self.filter = bilinear_filter(self.factor).requires_grad_(False).to(device)
            elif mode == 'bicubic':
                self.filter = bicubic_filter(self.factor).requires_grad_(False).to(device)
            else:
                raise Exception("The downsampling mode chosen doesn't exist")
        else:
            self.filter = None

        assert int(factor) == factor and factor > 1, 'downsampling factor should be a positive integer bigger than 1'
        if self.filter :
            self.Fh = filter_fft(self.filter, x.shape, real=False, device=device)
            self.Fhc = torch.conj(self.Fh)
            self.Fh2 = torch.abs(self.Fhc*self.Fh)

    def A(self, x):
        if self.filter:
            out = conv(x, self.filter, padding=self.padding)
        else:
            out = x

        y = out[:, :, ::self.factor, ::self.factor]  # downsample

        return y

    def A_adjoint(self, y):

        x = torch.zeros((y.shape[0],) + self.imsize, device=y.device)

        x[:, :, ::self.factor, ::self.factor] = y  # upsample

        if self.filter:
            x = conv_transpose(x, self.filter, padding=self.padding)

        return x

    def prox_l2(self, z, y, gamma):
        r'''
        If the padding is circular, it computes the proximal operator with the closed-formula of https://arxiv.org/abs/1510.00143.

        Otherwise, it computes it using the conjugate gradient algorithm which can be slow if applied many times.

        '''

        if self.padding == 'circular': # Formula from (Zhao, 2016)

            z_hat = gamma*self.A_adjoint(y) + z
            Fz_hat = fft.fft2(z_hat)

            def splits(a, sf):
                '''split a into sfxsf distinct blocks
                Args:
                    a: NxCxWxH
                    sf: split factor
                Returns:
                    b: NxCx(W/sf)x(H/sf)x(sf^2)
                '''
                b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
                b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
                return b

            top = torch.mean(splits(self.Fh*Fz_hat, self.factor),dim=-1)
            below = gamma*torch.mean(splits(self.Fh2, self.factor),dim=-1) + 1
            rc = self.Fhc * (top / below).repeat(1, 1, self.factor, self.factor)
            r = torch.real(fft.ifft2(rc))
            return z_hat - r
        else:
            return LinearPhysics.prox_l2(self, z, y, gamma)


def extend_filter(filter):
    b, c, h, w = filter.shape
    w_new = w
    h_new = h

    offset_w = 0
    offset_h = 0

    if w == 1:
        w_new = 3
        offset_w = 1
    elif w % 2 == 0:
        w_new += 1

    if h == 1:
        h_new = 3
        offset_h = 1
    elif h % 2 == 0:
        h_new += 1

    out = torch.zeros((b, c, h_new, w_new), device=filter.device)
    out[:, :, offset_h:h+offset_h, offset_w:w+offset_w] = filter
    return out


def conv(x, filter, padding):
    r'''
        Convolution of x and filter. The transposed of this operation is conv_transpose(x, filter, padding)

        :param x: (torch.Tensor) Image of size (B,C,W,H).
        :param filter: (torch.Tensor) Filter of size (1,C,W,H) for colour filtering or (1,1,W,H) for filtering each channel with the same filter.
        :param padding: (string) options = 'valid','circular','replicate','reflect'. If padding='valid' the blurred output is smaller than the image (no padding), otherwise the blurred output has the same size as the image.

    '''
    b, c, h, w = x.shape

    filter = filter.flip(-1).flip(-2) # In order to perform convolution and not correlation like Pytorch native conv

    filter = extend_filter(filter)

    ph = (filter.shape[2] - 1)/2
    pw = (filter.shape[3] - 1)/2


    if padding == 'valid':
        h_out = int(h - 2*ph)
        w_out = int(w - 2*pw)
    else:
        h_out = h
        w_out = w
        pw = int(pw)
        ph = int(ph)
        x = F.pad(x, (pw, pw, ph, ph), mode=padding, value=0)

    if filter.shape[1] == 1:
        y = torch.zeros((b, c, h_out, w_out), device=x.device)
        for i in range(b):
            for j in range(c):
                y[i, j, :, :] = F.conv2d(x[i, j, :, :].unsqueeze(0).unsqueeze(1),
                                         filter, padding='valid').unsqueeze(1)
    else:
        y = F.conv2d(x, filter, padding='valid')

    return y


def conv_transpose(y, filter, padding):
    r'''
        Tranposed convolution of x and filter. The transposed of this operation is conv(x, filter, padding)

        :param torch.tensor x: Image of size (B,C,W,H).
        :param torch.tensor filter: Filter of size (1,C,W,H) for colour filtering or (1,C,W,H) for filtering each channel with the same filter.
        :param str padding: options are ``'valid'``, ``'circular'``, ``'replicate'`` and ``'reflect'``.
            If ``padding='valid'`` the blurred output is smaller than the image (no padding)
            otherwise the blurred output has the same size as the image.
    '''

    b, c, h, w = y.shape

    filter = filter.flip(-1).flip(-2) # In order to perform convolution and not correlation like Pytorch native conv

    filter = extend_filter(filter)

    ph = (filter.shape[2] - 1)/2
    pw = (filter.shape[3] - 1)/2


    h_out = int(h + 2 * ph)
    w_out = int(w + 2 * pw)
    pw = int(pw)
    ph = int(ph)

    x = torch.zeros((b, c, h_out, w_out), device=y.device)
    if filter.shape[1] == 1:
        for i in range(b):
            if filter.shape[0] > 1:
                f = filter[i, :, :, :].unsqueeze(0)
            else:
                f = filter

            for j in range(c):
                x[i, j, :, :] = F.conv_transpose2d(y[i, j, :, :].unsqueeze(0).unsqueeze(1), f)
    else:
        x = F.conv_transpose2d(y, filter)

    if padding == 'valid':
        out = x
    elif padding == 'zero':
        out = x[:, :, ph:-ph, pw:-pw]
    elif padding == 'circular':
        out = x[:, :, ph:-ph, pw:-pw]
        # sides
        out[:, :, :ph, :] += x[:, :, -ph:, pw:-pw]
        out[:, :, -ph:, :] += x[:, :, :ph, pw:-pw]
        out[:, :, :, :pw] += x[:, :, ph:-ph, -pw:]
        out[:, :, :, -pw:] += x[:, :, ph:-ph, :pw]
        # corners
        out[:, :, :ph, :pw] += x[:, :, -ph:, -pw:]
        out[:, :, -ph:, -pw:] += x[:, :, :ph, :pw]
        out[:, :, :ph, -pw:] += x[:, :, -ph:, :pw]
        out[:, :, -ph:, :pw] += x[:, :, :ph, -pw:]

    elif padding == 'reflect':
        out = x[:, :, ph:-ph, pw:-pw]
        # sides
        out[:, :, 1:1+ph, :] += x[:, :, :ph, pw:-pw].flip(dims=(2,))
        out[:, :, -ph-1:-1, :] += x[:, :, -ph:, pw:-pw].flip(dims=(2,))
        out[:, :, :, 1:1+pw] += x[:, :, ph:-ph, :pw].flip(dims=(3,))
        out[:, :, :, -pw-1:-1] += x[:, :, ph:-ph, -pw:].flip(dims=(3,))
        # corners
        out[:, :, 1:1+ph, 1:1+pw] += x[:, :, :ph, :pw].flip(dims=(2, 3))
        out[:, :, -ph-1:-1, -pw-1:-1] += x[:, :, -ph:, -pw:].flip(dims=(2, 3))
        out[:, :, -ph-1:-1, 1:1+pw] += x[:, :, -ph:, :pw].flip(dims=(2, 3))
        out[:, :, 1:1+ph, -pw-1:-1] += x[:, :, :ph, -pw:].flip(dims=(2, 3))

    elif padding == 'replicate':
        out = x[:, :, ph:-ph, pw:-pw]
        # sides
        out[:, :, 0, :] += x[:, :, :ph, pw:-pw].sum(2)
        out[:, :, -1, :] += x[:, :, -ph:, pw:-pw].sum(2)
        out[:, :, :, 0] += x[:, :, ph:-ph, :pw].sum(3)
        out[:, :, :, -1] += x[:, :, ph:-ph, -pw:].sum(3)
        # corners
        out[:, :, 0, 0] += x[:, :, :ph, :pw].sum(3).sum(2)
        out[:, :, -1, -1] += x[:, :, -ph:, -pw:].sum(3).sum(2)
        out[:, :, -1, 0] += x[:, :, -ph:, :pw].sum(3).sum(2)
        out[:, :, 0, -1] += x[:, :, :ph, -pw:].sum(3).sum(2)
    return out


class BlindBlur(Physics):
    r'''
    Blind blur operator.

    If performs

    .. math::

        y = w*x

    where :math:`*` denotes convolution and :math:`w` is an unknown filter.
    This class uses ``torch.conv2d`` for performing the convolutions.

    The signal is described by a tuple (x,w) where the first element is the clean image, and the second element
    is the blurring kernel. The measurements y are a tensor representing the convolution of x and w.

    :param int kernel_size: maximum support size of the (unknown) blurring kernels.
    :param str padding: options are ``'valid'``, ``'circular'``, ``'replicate'`` and ``'reflect'``.
        If ``padding='valid'`` the blurred output is smaller than the image (no padding)
        otherwise the blurred output has the same size as the image.

    '''
    def __init__(self, kernel_size=3, padding='circular', **kwargs):
        super().__init__(**kwargs)
        self.padding = padding

        if type(kernel_size) is not list or type(kernel_size) is not tuple:
            self.kernel_size = [kernel_size, kernel_size]

    def A(self, s):
        r'''

        :param tuple, list x: Tuple containing two torch.tensor, x[0] with the image and x[1] with the filter.
        :return: (torch.tensor) blurred measurement.
        '''
        x = s[0]
        w = s[1]
        return conv(x, w, self.padding)

    def A_dagger(self, y):
        r'''
        Returns the trivial inverse where x[0] = blurry input and x[1] with a delta filter, such that
        the convolution of x[0] and x[1] is y.

        .. note:

            This trivial inverse can be useful for some reconstruction networks, such as ``deepinv.models.ArtifactRemoval``.

        :param torch.tensor y: blurred measurement.
        :return: Tuple containing the trivial inverse.
        '''
        x = y.clone()
        mid_h = int(self.kernel_size[0]/2)
        mid_w = int(self.kernel_size[1]/2)
        w = torch.zeros((y.shape[0], 1, self.kernel_size[0], self.kernel_size[1]))
        w[:, :, mid_h, mid_w] = 1.

        return x, w


class Blur(LinearPhysics):
    r'''

    Blur operator.

    This forward operator performs

    .. math:: y = w*x

    where :math:`*` denotes convolution and :math:`w` is a filter.

    This class uses ``torch.conv2d`` for performing the convolutions.

    :param torch.Tensor filter: Tensor of size (1, 1, H, W) or (1, C, H, W) containing the blur filter, e.g., ``deepinv.physics.blur.gaussian_blur()``
    :param str padding: options are ``'valid'``, ``'circular'``, ``'replicate'`` and ``'reflect'``. If ``padding='valid'`` the blurred output is smaller than the image (no padding)
        otherwise the blurred output has the same size as the image.
    :param str device: cpu or cuda.

    '''
    def __init__(self, filter, padding='circular', device='cpu', **kwargs):
        super().__init__(**kwargs)
        self.padding = padding
        self.device = device
        self.filter = filter.requires_grad_(False).to(device)

    def A(self, x):
        return conv(x, self.filter, self.padding)

    def A_adjoint(self, y):
        return conv_transpose(y, self.filter, self.padding)


class BlurFFT(DecomposablePhysics):
    '''

    FFT-based blur operator.

    It performs

    .. math:: y = w*x

    where :math:`*` denotes convolution and :math:`w` is a filter.

    Blur operator based on ``torch.fft`` operations, which assumes a circular padding of the input, and allows for
    the singular value decomposition via ``deepinv.Physics.DecomposablePhysics`` and has fast pseudo-inverse and prox operators.

    :param tuple img_size: Input image size in the form (C, H, W).
    :param torch.tensor filter: torch.Tensor of size (1, 1, H, W) or (1, C,H,W) containing the blur filter, e.g., ``deepinv.physics.blur.gaussian_blur()``.
    :param str device: cpu or cuda

    '''

    def __init__(self,  img_size, filter, device='cpu', **kwargs):
        super().__init__(**kwargs)


        #assert img_size[-2] > filter.shape[-2] and img_size[-3] > filter.shape[-3], 'filter should be smaller than the image'

        self.img_size = img_size
        self.mask = filter_fft(filter, img_size)
        self.mask = self.mask.requires_grad_(False).to(device)

    def V_adjoint(self, x):
        return fft.fft2(x, norm="ortho")

    def U(self, x):
        return fft.irfft2(x, norm="ortho", s=self.img_size[-2:])

    def U_adjoint(self, x):
        return self.V_adjoint(x)

    def V(self, x):
        return self.U(x)


# test code
if __name__ == "__main__":
    device = 'cuda:0'

    import matplotlib.pyplot as plt

    x = torchvision.io.read_image('../../../datasets/set3c/0/butterfly.png')
    x = x.unsqueeze(0).float().to(device)/255

    # test on non symmetric blur kernel
    import os
    import hdf5storage
    kernel_path = os.path.join('../../../datasets/kernels', 'Levin09.mat')
    kernels = hdf5storage.loadmat(kernel_path)['kernels']
    filter = torch.tensor(np.expand_dims(kernels[0,1], axis=(0, 1)))
    # print(filter.shape)
    # plt.imshow(filter[0,0,:,:].cpu().numpy())
    # plt.show()
    # filter = filter.flip(-1).flip(-2)
    # plt.imshow(filter[0,0,:,:].cpu().numpy())
    # plt.show()
    blurFFT = BlurFFT(filter=filter, img_size=x.shape[1:], device=device)
    blur = Blur(filter=filter, device=device)

    #physics = Downsampling(factor=factor, img_size=(3, pix, pix), mode='gauss', device=device)
    #physics.noise_model = dinv.physics.GaussianNoise(sigma=.1)

    y1 = blurFFT.A(x)
    y2 = blur.A(x)
    print(y1)
    print(y2)

    # print(physics.power_method(x))
    #x = [x, w]
    #xhat = physics.A_adjoint(y)

    #xhat = physics.A_dagger(y)
    #xhat = physics.prox_l2(y, torch.zeros_like(x), gamma=.1)

    #x = x[0]
    #xhat = xhat[0]


    # plt.imshow(x.squeeze(0).permute(1, 2, 0).cpu().numpy())
    # plt.show()
    # plt.imshow(y.squeeze(0).permute(1, 2, 0).cpu().numpy())
    # plt.show()
    # plt.imshow(xhat.squeeze(0).permute(1, 2, 0).cpu().numpy())
    # plt.show()

    # plt.imshow(physics.A(xhat).squeeze(0).permute(1, 2, 0).cpu().numpy())
    # plt.show()
