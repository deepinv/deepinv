import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from utils.utils import get_transforms

import deepinv as dinv

from deepinv.physics.generator import RandomMaskGenerator, MotionBlurGenerator, DiffractionBlurGenerator, GeneratorMixture, SigmaGenerator
from deepinv.datasets import DIV2K

from physics.blur_generator import GaussianBlurGenerator
from physics.inpainting_generator import InpaintingMaskGenerator


train_patch_size = 128


def to_nn_parameter(x):
    if isinstance(x, torch.Tensor):
        return torch.nn.Parameter(x, requires_grad=False)
    else:
        return torch.nn.Parameter(torch.tensor(x), requires_grad=False)


class GaussianNoise(torch.nn.Module):
    r"""

    Gaussian noise :math:`y=z+\epsilon` where :math:`\epsilon\sim \mathcal{N}(0,I\sigma^2)`.

    |sep|

    :Examples:

        Adding gaussian noise to a physics operator by setting the ``noise_model``
        attribute of the physics operator:

        >>> from deepinv.physics import Denoising, GaussianNoise
        >>> import torch
        >>> physics = Denoising()
        >>> physics.noise_model = GaussianNoise()
        >>> x = torch.rand(1, 1, 2, 2)
        >>> y = physics(x)

    :param float sigma: Standard deviation of the noise.

    """

    def __init__(self, sigma=0.1):
        super().__init__()
        self.update_parameters(sigma)

    def forward(self, x, sigma=None, **kwargs):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :param float, torch.Tensor sigma: standard deviation of the noise.
            If not None, it will overwrite the current noise level.
        :returns: noisy measurements
        """
        self.update_parameters(sigma)
        return x + torch.randn_like(x) * self.sigma[(..., *([None] * (x.dim() - 1)))].to(x.device)

    def update_parameters(self, sigma=None, **kwargs):
        r"""
        Updates the standard deviation of the noise.

        :param float, torch.Tensor sigma: standard deviation of the noise.
        """
        if sigma is not None:
            self.sigma = to_nn_parameter(sigma)

def get_div2k(train_patch_size, device='cpu'):

    target_size = (train_patch_size, train_patch_size)

    # Define the transform pipeline
    transform = transforms.Compose([
        transforms.RandomCrop(train_patch_size, pad_if_needed=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    div2k_dataset = DIV2K(root='/gpfsscratch/rech/nyd/commun/', download=False, transform=transform)
    psf_size = 31

    motion_generator = MotionBlurGenerator(
        (psf_size, psf_size), l=0.6, sigma=1, device=device
    ) + SigmaGenerator(device=device) + SigmaGenerator(sigma_min=0.01, sigma_max=0.5, device=device)

    diffraction_generator = DiffractionBlurGenerator(psf_size=(psf_size, psf_size), num_channels=1, device=device) + SigmaGenerator(sigma_min=0.01, sigma_max=0.5, device=device)
    gaussian_blur_generator = GaussianBlurGenerator(psf_size=(psf_size, psf_size), num_channels=1, device=device) + SigmaGenerator(sigma_min=0.01, sigma_max=0.5, device=device)

    generator = GeneratorMixture([motion_generator, diffraction_generator, gaussian_blur_generator], [1/3., 1/3., 1/3.])

    physics = dinv.physics.BlurFFT(img_size=(3, train_patch_size, train_patch_size),
                                   noise_model=GaussianNoise(sigma=.1),
                                   device=device)

    return div2k_dataset, generator, physics


def get_inpainting_physics(train_patch_size, device='cpu'):
    physics = dinv.physics.Inpainting(
        mask=0.1,
        tensor_size=(3, train_patch_size, train_patch_size),
        noise_model=dinv.physics.GaussianNoise(sigma=.1),
        device=device,
    )

    generator = InpaintingMaskGenerator((train_patch_size, train_patch_size), block_size_ratio=0.2, num_blocks=10, device=device) + SigmaGenerator(sigma_min=0.01, sigma_max=0.5, device=device)

    return generator, physics


def get_SR_physics(train_patch_size, device='cpu', factor=2):
    physics = dinv.physics.Downsampling(
        factor=factor,
        filter='bicubic',
        img_size=(3, train_patch_size, train_patch_size),
        noise_model=dinv.physics.GaussianNoise(sigma=.1),
        padding='circular',
        device=device,
    )

    generator = SigmaGenerator(sigma_min=0., sigma_max=0.05, device=device)

    return generator, physics

def get_drunet_dataset(train_patch_size, device='cpu', pth='/pth/to/train', sigma_min=0.01, sigma_max=0.5):

    target_size = (train_patch_size, train_patch_size)

    # Define the transform pipeline
    transform = transforms.Compose([
        transforms.RandomCrop(train_patch_size, pad_if_needed=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.ImageFolder(root=pth, transform=transform)
    physics = dinv.physics.DecomposablePhysics(device=device,
                                               noise_model=GaussianNoise(sigma=0.1))
    generator = SigmaGenerator(sigma_min=sigma_min, sigma_max=sigma_max, device=device)

    return dataset, generator, physics


def get_BSD68(train_patch_size, device='cpu'):

    target_size = (train_patch_size, train_patch_size)

    # Define the transform pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(target_size)
    ])

    bsd68_dataset = torchvision.datasets.ImageFolder(root='/gpfsscratch/rech/fwf/commun/CBSD68/', transform=transform)
    psf_size = 31

    motion_generator = MotionBlurGenerator(
        (psf_size, psf_size), l=0.6, sigma=1, device=device
    )

    physics = dinv.physics.BlurFFT(img_size=(3, train_patch_size, train_patch_size),
                                   noise_model=dinv.physics.GaussianNoise(sigma=.1),
                                   device=device)

    return bsd68_dataset, motion_generator, physics


def get_fastMRI(train_patch_size, device='cpu', train=True):

    def rescale(x, factor=1e4):
        return x * factor

    target_size = (train_patch_size, train_patch_size)

    # Define the transform pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        torchvision.transforms.Lambda(rescale),
        transforms.RandomCrop(target_size, pad_if_needed=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])

    training_str = 'train' if train else 'val'

    fastmri_dataset = dinv.datasets.FastMRISliceDataset(root='/gpfsscratch/rech/nyd/commun/fastMRI/knee_singlecoil/singlecoil_'+str(training_str)+'/',
                                                        challenge='singlecoil',
                                                        transform_kspace=transform,
                                                        transform_target=transform,)


    mask_generator = RandomMaskGenerator((target_size[0], target_size[1]))
    physics = dinv.physics.MRI(img_size=target_size, device=device)

    return fastmri_dataset, mask_generator, physics
