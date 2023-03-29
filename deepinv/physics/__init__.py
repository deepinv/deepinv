from deepinv.physics.inpainting import Inpainting
from deepinv.physics.compressed_sensing import CompressedSensing
from deepinv.physics.blur import Blur, BlindBlur, Downsampling, BlurFFT
from deepinv.physics.noise import GaussianNoise, PoissonNoise, PoissonGaussianNoise, UniformNoise
from deepinv.physics.range import Decolorize
from .noise import Denoising
from .mri import MRI

try:
    from deepinv.physics.ct import CT
except:
    print('ERROR with CT, to correct')
