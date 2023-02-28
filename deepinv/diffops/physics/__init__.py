from deepinv.diffops.physics.inpainting import Inpainting
from deepinv.diffops.physics.compressed_sensing import CompressedSensing
from deepinv.diffops.physics.noise import Denoising, GaussianNoise, PoissonNoise, PoissonGaussianNoise, UniformNoise
from deepinv.diffops.physics.blur import Blur, BlindBlur, Downsampling, BlurFFT
from deepinv.diffops.physics.mri import MRI
try :
    from deepinv.diffops.physics.ct import CT
except :
    print('ERROR with CT, to correct')
from deepinv.diffops.physics.range import Decolorize
