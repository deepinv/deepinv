from .inpainting import Inpainting
from .compressed_sensing import CompressedSensing
from .blur import Blur, BlindBlur, Downsampling, BlurFFT
from .range import Decolorize
from .haze import Haze
from .forward import Denoising, Physics, LinearPhysics, DecomposablePhysics
from .noise import GaussianNoise, PoissonNoise, PoissonGaussianNoise, UniformNoise
from .mri import MRI
from .lidar import SinglePhotonLidar
from .singlepixel import SinglePixelCamera

try:
    from deepinv.physics.ct import CT
except:
    print("ERROR with CT, to correct")
