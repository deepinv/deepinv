from .inpainting import Inpainting
from .compressed_sensing import CompressedSensing
from .blur import Blur, BlindBlur, Downsampling, BlurFFT
from .range import Decolorize
from .haze import Haze
from .forward import Denoising, Physics, LinearPhysics, DecomposablePhysics
from .noise import GaussianNoise, PoissonNoise, PoissonGaussianNoise, UniformNoise
from .mri import MRI
from .tomography import Tomography
from .lidar import SinglePhotonLidar
from .singlepixel import SinglePixelCamera
