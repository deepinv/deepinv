from .convolution import (
    conv2d,
    conv_transpose2d,
    conv2d_fft,
    conv_transpose2d_fft,
    filter_fft_2d,
    conv3d,
    conv_transpose3d,
)

from .product_convolution import product_convolution2d, product_convolution2d_adjoint, product_convolution2d_patches, product_convolution2d_adjoint_patches

from .multiplier import (
    multiplier,
    multiplier_adjoint,
)

from .hist import histogram, histogramdd
from .downsampling import downsample
from .radon import Radon, IRadon, RampFilter
from .interp import ThinPlateSpline
