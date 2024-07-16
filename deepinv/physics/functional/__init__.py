from .interp import ThinPlateSpline
from .radon import Radon, IRadon, RampFilter
from .downsampling import downsample
from .hist import histogram, histogramdd
from .multiplier import (
    multiplier,
    multiplier_adjoint,
)
from .convolution import (
    conv2d,
    conv_transpose2d,
    conv2d_fft,
    conv_transpose2d_fft,
    filter_fft_2d,
    conv3d,
    conv_transpose3d,
)

from .product_convolution import (
    product_convolution2d,
    product_convolution2d_adjoint,
    product_convolution2d_patches,
    product_convolution2d_adjoint_patches,
    get_psf_product_convolution2d,
    get_psf_product_convolution2d_patches
)
