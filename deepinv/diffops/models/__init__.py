from .denoiser import register, make

from .ae import AE as ae
from .unet import UNet as unet
from .drunet import UNetRes as drunet
from .drunet import test_mode as drunet_testmode
from .dncnn import DnCNN as dncnn
from .iterative.unroll import ArtifactRemoval
from .tgv import TGV as TGV
from .wavdict import WaveletPrior, WaveletDict
