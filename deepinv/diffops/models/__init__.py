from .denoiser import register, make

from .ae import AE as ae
from .unet import UNet as unet
from .dncnn import DnCNN as dncnn
from .iterative.unroll import ArtifactRemoval
from .tgv import TGV as TGV
from .wavdict import WaveletPrior, WaveletDict
from .GSPnP import GSDRUNet
