from typing import Tuple, Union
import torch
from torch import Tensor

from deepinv.physics.forward import DecomposablePhysics
from deepinv.physics.generator import BernoulliSplittingMaskGenerator


class CompressiveSpectralImaging(DecomposablePhysics):
    """Compressive Hyperspectral Imaging operator.
    
    Coded-aperture snapshot spectral imaging (CASSI) operator, which is a popular
    approach for hyperspectral imaging.

    The CASSI operator performs a combination of masking ("coded aperture"), shearing,
    and flattening in the channel dim.
    We provide two specific implementations of CASSI: Single-Disperser (i.e. only spatial encoding)
    and Spatial-Spectral encoding:

    .. math::



    For more details see e.g. `this overview <https://zaguan.unizar.es/record/75680/files/texto_completo.pdf>`_.

    |sep|

    :Examples:

    

    :param tuple img_size: image size, must be of form (C,H,W) where C is number of bands.
    :param Tensor, float mask: coded-aperture mask. If ``None``, generate random mask using
        :class:`deepinv.physics.generator.BernoulliSplittingMaskGenerator` with masking ratio
        of 0.5, if mask is ``float``, sets mask ratio to this.
    :param str mode: 'sd' = single disperser (i.e. only spatial encoding) or
        'ss' = spatial-spectral encoding, defaults to 'ss'. See above for details.
    :param float shear_factor: shear amount for spatial/spectral encoding. If 1, this equates
        to integer pixel shear. Defaults to 0.5.
    :param str shear_dim: shear in H-C plane or W-C plane where C is channel dim, defaults to "h"
    :param torch.device device: torch device, only used if ``mask`` is ``None`` or ``float``
    :param torch.Generator rng: torch random generator, only used if ``mask`` is ``None`` or ``float``
    """

    def __init__(
        self,
        img_size: Tuple[int, int, int],  # C,H,W
        mask: Union[Tensor, float] = None,
        mode: str = "ss",
        shear_factor: float = 0.5,
        shear_dim: str = "h",
        device: torch.device = "cpu",
        rng: torch.Generator = None,
        **kwargs,
    ):
        
        super().__init__(**kwargs)

        if len(img_size) != 3:
            raise ValueError("img_size must be (C, H, W)")

        self.img_size = img_size  # C,H,W
        self.shear_factor = shear_factor

        if shear_dim.lower() == "h":
            self.shear_dim = 2
        elif shear_dim.lower() == "w":
            self.shear_dim = 3
        else:
            raise ValueError("shear_dim must be either 'h' or 'w'.")

        if mode.lower() not in ("ss", "sd"):
            raise ValueError("mode must be either 'ss' or 'sd'.")
        self.mode = mode.lower()

        if mask is None or isinstance(mask, float):
            mask = BernoulliSplittingMaskGenerator(
                img_size,
                split_ratio=0.5 if mask is None else mask,
                device=device,
                rng=rng,
            ).step()[
                "mask"
            ]  # B,C,H,W pixelwise

        self.mask = mask

    def shear(self, x, un=False):
        shifts = (torch.arange(x.shape[1]) * self.shear_factor).floor().int()
        return torch.cat(
            [
                torch.roll(
                    x[:, [c]], s.item() if not un else -s.item(), dims=self.shear_dim
                )
                for c, s in enumerate(shifts)
            ],
            dim=1,
        )

    def flatten(self, x: Tensor) -> Tensor:
        return x.mean(axis=1, keepdim=True)

    def unflatten(self, y: Tensor) -> Tensor:
        return y.expand(y.shape[0], self.img_size[0], *y.shape[2:]) / (self.img_size[0])

    def V_adjoint(self, x: Tensor) -> Tensor:
        if self.mode == "ss":
            return self.shear(x)
        elif self.mode == "sd":
            return x

    def U(self, x: Tensor) -> Tensor:
        if self.mode == "ss":
            return self.flatten(self.shear(x, un=True))
        elif self.mode == "sd":
            return self.flatten(self.shear(x))

    def U_adjoint(self, y: Tensor) -> Tensor:
        if self.mode == "ss":
            return self.shear(self.unflatten(y))
        elif self.mode == "sd":
            return self.shear(self.unflatten(y), un=True)

    def V(self, y: Tensor) -> Tensor:
        if self.mode == "ss":
            return self.shear(y, un=True)
        elif self.mode == "sd":
            return y
