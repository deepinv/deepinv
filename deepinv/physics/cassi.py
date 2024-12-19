from typing import Tuple, Union
import torch
from torch import Tensor
from torch.nn.functional import pad

from deepinv.physics.forward import DecomposablePhysics
from deepinv.physics.generator import BernoulliSplittingMaskGenerator
from deepinv.physics.functional.convolution import conv2d


class CompressiveSpectralImaging(DecomposablePhysics):
    r"""Compressive Hyperspectral Imaging operator.

    Coded-aperture snapshot spectral imaging (CASSI) operator, which is a popular
    approach for hyperspectral imaging.

    The CASSI operator performs a combination of masking ("coded aperture"), shearing,
    and flattening in the channel dim.
    We provide two specific popular CASSI models: Single-Disperser (i.e. only spatial encoding)
    and Spatial-Spectral encoding:

    .. math::

        y =
        \begin{cases} 
            \Sigma_{c=1}^{C} S^{-1} MSx & \text{if mode='ss'} \\ 
            \Sigma_{c=1}^{C} SMx & \text{if mode='sd'}
        \end{cases}

    where :math:`M` is a binary mask (the "coded aperture") and :math:`S` is a pixel shear in the 2D
    channel-height of channel-width plane.

    For more details see e.g. `this overview <https://zaguan.unizar.es/record/75680/files/texto_completo.pdf>`_.

    |sep|

    :Examples:

        >>> from deepinv.physics import CompressiveSpectralImaging
        >>> physics = CompressiveSpectralImaging(img_size=(7, 32, 32))
        >>> x = torch.rand(1, 7, 32, 32) # 7-band image
        >>> y = physics(x)
        >>> y.shape
        torch.Size([1, 1, 32, 32])


    :param tuple img_size: image size, must be of form (C,H,W) where C is number of bands.
    :param Tensor, float mask: coded-aperture mask. If ``None``, generate random mask using
        :class:`deepinv.physics.generator.BernoulliSplittingMaskGenerator` with masking ratio
        of 0.5, if mask is ``float``, sets mask ratio to this.
    :param str mode: 'sd' = single disperser (i.e. only spatial encoding) or
        'ss' = spatial-spectral encoding, defaults to 'ss'. See above for details.
    :param float shear_factor: shear amount for spatial/spectral encoding. If 1, this equates
        to integer pixel shear. Defaults to 0.5.
    :param str shear_dir: shear in H-C plane or W-C plane where C is channel dim, defaults to "h"
    :param torch.device device: torch device, only used if ``mask`` is ``None`` or ``float``
    :param torch.Generator rng: torch random generator, only used if ``mask`` is ``None`` or ``float``
    """

    def __init__(
        self,
        img_size: Tuple[int, int, int],  # C,H,W
        mask: Union[Tensor, float] = None,
        mode: str = "ss",
        shear_factor: float = 0.5,
        shear_dir: str = "h",
        device: torch.device = "cpu",
        rng: torch.Generator = None,
        **kwargs,
    ):

        super().__init__(**kwargs)

        if len(img_size) != 3:
            raise ValueError("img_size must be (C, H, W)")

        self.img_size = img_size  # C,H,W
        self.C = img_size[0]
        self.shear_factor = shear_factor

        if shear_dir.lower() not in ("h", "w"):
            raise ValueError("shear_dir must be either 'h' or 'w'.")
        self.shear_dir = shear_dir.lower()

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

        if self.mode == "ss":
            self.mask = self.pad(self.mask)

    def pad(self, x):
        if self.shear_dir == "h":
            return pad(x, (0, 0, 0, self.C - 1), value=1.0)
        elif self.shear_dir == "w":
            return pad(x, (0, self.C - 1), value=1.0)

    def crop(self, x):
        if self.shear_dir == "h":
            return x[:, :, : (1 - self.C), :]
        elif self.shear_dir == "w":
            return x[:, :, :, : (1 - self.C)]

    def shear(self, x: Tensor, un=False) -> Tensor:
        """Efficient pixel shear in channel-spatial plane

        :param Tensor x: input image of shape (B,C,H,W)
        :param bool un: if ``True``, unshear in opposite direction.
        """
        H, W = x.shape[-2:]

        w = torch.zeros_like(x)
        for i in range(self.C):
            if self.shear_dir == "h":
                w[:, i, (H - 1) // 2 - 0 + (-i if un else i), (W - 1) // 2 - 0] = 1
            elif self.shear_dir == "w":
                w[:, i, (H - 1) // 2 - 0, (W - 1) // 2 - 0 + (-i if un else i)] = 1

        return conv2d(x, w, padding="constant")

    def flatten(self, x: Tensor) -> Tensor:
        """Average over channel dimension

        :param Tensor x: input image of shape B,C,H,W
        """
        return x.mean(axis=1, keepdim=True)

    def unflatten(self, y: Tensor) -> Tensor:
        """Repeat over channel dimension

        :param Tensor y: input image of shape B,C,H,W
        """
        return y.expand(y.shape[0], self.img_size[0], *y.shape[2:]) / (self.img_size[0])

    def V_adjoint(self, x: Tensor) -> Tensor:
        if x.shape[1:] != self.img_size:
            raise ValueError("Input must be same shape as img_shape.")

        if self.mode == "ss":
            return self.shear(self.pad(x))
        elif self.mode == "sd":
            return x

    def U(self, x: Tensor) -> Tensor:
        if self.mode == "ss":
            return self.crop(self.flatten(self.shear(x, un=True)))
        elif self.mode == "sd":
            return self.flatten(self.shear(self.pad(x)))

    def U_adjoint(self, y: Tensor) -> Tensor:
        if self.mode == "ss":
            return self.shear(self.pad(self.unflatten(y)))
        elif self.mode == "sd":
            return self.crop(self.shear(self.unflatten(y), un=True))

    def V(self, y: Tensor) -> Tensor:
        if self.mode == "ss":
            x = self.crop(self.shear(y, un=True))
        elif self.mode == "sd":
            x = y

        if x.shape[1:] != self.img_size:
            raise ValueError("Output must be same shape as img_size.")
        return x
