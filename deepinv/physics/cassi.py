from typing import Tuple, Union
import torch
from torch import Tensor
from torch.nn.functional import pad

from deepinv.physics.forward import LinearPhysics
from deepinv.physics.generator import BernoulliSplittingMaskGenerator
from deepinv.physics.functional.convolution import conv2d


class CompressiveSpectralImaging(LinearPhysics):
    r"""Compressive Hyperspectral Imaging operator.

    Coded-aperture snapshot spectral imaging (CASSI) operator, which is a popular
    approach for hyperspectral imaging.

    The CASSI operator performs a combination of masking ("coded aperture"), shearing,
    and flattening in the channel dim.
    We provide two specific popular CASSI models: single-disperser (i.e. only spatial encoding)
    and spatial-spectral encoding:

    .. math::

        y =
        \begin{cases} 
            \Sigma_{c=1}^{C} S^{-1} MSx & \text{if mode='spatial-spectral'} \\ 
            \Sigma_{c=1}^{C} SMx & \text{if mode='single-disperser'}
        \end{cases}

    where :math:`M` is a binary mask (the "coded aperture"), :math:`S` is a pixel shear in the 2D
    channel-height of channel-width plane and :math:`C` is number of channels. 
    Note that the output size of the single-disperser mode has the ``H`` or ``W`` dim extended by ``C-1`` pixels.

    For more details see e.g. the paper `High-Quality Hyperspectral Reconstruction Using a Spectral Prior <https://zaguan.unizar.es/record/75680/files/texto_completo.pdf>`_.

    The implementation is a type of linear physics as it is not completely decomposable due to edge effects and different scaling.

    |sep|

    :Examples:

        >>> from deepinv.physics import CompressiveSpectralImaging
        >>> physics = CompressiveSpectralImaging(img_size=(7, 32, 32))
        >>> x = torch.rand(1, 7, 32, 32) # 7-band image
        >>> y = physics(x)
        >>> y.shape
        torch.Size([1, 1, 32, 32])


    :param tuple img_size: image size, must be of form (C,H,W) where C is number of bands.
    :param torch.Tensor, float mask: coded-aperture mask. If ``None``, generate random mask using
        :class:`deepinv.physics.generator.BernoulliSplittingMaskGenerator` with masking ratio
        of 0.5, if mask is ``float``, sets mask ratio to this. If ``Tensor``, set mask to this,
        must be of shape ``(B,C,H,W)``.
    :param str mode: 'sd' for SD-CASSI i.e. single disperser (only spatial encoding) or
        'ss' for SS-CASSI i.e. spatial-spectral encoding. Defaults to 'ss'. See above for details.
    :param str shear_dir: "h" for shear in H-C plane or "w" for shear in W-C plane where C is channel dim, defaults to "h"
    :param torch.device device: torch device, only used if ``mask`` is ``None`` or ``float``
    :param torch.Generator rng: torch random generator, only used if ``mask`` is ``None`` or ``float``
    """

    def __init__(
        self,
        img_size: Tuple[int, int, int],  # C,H,W
        mask: Union[Tensor, float] = None,
        mode: str = "ss",
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

        if shear_dir.lower() not in ("h", "w"):
            raise ValueError("shear_dir must be either 'h' or 'w'.")
        self.shear_dir = shear_dir.lower()

        if mode.lower() not in ("ss", "sd"):
            raise ValueError("mode must be either 'ss' or 'sd'.")
        self.mode = mode.lower()

        if mask is None or isinstance(mask, float):
            # B,C,H,W pixelwise
            mask = BernoulliSplittingMaskGenerator(
                img_size,
                split_ratio=0.5 if mask is None else mask,
                device=device,
                rng=rng,
            ).step()["mask"]

        self.update_parameters(mask=mask, **kwargs)

        # In SS-CASSI, masking happens on the padded image after shearing
        if self.mode == "ss":
            self.mask = self.pad(self.mask)

    def pad(self, x: Tensor) -> Tensor:
        """Pad image on bottom or on right.

        :param torch.Tensor x: input image
        """
        if self.shear_dir == "h":
            return pad(x, (0, 0, 0, self.C - 1), value=1.0)
        elif self.shear_dir == "w":
            return pad(x, (0, self.C - 1), value=1.0)

    def crop(self, x: Tensor) -> Tensor:
        """Crop image on bottom or on right.

        :param torch.Tensor x: input padded image
        """
        if self.shear_dir == "h":
            return x[:, :, : (1 - self.C), :]
        elif self.shear_dir == "w":
            return x[:, :, :, : (1 - self.C)]

    def shear(self, x: Tensor, un=False) -> Tensor:
        """Efficient pixel shear in channel-spatial plane

        :param torch.Tensor x: input image of shape (B,C,H,W)
        :param bool un: if ``True``, unshear in opposite direction.
        """
        H, W = x.shape[-2:]

        # Construct kernel
        ker = torch.zeros_like(x)
        for i in range(self.C):
            if self.shear_dir == "h":
                ker[:, i, (H - 1) // 2 - 0 + (-i if un else i), (W - 1) // 2 - 0] = 1
            elif self.shear_dir == "w":
                ker[:, i, (H - 1) // 2 - 0, (W - 1) // 2 - 0 + (-i if un else i)] = 1

        return conv2d(x, ker, padding="constant")

    def flatten(self, x: Tensor) -> Tensor:
        """Average over channel dimension

        :param torch.Tensor x: input image of shape B,C,H,W
        """
        return x.mean(axis=1, keepdim=True)

    def unflatten(self, y: Tensor) -> Tensor:
        """Repeat over channel dimension

        :param torch.Tensor y: input image of shape B,C,H,W
        """
        return y.expand(y.shape[0], self.img_size[0], *y.shape[2:]) / (self.img_size[0])

    def A(self, x: Tensor, mask: Tensor = None, **kwargs) -> Tensor:
        r"""
        Applies the CASSI forward operator.

        If a mask is provided, it updates the class attribute ``mask`` on the fly.

        :param torch.Tensor x: input image
        :param torch.Tensor mask: CASSI mask
        :return: (:class:`torch.Tensor`) output measurements
        """
        if x.shape[1:] != self.img_size:
            raise ValueError("Input must be same shape as img_shape.")

        self.update_parameters(mask=mask)

        # fmt: off
        if self.mode == "ss":
            y = self.crop(self.flatten(
                self.shear(
                    self.mask * self.shear(
                        self.pad(x)
                    ),
                un=True)
            ))
        
        elif self.mode == "sd":
            y = self.flatten(
                self.shear(
                    self.pad(self.mask * x)
                )
            )

        # fmt: on
        return y

    def A_adjoint(self, y: Tensor, mask: Tensor = None, **kwargs) -> Tensor:
        r"""
        Applies the CASSI adjoint operator.

        If a mask is provided, it updates the class attribute ``mask`` on the fly.

        :param torch.Tensor x: input measurements
        :param torch.Tensor mask: CASSI mask
        :return: (:class:`torch.Tensor`) output image
        """
        self.update_parameters(mask=mask)

        # fmt: off
        if self.mode == "ss":
            x = self.crop(
                self.shear(
                    self.mask * self.shear(
                        self.pad(
                            self.unflatten(y)
                        )
                    ),
                un=True)
            )

        elif self.mode == "sd":
            x = self.mask * self.crop(
                self.shear(
                    self.unflatten(y),
                un=True)
            )

        # fmt: on
        if x.shape[1:] != self.img_size:
            raise ValueError("Output must be same shape as img_size.")
        return x

    def update_parameters(self, mask: Tensor, **kwargs):
        self.mask = mask if mask is not None else self.mask
