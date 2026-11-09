from __future__ import annotations
from typing import ClassVar
from deepinv.physics.forward import DecomposablePhysics, Physics
from deepinv.physics.mri import MRI
from deepinv.physics.generator import BernoulliSplittingMaskGenerator
import torch
import torch.nn.functional as F


class Inpainting(DecomposablePhysics):
    r"""
    Inpainting forward operator, keeps a subset of entries.

    The operator is described by the diagonal matrix

    .. math::

        A = \text{diag}(m) \in \mathbb{R}^{n\times n}

    where :math:`m` is a binary mask with :math:`n` entries.

    This operator is linear and has a trivial SVD decomposition, which allows for fast computation
    of the pseudo-inverse and proximal operator.

    An existing operator can be loaded from a saved `.pth` file via `self.load_state_dict(save_path)`,
    in a similar fashion to :class:`torch.nn.Module`.

    Masks can also be created on-the-fly using mask generators such as
    :class:`BernoulliSplittingMaskGenerator <deepinv.physics.generator.BernoulliSplittingMaskGenerator>`, see example below.

    :param torch.Tensor, float mask: If the input is a float, the entries of the mask will be sampled from a bernoulli
        distribution with probability equal to ``mask``. If the input is a :class:`torch.Tensor` matching `img_size`,
        the mask will be set to this tensor. If ``mask`` is :class:`torch.Tensor`, it must be shape that is broadcastable
        to input shape and will be broadcast during forward call.
        If ``None``, it must be set during forward pass or using ``update`` method.
    :param tuple img_size: size of the input images without batch dimension e.g. of shape ``(C, H, W)`` or ``(C, M)`` or ``(M,)``.
    :param torch.device device: gpu or cpu.
    :param bool pixelwise: Apply the mask in a pixelwise fashion, i.e., zero all channels in a given pixel simultaneously.
        If existing mask passed (i.e. mask is :class:`torch.Tensor`), this has no effect.
    :param torch.Generator rng: a pseudorandom random number generator for the mask generation. Default to None.

    |sep|

    :Examples:

        Inpainting operator using defined mask, removing the second column of a 3x3 image:

        >>> from deepinv.physics import Inpainting
        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> x = torch.randn(1, 1, 3, 3) # Define random 3x3 image
        >>> mask = torch.zeros(1, 3, 3) # Define empty mask
        >>> mask[:, 2, :] = 1 # Keeping last line only
        >>> physics = Inpainting(mask=mask, img_size=x.shape[1:])
        >>> physics(x)
        tensor([[[[ 0.0000, -0.0000, -0.0000],
                  [ 0.0000, -0.0000, -0.0000],
                  [ 0.4033,  0.8380, -0.7193]]]])

        Inpainting operator using random mask, keeping 70% of the entries of a 3x3 image:

        >>> from deepinv.physics import Inpainting
        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> x = torch.randn(1, 1, 3, 3) # Define random 3x3 image
        >>> physics = Inpainting(mask=0.7, img_size=x.shape[1:])
        >>> physics(x)
        tensor([[[[ 1.5410, -0.0000, -2.1788],
                  [ 0.5684, -0.0000, -1.3986],
                  [ 0.4033,  0.0000, -0.0000]]]])

        Generate random masks on-the-fly using mask generators:

        >>> from deepinv.physics import Inpainting
        >>> from deepinv.physics.generator import BernoulliSplittingMaskGenerator
        >>> x = torch.randn(1, 1, 3, 3) # Define random 3x3 image
        >>> physics = Inpainting(img_size=x.shape[1:])
        >>> gen = BernoulliSplittingMaskGenerator(x.shape[1:], split_ratio=0.7)
        >>> params = gen.step(batch_size=1, seed = 0) # Generate random mask
        >>> physics(x, **params) # Set mask on-the-fly
        tensor([[[[-0.4033, -0.0000,  0.1820],
                  [-0.8567,  1.1006, -1.0712],
                  [ 0.1227, -0.0000,  0.3731]]]])
        >>> physics.update(**params) # Alternatively update mask before forward call
        >>> physics(x)
        tensor([[[[-0.4033, -0.0000,  0.1820],
                  [-0.8567,  1.1006, -1.0712],
                  [ 0.1227, -0.0000,  0.3731]]]])

    """

    def __init__(
        self,
        img_size: tuple[int],
        mask: torch.Tensor | float = None,
        pixelwise: bool = True,
        device: torch.device | str = "cpu",
        rng: torch.Generator = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)

        if isinstance(mask, torch.Tensor):
            mask = mask.to(device)
        elif isinstance(mask, float):
            self.gen = BernoulliSplittingMaskGenerator(
                img_size=img_size,
                split_ratio=mask,
                pixelwise=pixelwise,
                device=device,
                rng=rng,
            )
            mask = self.gen.step(batch_size=None)["mask"]
        elif mask is None:
            pass
        else:
            raise ValueError(
                "mask should either be torch.nn.Parameter, torch.Tensor, float or None."
            )

        if mask is not None and len(mask.shape) == len(img_size):
            mask = mask.unsqueeze(0)

        self.img_size = img_size
        self.register_buffer("mask", mask)
        self.to(device)

    def noise(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Incorporates noise into the measurements :math:`\tilde{y} = N(y)`

        :param torch.Tensor x:  clean measurements
        :return torch.Tensor: noisy measurements
        """
        noise = self.U(
            self.V_adjoint(
                self.V(self.U_adjoint(self.noise_model(x, **kwargs)) * self.mask)
            )
        )
        return noise

    def __mul__(self, other: Physics) -> Physics:
        r"""
        Concatenates two forward operators :math:`A = A_1\circ A_2` via the mul operation

        If the second operator is an Inpainting or MRI operator, the masks are multiplied element-wise,
        otherwise the default implementation of LinearPhysics is used (see :func:`deepinv.physics.LinearPhysics.__mul__`).

        :param deepinv.physics.Physics other: Physics operator :math:`A_2`
        :return: (:class:`deepinv.physics.Physics`) concatenated operator

        """

        if isinstance(other, self.__class__):
            return self.__class__(
                img_size=self.img_size,
                mask=self.mask * other.mask,
                noise_model=self.noise_model,
                device=self.mask.device if self.mask is not None else None,
            )
        elif isinstance(other, MRI):  # handles derived classes
            return other.__class__(
                mask=self.mask * other.mask,
                noise_model=self.noise_model,
                img_size=other.img_size,
                device=self.mask.device if self.mask is not None else None,
            )
        else:
            return super().__mul__(
                other, device=self.mask.device if self.mask is not None else None
            )


class Demosaicing(Inpainting):
    r"""
    Demosaicing operator.

    The operator keeps one band per pixel, according to a filter array pattern repeated
    over the sensor (a Bayer pattern for RGB cameras, or a multispectral filter array
    for snapshot multispectral cameras).

    :param tuple img_size: size of the input images, e.g. (H, W) or (C, H, W). If (H, W), the number of channels C is taken from the pattern.
    :param pattern: name of a built-in pattern (see :attr:`PATTERNS`), or a custom 2D tile of band indices.
        ``"bayer"`` (alias ``"bggr"``), ``"rggb"``, ``"grbg"``, ``"gbrg"`` are 2x2 Bayer patterns
        (see https://en.wikipedia.org/wiki/Bayer_filter), ``"rgb_nir"`` is a 4-band RGB+NIR array,
        ``"brauers"`` a 6-band 3x2 mosaic and ``"imec_16"`` a 16-band 4x4 mosaic.
    :param torch.device device: ``gpu`` or ``cpu``

    |sep|

    :Examples:

        Demosaicing operator using Bayer pattern for a 4x4 image:

        >>> from deepinv.physics import Demosaicing
        >>> x = torch.ones(1, 3, 4, 4)
        >>> physics = Demosaicing(img_size=(4, 4))
        >>> physics(x)[0, 1, :, :] # Green channel
        tensor([[0., 1., 0., 1.],
                [1., 0., 1., 0.],
                [0., 1., 0., 1.],
                [1., 0., 1., 0.]])

        16-band snapshot multispectral camera (4x4 mosaic):

        >>> x = torch.ones(1, 16, 4, 4)
        >>> physics = Demosaicing(img_size=(4, 4), pattern="imec_16")
        >>> physics(x)[0, 0] # band 0 is sampled at the top-left pixel of each tile
        tensor([[1., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]])

    """

    #: Built-in patterns: each tile gives the band index sampled at each pixel of the tile.
    PATTERNS: ClassVar[dict[str, list[list[int]]]] = {
        # 2x2 Bayer patterns, bands ordered (R, G, B)
        "rggb": [[0, 1], [1, 2]],
        "grbg": [[1, 0], [2, 1]],
        "gbrg": [[1, 2], [0, 1]],
        "bggr": [[2, 1], [1, 0]],
        "bayer": [[2, 1], [1, 0]],  # BGGR phase, as in the original implementation
        "rgb_nir": [[1, 3], [3, 0], [2, 3], [3, 1]],  # bands (R, G, B, NIR)
        "brauers": [[0, 1, 2], [3, 4, 5]],  # 6-band 3x2 mosaic
        "imec_16": [
            [4 * i + j for j in range(4)] for i in range(4)
        ],  # 16-band 4x4 mosaic
    }

    def __init__(
        self,
        img_size: tuple[int],
        pattern: str | list | tuple = "bayer",
        device: torch.device | str = "cpu",
        **kwargs,
    ):
        name = pattern if isinstance(pattern, str) else "custom tile"
        if isinstance(pattern, str):
            if pattern not in self.PATTERNS:
                raise ValueError(
                    f"The {pattern} pattern is not implemented. Available patterns are "
                    f"{list(self.PATTERNS.keys())}, or pass a custom 2D tile of band indices."
                )
            tile = self.PATTERNS[pattern]
        else:  # custom tile
            tile = pattern

        tile_t = torch.as_tensor(tile)
        if tile_t.dtype.is_floating_point and (tile_t != tile_t.round()).any():
            raise ValueError("Pattern tile entries must be non-negative integers")
        tile_t = tile_t.to(torch.long)
        if tile_t.dim() != 2 or tile_t.numel() == 0 or (tile_t < 0).any():
            raise ValueError(
                "Pattern tile must be a non-empty 2D array of band indices (>= 0)"
            )

        n_bands = int(tile_t.max()) + 1
        th, tw = tile_t.shape

        if len(img_size) == 2:
            img_size = (n_bands, *img_size)
        if len(img_size) != 3:
            raise ValueError("img_size should be of shape (H, W) or (C, H, W)")
        c, h, w = img_size
        if c != n_bands:
            raise ValueError(
                f"The {name} pattern has {n_bands} bands, but img_size has C={c}"
            )

        # Repeat the tile over the sensor, crop to (H, W) and one-hot encode the bands
        layout = tile_t.to(device).repeat((h + th - 1) // th, (w + tw - 1) // tw)[
            :h, :w
        ]
        mask = F.one_hot(layout, c).permute(2, 0, 1).to(torch.float32)

        super().__init__(
            img_size=(c, h, w), mask=mask, pixelwise=False, device=device, **kwargs
        )
