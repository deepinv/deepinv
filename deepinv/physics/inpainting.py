from deepinv.physics.forward import DecomposablePhysics
from deepinv.physics.mri import MRI
from deepinv.physics.generator import BernoulliSplittingMaskGenerator
import torch


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
        distribution with probability equal to ``mask``. If the input is a :class:`torch.Tensor` matching `tensor_size`,
        the mask will be set to this tensor. If ``mask`` is :class:`torch.Tensor`, it must be shape that is broadcastable
        to input shape and will be broadcast during forward call.
        If ``None``, it must be set during forward pass or using ``update_parameters`` method.
    :param tuple tensor_size: size of the input images without batch dimension e.g. of shape ``(C, H, W)`` or ``(C, M)`` or ``(M,)``.
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
        >>> physics = Inpainting(mask=mask, tensor_size=x.shape[1:])
        >>> physics(x)
        tensor([[[[ 0.0000, -0.0000, -0.0000],
                  [ 0.0000, -0.0000, -0.0000],
                  [ 0.4033,  0.8380, -0.7193]]]])

        Inpainting operator using random mask, keeping 70% of the entries of a 3x3 image:

        >>> from deepinv.physics import Inpainting
        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> x = torch.randn(1, 1, 3, 3) # Define random 3x3 image
        >>> physics = Inpainting(mask=0.7, tensor_size=x.shape[1:])
        >>> physics(x)
        tensor([[[[ 1.5410, -0.0000, -2.1788],
                  [ 0.5684, -0.0000, -1.3986],
                  [ 0.4033,  0.0000, -0.0000]]]])

        Generate random masks on-the-fly using mask generators:

        >>> from deepinv.physics import Inpainting
        >>> from deepinv.physics.generator import BernoulliSplittingMaskGenerator
        >>> x = torch.randn(1, 1, 3, 3) # Define random 3x3 image
        >>> physics = Inpainting(tensor_size=x.shape[1:])
        >>> gen = BernoulliSplittingMaskGenerator(x.shape[1:], split_ratio=0.7)
        >>> params = gen.step(batch_size=1, seed = 0) # Generate random mask
        >>> physics(x, **params) # Set mask on-the-fly
        tensor([[[[-0.4033, -0.0000,  0.1820],
                  [-0.8567,  1.1006, -1.0712],
                  [ 0.1227, -0.0000,  0.3731]]]])
        >>> physics.update_parameters(**params) # Alternatively update mask before forward call
        >>> physics(x)
        tensor([[[[-0.4033, -0.0000,  0.1820],
                  [-0.8567,  1.1006, -1.0712],
                  [ 0.1227, -0.0000,  0.3731]]]])

    """

    def __init__(
        self,
        tensor_size,
        mask=None,
        pixelwise=True,
        device="cpu",
        rng: torch.Generator = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(mask, torch.nn.Parameter) or isinstance(mask, torch.Tensor):
            mask = mask.to(device)
        elif isinstance(mask, float):
            gen = BernoulliSplittingMaskGenerator(
                tensor_size=tensor_size,
                split_ratio=mask,
                pixelwise=pixelwise,
                device=device,
                rng=rng,
            )
            mask = gen.step(batch_size=None)["mask"]
        elif mask is None:
            pass
        else:
            raise ValueError(
                "mask should either be torch.nn.Parameter, torch.Tensor, float or None."
            )

        if mask is not None and len(mask.shape) == len(tensor_size):
            mask = mask.unsqueeze(0)

        self.tensor_size = tensor_size
        self.update_parameters(mask=mask)

    def noise(self, x, **kwargs):
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

    def __mul__(self, other):
        r"""
        Concatenates two forward operators :math:`A = A_1\circ A_2` via the mul operation

        If the second operator is an Inpainting or MRI operator, the masks are multiplied elementwise,
        otherwise the default implementation of LinearPhysics is used (see :func:`deepinv.physics.LinearPhysics.__mul__`).

        :param deepinv.physics.Physics other: Physics operator :math:`A_2`
        :return: (:class:`deepinv.physics.Physics`) concantenated operator

        """

        if isinstance(other, self.__class__):
            return self.__class__(
                tensor_size=self.tensor_size,
                mask=self.mask * other.mask,
                noise_model=self.noise_model,
                device=self.mask.device,
            )
        elif isinstance(other, MRI):  # handles derived classes
            return other.__class__(
                mask=self.mask * other.mask,
                noise_model=self.noise_model,
                img_size=other.img_size,
                device=self.mask.device,
            )
        else:
            return super().__mul__(other)


class Demosaicing(Inpainting):
    r"""
    Demosaicing operator.

    The operator chooses one color per pixel according to the pattern specified.

    :param tuple img_size: size of the input images, e.g. (H, W) or (C, H, W).
    :param str pattern: ``bayer`` (see https://en.wikipedia.org/wiki/Bayer_filter) or other patterns.
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

    """

    def __init__(self, img_size, pattern="bayer", device="cpu", **kwargs):
        if pattern == "bayer":
            if len(img_size) == 2:
                img_size = (3, img_size[0], img_size[1])

            mask = torch.zeros(img_size, device=device)
            # red
            mask[0, 1::2, 1::2] = 1
            # green
            mask[1, 0::2, 1::2] = 1
            mask[1, 1::2, 0::2] = 1
            # blue
            mask[2, 0::2, 0::2] = 1
        else:
            raise ValueError(f"The {pattern} pattern is not implemented")
        super().__init__(
            tensor_size=mask.shape, mask=mask, pixelwise=False, device=device, **kwargs
        )
