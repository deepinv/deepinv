from deepinv.physics.forward import DecomposablePhysics
from deepinv.physics.mri import MRI
import torch


class Inpainting(DecomposablePhysics):
    r"""

    Inpainting forward operator, keeps a subset of entries.

    The operator is described by the diagonal matrix

    .. math::

        A = \text{diag}(m) \in \mathbb{R}^{n\times n}

    where :math:`m` is a binary mask with n entries.

    This operator is linear and has a trivial SVD decomposition, which allows for fast computation
    of the pseudo-inverse and proximal operator.

    An existing operator can be loaded from a saved ``.pth`` file via ``self.load_state_dict(save_path)``,
    in a similar fashion to ``torch.nn.Module``.

    :param torch.Tensor, float mask: If the input is a float, the entries of the mask will be sampled from a bernoulli
        distribution with probability equal to ``mask``. If the input is a ``torch.tensor`` matching tensor_size,
        the mask will be set to this tensor. If ``mask`` is ``torch.Tensor``, it must be shape that is broadcastable to input shape and will be broadcast during forward call.
    :param tuple tensor_size: size of the input images (C, H, W) or (B, C, H, W).
    :param torch.device device: gpu or cpu
    :param bool pixelwise: Apply the mask in a pixelwise fashion, i.e., zero all channels in a given pixel simultaneously.

    |sep|

    :Examples:

        Inpainting operator using defined mask, removing the second column of a 3x3 image:

        >>> from deepinv.physics import Inpainting
        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> x = torch.randn(1, 1, 3, 3) # Define random 3x3 image
        >>> mask = torch.zeros(1, 3, 3) # Define empty mask
        >>> mask[:, 2, :] = 1 # Keeping last line only
        >>> physics = Inpainting(mask=mask, tensor_size=(1, 1, 3, 3))
        >>> physics(x)
        tensor([[[[ 0.0000, -0.0000, -0.0000],
                  [ 0.0000, -0.0000, -0.0000],
                  [ 0.4033,  0.8380, -0.7193]]]])

        Inpainting operator using random mask, keeping 70% of the entries of a 3x3 image:

        >>> from deepinv.physics import Inpainting
        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> x = torch.randn(1, 3, 3) # Define random 3x3 image
        >>> physics = Inpainting(mask=0.7, tensor_size=(1, 1, 3, 3))
        >>> physics(x)
        tensor([[[[[ 1.5410, -0.0000, -2.1788],
                   [ 0.0000, -1.0845, -1.3986],
                   [ 0.0000,  0.8380, -0.7193]]]]])

    """

    def __init__(self, tensor_size, mask, pixelwise=True, device="cpu", **kwargs):
        super().__init__(**kwargs)
        if isinstance(mask, torch.nn.Parameter) or isinstance(mask, torch.Tensor):
            mask = mask.to(device)
        elif type(mask) == float:
            mask_rate = mask
            mask = torch.ones(tensor_size, device=device)
            aux = torch.rand_like(mask)
            if not pixelwise:
                mask[aux > mask_rate] = 0
            else:
                mask[:, aux[0, :, :] > mask_rate] = 0

        if len(mask.shape) == len(tensor_size):
            mask = mask.unsqueeze(0)

        self.tensor_size = tensor_size
        self.mask = torch.nn.Parameter(mask, requires_grad=False)

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
        otherwise the default implementation of LinearPhysics is used (see :meth:`deepinv.physics.LinearPhysics.__mul__`).

        :param deepinv.physics.Physics other: Physics operator :math:`A_2`
        :return: (deepinv.physics.Physics) concantenated operator

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
