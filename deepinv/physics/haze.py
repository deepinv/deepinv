import torch
from deepinv.physics.forward import Physics
from deepinv.utils.tensorlist import TensorList


class Haze(Physics):
    r"""
    Standard haze model

    The operator is defined as https://ieeexplore.ieee.org/abstract/document/5567108/

     .. math::

        y = t \odot I + a (1-t)

     where :math:`t = \exp(-\beta d - o)` is the medium transmission,  :math:`I` is the intensity (possibly RGB) image,
     :math:`\odot` denotes element-wise multiplication, :math:`a>0` is the atmospheric light,
     :math:`d` is the scene depth, and :math:`\beta>0` and :math:`o` are constants.

    This is a non-linear inverse problems, whose unknown parameters are :math:`I`, :math:`d`, :math:`a`.

    :param float beta: constant :math:`\beta>0`
    :param float offset: constant :math:`o`

    """

    def __init__(self, beta=0.1, offset=0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.offset = offset

    def A(self, x, **kwargs):
        r"""
        :param list, tuple x:  The input x should be a tuple/list such that x[0] = image torch.tensor :math:`I`,
         x[1] = depth torch.tensor :math:`d`, x[2] = scalar or torch.tensor of one element :math:`a`.
        :return: (torch.tensor) hazy image.

        """
        im = x[0]
        d = x[1]
        A = x[2]

        t = torch.exp(-self.beta * (d + self.offset))
        y = t * im + (1 - t) * A
        return y

    def A_dagger(self, y, **kwargs):
        r"""

        Returns the trivial inverse where x[0] = y (trivial estimate of the image :math:`I`),
        x[1] = tensor of depth :math:`d` equal to one, x[2] = 1 for :math:`a`.

        .. note:

            This trivial inverse can be useful for some reconstruction networks, such as ``deepinv.models.ArtifactRemoval``.


        :param torch.Tensor y: Hazy image.
        :return: (deepinv.utils.ListTensor) trivial inverse.

        """
        b, c, h, w = y.shape
        d = torch.ones((b, 1, h, w), device=y.device)
        A = torch.ones(1, device=y.device)
        return TensorList([y, d, A])
