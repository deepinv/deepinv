import torch
from deepinv.loss.loss import Loss


class TVLoss(Loss):
    r"""
    Total variation loss (:math:`\ell_2` norm).

    It computes the loss :math:`\|D\hat{x}\|_2^2`,
    where :math:`D` is a normalized linear operator that computes the vertical and horizontal (and depth, for 3D volumes) first order differences
    of the reconstructed image :math:`\hat{x}`.

    :param float weight: scalar weight for the TV loss.
    """

    def __init__(self, weight: float = 1.0):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = weight
        self._name = "tv"

    def forward(self, x_net: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Computes the TV loss.

        :param torch.Tensor x_net: reconstructed image.
        :return: torch.Tensor loss of size (batch_size,)
        """
        tv = 0
        for dim in range(2, x_net.dim()):
            diff = torch.diff(x_net, dim=dim)
            tv = tv + diff.pow(2).reshape(x_net.size(0), -1).sum(1) / self.tensor_size(
                diff
            )
        return self.tv_loss_weight * 2 * tv

    @staticmethod
    def tensor_size(t: torch.Tensor) -> int:
        return t[0].numel()
