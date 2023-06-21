import torch
from deepinv.physics import Inpainting
import numpy as np


class SplittingLoss(torch.nn.Module):
    r"""
    Measurement splitting loss

    Splits the measurement and forward operator (of size :math:`m`)
    into two smaller pairs  :math:`(y_1,A_1)` (of size :math:`m_1`) and  :math:`(y_2,A_2)` (of size :math:`m_2`) ,
    to compute the self-supervised loss:

    .. math::

        \frac{m}{m_2}\| y_2 - A_2 \inversef{y_1,A_1}\|^2

    where :math:`R` is the trainable network. See https://pubmed.ncbi.nlm.nih.gov/32614100/.

    By default, the error is computed using the MSE metric, however any other metric (e.g., :math:`\ell_1`)
    can be used as well.

    :param torch.nn.Module metric: metric used for computing data consistency,
        which is set as the mean squared error by default.
    :param float split_ratio: splitting ratio, should be between 0 and 1. The size of :math:`y_1` increases
        with the splitting ratio.
    :param bool regular_mask: If ``True``, it will use a regular mask, otherwise it uses a random mask.
    """

    def __init__(self, metric=torch.nn.MSELoss(), split_ratio=0.9, regular_mask=False):
        super(SplittingLoss, self).__init__()
        self.name = "ms"
        self.metric = metric
        self.regular_mask = regular_mask
        self.split_ratio = split_ratio

    def forward(self, y, physics, model, **kwargs):
        r"""
        Computes the measurement splitting loss

        :param torch.Tensor y: Measurements.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction function.
        :return: (torch.Tensor) loss.
        """
        tsize = y.size()[1:]

        # sample a splitting
        mask = torch.ones(tsize).to(y.device)
        if not self.regular_mask:
            mask[torch.rand_like(mask) > self.split_ratio] = 0
        else:
            stride = int(1 / (1 - self.split_ratio))
            start = np.random.randint(stride)
            mask[..., start::stride, start::stride] = 0.0

        # create inpainting masks
        inp = Inpainting(tsize, mask)
        inp2 = Inpainting(tsize, 1 - mask)

        # concatenate operators
        physics1 = inp * physics  # A_1 = P*A
        physics2 = inp2 * physics  # A_2 = (I-P)*A

        # divide measurements
        y1 = inp.A(y)
        y2 = inp2.A(y)

        loss_ms = self.metric(physics2.A(model(y1, physics1)), y2)
        loss_ms /= 1 - self.split_ratio  # normalize loss

        return loss_ms


# if __name__ == "__main__":
#     import deepinv as dinv
#
#     sigma = 0.1
#     physics = dinv.physics.Denoising()
#     physics.noise_model = dinv.physics.GaussianNoise(sigma)
#
#     # choose a reconstruction architecture
#     backbone = dinv.models.MedianFilter()
#     f = dinv.models.ArtifactRemoval(backbone)
#     batch_size = 1
#     imsize = (3, 128, 128)
#
#     for split_ratio in np.linspace(0.7, 0.99, 10):
#         x = torch.ones((batch_size,) + imsize, device=dinv.device)
#         y = physics(x)
#
#         # choose training losses
#         loss = SplittingLoss(split_ratio=split_ratio, regular_mask=True)
#         x_net = f(y, physics)
#         mse = dinv.metric.mse()(physics.A(x), physics.A(x_net))
#         split_loss = loss(y, physics, f)
#
#         print(
#             f"split_ratio:{split_ratio:.2f}  mse: {mse:.2e}, split-loss: {split_loss:.2e}"
#         )
#         rel_error = (split_loss - mse).abs() / mse
#         print(f"rel_error: {rel_error:.2f}")
