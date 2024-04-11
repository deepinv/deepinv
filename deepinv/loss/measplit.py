import torch
from deepinv.physics import Inpainting
import numpy as np
from deepinv.loss.loss import Loss


class SplittingLoss(Loss):
    r"""
    Measurement splitting loss.

    Splits the measurement and forward operator (of size :math:`m`)
    into two smaller pairs  :math:`(y_1,A_1)` (of size :math:`m_1`) and  :math:`(y_2,A_2)` (of size :math:`m_2`) ,
    to compute the self-supervised loss:

    .. math::

        \frac{m}{m_2}\| y_2 - A_2 \inversef{y_1}{A_1}\|^2

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
        super().__init__()
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
        inp = Inpainting(tsize, mask, device=y.device)
        inp2 = Inpainting(tsize, 1 - mask, device=y.device)

        # concatenate operators
        physics1 = inp * physics  # A_1 = P*A
        physics2 = inp2 * physics  # A_2 = (I-P)*A

        # divide measurements
        y1 = inp.A(y)
        y2 = inp2.A(y)

        loss_ms = self.metric(physics2.A(model(y1, physics1)), y2)
        loss_ms /= 1 - self.split_ratio  # normalize loss

        return loss_ms


class Neighbor2Neighbor(Loss):
    r"""
    Neighbor2Neighbor loss.

    Splits the noisy measurements using two masks :math:`A_1` and :math:`A_2`, each choosing a different neighboring
    map (see details in `"Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images"
    <https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_Neighbor2Neighbor_Self-Supervised_Denoising_From_Single_Noisy_Images_CVPR_2021_paper.pdf>`_).

    The self-supervised loss is computed as:

    .. math::

        \| A_2 y - R(A_1 y)\|^2 + \gamma \| A_2 y - R(A_1 y) - (A_2 R(y) - A_1 R(y))\|^2

    where :math:`R` is the trainable denoiser network, :math:`\gamma>0` is a regularization parameter
    and no gradient is propagated when computing :math:`R(y)`.

    By default, the error is computed using the MSE metric, however any other metric (e.g., :math:`\ell_1`)
    can be used as well.

    The code has been adapted from the repository https://github.com/TaoHuang2018/Neighbor2Neighbor.

    :param torch.nn.Module metric: metric used for computing data consistency,
        which is set as the mean squared error by default.
    :param float gamma: regularization parameter :math:`\gamma`.
    """

    def __init__(self, metric=torch.nn.MSELoss(), gamma=2.0):
        super().__init__()
        self.name = "neigh2neigh"
        self.metric = metric
        self.gamma = gamma

    def space_to_depth(self, x, block_size):
        n, c, h, w = x.size()
        unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
        return unfolded_x.view(n, c * block_size**2, h // block_size, w // block_size)

    def generate_mask_pair(self, img):
        # prepare masks (N x C x H/2 x W/2)
        n, c, h, w = img.shape
        mask1 = torch.zeros(
            size=(n * h // 2 * w // 2 * 4,), dtype=torch.bool, device=img.device
        )
        mask2 = torch.zeros(
            size=(n * h // 2 * w // 2 * 4,), dtype=torch.bool, device=img.device
        )
        # prepare random mask pairs
        idx_pair = torch.tensor(
            [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
            dtype=torch.int64,
            device=img.device,
        )
        rd_idx = torch.zeros(
            size=(n * h // 2 * w // 2,), dtype=torch.int64, device=img.device
        )
        torch.randint(low=0, high=8, size=(n * h // 2 * w // 2,), out=rd_idx)
        rd_pair_idx = idx_pair[rd_idx]
        rd_pair_idx += torch.arange(
            start=0,
            end=n * h // 2 * w // 2 * 4,
            step=4,
            dtype=torch.int64,
            device=img.device,
        ).reshape(-1, 1)
        # get masks
        mask1[rd_pair_idx[:, 0]] = 1
        mask2[rd_pair_idx[:, 1]] = 1
        return mask1, mask2

    def generate_subimages(self, img, mask):
        n, c, h, w = img.shape
        subimage = torch.zeros(
            n, c, h // 2, w // 2, dtype=img.dtype, layout=img.layout, device=img.device
        )
        # per channel
        for i in range(c):
            img_per_channel = self.space_to_depth(img[:, i : i + 1, :, :], block_size=2)
            img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
            subimage[:, i : i + 1, :, :] = (
                img_per_channel[mask].reshape(n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
            )
        return subimage

    def forward(self, y, physics, model, **kwargs):
        r"""
        Computes the neighbor2neighbor loss.


        :param torch.Tensor y: Measurements.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction function.
        :return: (torch.Tensor) loss.
        """

        assert len(y.shape) == 4, "Input measurements should be images"
        assert (
            y.shape[2] % 2 == 0 and y.shape[3] % 2 == 0
        ), "Image dimensions should be even"

        mask1, mask2 = self.generate_mask_pair(y)

        y1 = self.generate_subimages(y, mask1)
        xhat1 = model(y1, physics)
        y2 = self.generate_subimages(y, mask2)

        xhat = model(y, physics).detach()
        y1_hat = self.generate_subimages(xhat, mask1)
        y2_hat = self.generate_subimages(xhat, mask2)

        loss_n2n = self.metric(xhat1, y2) + self.gamma * self.metric(
            xhat1 - y1_hat, y2 - y2_hat
        )

        return loss_n2n


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
