import torch
from deepinv.physics import Inpainting
import numpy as np
from deepinv.loss.loss import Loss


def get_mask(tensor_size, split_ratio, regular_mask=False, device="cpu"):
    # sample a splitting
    mask = torch.ones(tensor_size).to(device)
    if not regular_mask:
        mask[torch.rand_like(mask) > split_ratio] = 0
    else:
        stride = int(1 / (1 - split_ratio))
        start = np.random.randint(stride)
        mask[..., start::stride, start::stride] = 0.0
    return mask


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

    .. warning::

        The model should be adapted before training using the method :meth:`adapt_model` to include the splitting mechanism at the input.

    .. note::

        To obtain the best test performance, the trained model should be averaged at test time
        over multiple realizations of the splitting, i.e.
        :math:`\hat{x} = \frac{1}{N}\sum_{i=1}^N \inversef{y_1^{(i)}}{A_1^{(i)}}`.

    :param torch.nn.Module metric: metric used for computing data consistency,
        which is set as the mean squared error by default.
    :param float split_ratio: splitting ratio, should be between 0 and 1. The size of :math:`y_1` increases
        with the splitting ratio.
    :param bool regular_mask: If ``True``, it will use a regular mask, otherwise it uses a random mask.

    |sep|

    :Example:

    >>> import torch
    >>> import deepinv as dinv
    >>> physics = dinv.physics.Inpainting()
    >>> model = dinv.models.MedianFilter()
    >>> loss = dinv.loss.SplittingLoss(split_ratio=0.9)
    >>> model = loss.adapt_model(model, MC_samples=2) # important step!
    >>> x = torch.ones((1, 1, 8, 8))
    >>> y = physics(x)
    >>> x_net = model(y, physics)
    >>> l = loss(x_net, y, physics, model)
    >>> print(l > 0)
    True



    """

    def __init__(self, metric=torch.nn.MSELoss(), split_ratio=0.9, regular_mask=False):
        super().__init__()
        self.name = "ms"
        self.metric = metric
        self.regular_mask = regular_mask
        self.split_ratio = split_ratio

    def forward(self, x_net, y, physics, model, **kwargs):
        r"""
        Computes the measurement splitting loss

        :param torch.Tensor y: Measurements.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction function.
        :return: (torch.Tensor) loss.
        """
        tsize = y.size()[1:]
        mask = model.get_mask()

        # create inpainting masks
        inp2 = Inpainting(tsize, 1 - mask, device=y.device)

        # concatenate operators
        physics2 = inp2 * physics  # A_2 = (I-P)*A

        # divide measurements
        y2 = inp2.A(y)

        loss_ms = self.metric(physics2.A(x_net), y2)
        loss_ms /= 1 - self.split_ratio  # normalize loss

        return loss_ms

    def adapt_model(self, model, MC_samples=5):
        r"""
        Apply random splitting to input.

        This method modifies a reconstruction
        model :math:`R` to include the splitting mechanism at the input:

        .. math::

            \hat{R}(y, A) = \frac{1}{N}\sum_{i=1}^N \inversef{y_1^{(i)}}{A_1^{(i)}}

        where :math:`N\geq 1` is the number of Monte Carlo samples,
        and :math:`y_1^{(i)}` and :math:`A_1^{(i)}` are obtained by
        randomly splitting the measurements :math:`y` and operator :math:`A`.
        During training (i.e. when ``model.train()``), we use only one sample, i.e. :math:`N=1`
        for computational efficiency, whereas at test time, we use multiple samples for better performance.

        :param torch.nn.Module model: Reconstruction model.
        :param int MC_samples: Number of samples used for averaging.
        :return: (torch.nn.Module) Model modified for evaluation.
        """

        return SplittingModel(model, self.split_ratio, self.regular_mask, MC_samples)


class SplittingModel(torch.nn.Module):
    def __init__(self, model, split_ratio, regular_mask, MC_samples):
        super().__init__()
        self.model = model
        self.split_ratio = split_ratio
        self.regular_mask = regular_mask
        self.MC_samples = MC_samples
        self.mask = 0

    def forward(self, y, physics):
        MC = 1 if self.training else self.MC_samples
        tsize = y.size()[1:]
        out = 0
        with torch.set_grad_enabled(self.training):
            for i in range(MC):
                mask = get_mask(
                    tsize, self.split_ratio, self.regular_mask, device=y.device
                )
                inp = Inpainting(tsize, mask, device=y.device)
                y1 = inp.A(y)
                physics1 = inp * physics  # A_1 = P*A
                out += self.model(y1, physics1)

            if self.training:
                self.mask = mask
            out = out / MC
        return out

    def get_mask(self):
        return self.mask


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
