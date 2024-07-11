from typing import Optional
from warnings import warn
import torch
from deepinv.physics import Inpainting, Denoising, Physics
from deepinv.loss.loss import Loss
from deepinv.physics.generator import PhysicsGenerator, BernoulliSplittingMaskGenerator


class SplittingLoss(Loss):
    r"""
    Measurement splitting loss.

    Splits the measurement and forward operator :math:`\forw{}` (of size :math:`m`)
    into two smaller pairs  :math:`(y_1,A_1)` (of size :math:`m_1`) and  :math:`(y_2,A_2)` (of size :math:`m_2`) ,
    to compute the self-supervised loss:

    .. math::

        \frac{m}{m_2}\| y_2 - A_2 \inversef{y_1}{A_1}\|^2

    where :math:`R` is the trainable network, :math:`A_1 = M_1 \forw{}, A_2 = M_2 \forw{}`, and :math:`M_i` are randomly generated masks (i.e. diagonal matrices) such that :math:`M_1+M_2=\mathbb{I}_m`.

    .. note::

        If the forward operator has its own subsampling mask :math:`M_{\forw{}}`, e.g. :class:`deepinv.physics.Inpainting` or :class:`deepinv.physics.MRI`,
        the splitting masks will be subsets of the physics' mask such that :math:`M_1+M_2=M_{\forw{}}`

    This loss was used in SSDU for MRI in `Yaman et al. Self-supervised learning of physics-guided reconstruction neural networks without fully sampled reference data <https://pubmed.ncbi.nlm.nih.gov/32614100/>`_

    By default, the error is computed using the MSE metric, however any other metric (e.g., :math:`\ell_1`)
    can be used as well.

    .. warning::

        The model should be adapted before training using the method :meth:`adapt_model` to include the splitting mechanism at the input.

    .. note::

        To obtain the best test performance, the trained model should be averaged at test time
        over multiple realizations of the splitting, i.e.
        :math:`\hat{x} = \frac{1}{N}\sum_{i=1}^N \inversef{y_1^{(i)}}{A_1^{(i)}}`. To disable this, set ``MC_samples=1``.

    .. note::

        To disable measurement splitting (and use the full input) at evaluation time, set ``eval_split_input=True``. This is done in `SSDU <https://pubmed.ncbi.nlm.nih.gov/32614100/>`_.

    :param torch.nn.Module metric: metric used for computing data consistency,
        which is set as the mean squared error by default.
    :param float split_ratio: splitting ratio, should be between 0 and 1. The size of :math:`y_1` increases
        with the splitting ratio.
    :param deepinv.physics.generator.PhysicsGenerator, None mask_generator: function to generate the mask. If
        None, the :class:`deepinv.physics.generator.BernoulliSplittingMaskGenerator` is used.
    :param int MC_samples: Number of samples used for averaging. Must be greater than 0.
    :param bool eval_split_input: if True, perform input measurement splitting during evaluation. If False, use full measurement at eval (no MC samples are performed and eval_split_output will have no effect)
    :param bool eval_split_output: at evaluation time, pass the output through the output mask too. 
        i.e. :math:`(\sum_{j=1}^N M_2^{(j)})^{-1} \sum_{i=1}^N M_2^{(i)} \inversef{y_1^{(i)}}{A_1^{(i)}}`.
        Only valid when y is same domain (and dimension) as x. Defaults to False.
    :param bool pixelwise: if True, create pixelwise splitting masks i.e. zero all channels simultaneously.

    |sep|

    :Example:

    >>> import torch
    >>> import deepinv as dinv
    >>> physics = dinv.physics.Inpainting(tensor_size=(1, 8, 8), mask=0.5)
    >>> model = dinv.models.MedianFilter()
    >>> loss = dinv.loss.SplittingLoss(split_ratio=0.9, MC_samples=2)
    >>> model = loss.adapt_model(model) # important step!
    >>> x = torch.ones((1, 1, 8, 8))
    >>> y = physics(x)
    >>> x_net = model(y, physics, update_parameters=True) # save random mask in forward pass
    >>> l = loss(x_net, y, physics, model)
    >>> print(l.item() > 0)
    True


    """

    def __init__(
        self,
        metric=torch.nn.MSELoss(),
        split_ratio: float = 0.9,
        mask_generator: Optional[PhysicsGenerator] = None,
        MC_samples=5,
        eval_split_input=True,
        eval_split_output=False,
        pixelwise=True,
    ):
        super().__init__()
        self.name = "ms"
        self.metric = metric
        self.mask_generator = mask_generator
        self.split_ratio = split_ratio
        self.MC_samples = MC_samples
        self.eval_split_input = eval_split_input
        self.eval_split_output = eval_split_output
        self.pixelwise = pixelwise

    def forward(self, x_net, y, physics, model, **kwargs):
        r"""
        Computes the measurement splitting loss

        :param torch.Tensor y: Measurements.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction function.
        :return: (torch.Tensor) loss.
        """
        mask = model.get_mask()

        # create inpainting masks
        mask2 = getattr(physics, "mask", 1.0) - mask
        inp2 = Inpainting(y.size()[1:], mask=mask2, device=y.device)

        # concatenate operators
        physics2 = inp2 * physics  # A_2 = (I-P)*A
        physics2.noise_model = physics.noise_model

        # divide measurements
        y2 = inp2.A(y)

        loss_ms = self.metric(physics2.A(x_net), y2)

        loss_ms = loss_ms / mask2.mean()  # normalize loss

        return loss_ms

    def adapt_model(self, model: torch.nn.Module, MC_samples=None):
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
        For other parameters that control how splitting is applied, see the class parameters.

        :param torch.nn.Module model: Reconstruction model.
        :param int MC_samples: deprecated. Pass ``MC_samples`` at class initialisation instead.
        :return: (torch.nn.Module) Model modified for evaluation.
        """
        if MC_samples is not None:
            warn(
                "MC_samples parameter is deprecated. Pass MC_samples at init: SplittingLoss(MC_samples=...)"
            )

        if isinstance(model, SplittingModel):
            return model
        else:
            return SplittingModel(
                model,
                split_ratio=self.split_ratio,
                mask_generator=self.mask_generator,
                MC_samples=self.MC_samples,
                eval_split_input=self.eval_split_input,
                eval_split_output=self.eval_split_output,
                pixelwise=self.pixelwise,
            )


class SplittingModel(torch.nn.Module):
    """
    Model wrapper when using SplittingLoss.

    Performs input splitting during forward pass. At evaluation, perform forward passes for multiple realisations of splitting mask and average.

    For parameters, see :class:`deepinv.loss.SplittingLoss`
    """

    def __init__(
        self,
        model,
        split_ratio,
        mask_generator,
        MC_samples,
        eval_split_input,
        eval_split_output,
        pixelwise,
    ):
        super().__init__()
        self.model = model
        self.split_ratio = split_ratio
        self.MC_samples = MC_samples
        self.mask = 0
        self.mask_generator = mask_generator
        self.eval_split_input = eval_split_input
        self.eval_split_output = eval_split_output
        self.pixelwise = pixelwise

    def forward(
        self, y: torch.Tensor, physics: Physics, update_parameters: bool = False
    ):
        """Adapted model forward pass for input splitting. During training, only one splitting realisation is performed for computational efficiency.
        """
        out = 0

        if self.mask_generator is None:
            self.mask_generator = BernoulliSplittingMaskGenerator(
                tensor_size=y.size()[1:],
                split_ratio=self.split_ratio,
                pixelwise=self.pixelwise,
                device=y.device,
            )

        inp = Inpainting(y.size()[1:], device=y.device)

        with torch.set_grad_enabled(self.training):

            if not self.eval_split_input and not self.training:
                return self.model(y, physics)
            elif self.eval_split_output and self.eval_split_input and not self.training:
                return self._forward_split_output(y)
            else:
                MC_samples = 1 if self.training else self.MC_samples

                for _ in range(MC_samples):
                    # Perform input masking
                    mask = self.mask_generator.step(y.size(0), input_mask=getattr(physics, "mask", None))
                    y1 = inp.A(y, **mask)
                    physics1 = inp * physics  # A_1 = P*A
                    physics1.noise_model = physics.noise_model

                    # Forward pass
                    out += self.model(y1, physics1) / MC_samples

            if self.training and update_parameters:
                self.mask = mask["mask"]

        return out

    def _forward_split_output(self, y: torch.Tensor):
        """Perform splitting at model output too, only at eval time
        """
        out = 0
        normaliser = torch.zeros_like(y)
        inp = Inpainting(y.size()[1:], device=y.device)

        for _ in range(self.MC_samples):
            # Perform input masking
            mask = self.mask_generator.step(y.size(0), input_mask=getattr(physics, "mask", None))
            y1 = inp.A(y, **mask)
            physics1 = inp * physics  # A_1 = P*A
            physics1.noise_model = physics.noise_model

            # Forward pass
            x_hat = self.model(y1, physics1)

            # Output masking
            mask2 = getattr(physics, "mask", 1.0) - mask["mask"]
            inp2 = Inpainting(y.size()[1:], mask=mask2, device=y.device)
            out += inp2.A(x_hat)
            normaliser += mask2
        
        out[normaliser != 0] /= normaliser[normaliser != 0]

        return out

    def get_mask(self):
        if not isinstance(self.mask, torch.Tensor):
            raise ValueError(
                "Mask not generated during forward pass - use model(y, physics, update_parameters=True)"
            )
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


if __name__ == "__main__":
    import deepinv as dinv
    import torch
    import numpy as np

    sigma = 0.1
    physics = dinv.physics.Denoising()
    physics.noise_model = dinv.physics.GaussianNoise(sigma)
    # choose a reconstruction architecture
    backbone = dinv.models.MedianFilter()
    f = dinv.models.ArtifactRemoval(backbone)
    # choose training losses
    split_ratio = 0.9
    loss = SplittingLoss(split_ratio=split_ratio)
    f = loss.adapt_model(f, MC_samples=2)  # important step!

    batch_size = 1
    imsize = (3, 128, 128)
    device = "cuda"

    x = torch.ones((batch_size,) + imsize, device=device)
    y = physics(x)

    x_net = f(y, physics)
    mse = dinv.metric.mse()(physics.A(x), physics.A(x_net))
    split_loss = loss(y=y, x_net=x_net, physics=physics, model=f)

    print(
        f"split_ratio:{split_ratio:.2f}  mse: {mse:.2e}, split-loss: {split_loss:.2e}"
    )
    rel_error = (split_loss - mse).abs() / mse
    print(f"rel_error: {rel_error:.2f}")
