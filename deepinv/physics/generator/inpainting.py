from typing import Tuple, Union
from warnings import warn
import torch
from deepinv.physics.generator import PhysicsGenerator
from deepinv.physics.functional import random_choice


class BernoulliSplittingMaskGenerator(PhysicsGenerator):
    """Base generator for splitting/inpainting masks.

    Generates binary masks with an approximate given split ratio, according to a Bernoulli distribution. Can be used either for generating random inpainting masks for :class:`deepinv.physics.Inpainting`, or random splitting masks for :class:`deepinv.loss.SplittingLoss`.

    Optional pass in input_mask to subsample this mask given the split ratio. For mask ratio to be almost exactly as specified, use this option with a flat mask of ones as input.

    |sep|

    :Examples:

        Generate random mask

        >>> from deepinv.physics.generator import BernoulliSplittingMaskGenerator
        >>> gen = BernoulliSplittingMaskGenerator((1, 3, 3), split_ratio=0.6)
        >>> gen.step(batch_size=2)["mask"].shape
        torch.Size([2, 1, 3, 3])

        Generate splitting mask from given input_mask

        >>> from deepinv.physics.generator import BernoulliSplittingMaskGenerator
        >>> from deepinv.physics import Inpainting
        >>> physics = Inpainting((1, 3, 3), 0.9)
        >>> gen = BernoulliSplittingMaskGenerator((1, 3, 3), split_ratio=0.6)
        >>> gen.step(batch_size=2, input_mask=physics.mask)["mask"].shape
        torch.Size([2, 1, 3, 3])

    :param tuple[int] tensor_size: size of the tensor to be masked without batch dimension e.g. of shape (C, H, W) or (C, M) or (M,)
    :param float split_ratio: ratio of values to be kept.
    :param bool pixelwise: Apply the mask in a pixelwise fashion, i.e., zero all channels in a given pixel simultaneously.
    :param str, torch.device device: device where the tensor is stored (default: 'cpu').
    :param torch.Generator rng: torch random number generator.
    """

    def __init__(
        self,
        tensor_size: Tuple[int],
        split_ratio: float,
        pixelwise: bool = True,
        random_split_ratio: bool = False,
        min_split_ratio: float = 0.0,
        max_split_ratio: float = 1.0,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        rng: torch.Generator = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, device=device, dtype=dtype, rng=rng, **kwargs)
        self.tensor_size = tensor_size
        self.split_ratio = split_ratio
        self.pixelwise = pixelwise
        self.random_split_ratio = random_split_ratio
        self.min_split_ratio = min_split_ratio
        self.max_split_ratio = max_split_ratio

    def step(
        self, batch_size=1, input_mask: torch.Tensor = None, seed: int = None, **kwargs
    ) -> dict:
        r"""
        Generate a random mask.

        If ``input_mask`` is None, generates a standard random mask that can be used for :class:`deepinv.physics.Inpainting`.
        If ``input_mask`` is specified, splits the input mask into subsets given the split ratio.

        :param int batch_size: batch_size. If None, no batch dimension is created. If input_mask passed and has its own batch dimension > 1, batch_size is ignored.
        :param torch.Tensor, None input_mask: optional mask to be split. If None, all pixels are considered. If not None, only pixels where mask==1 are considered. input_mask shape can optionally include a batch dimension.
        :param int seed: the seed for the random number generator.

        :return: dictionary with key **'mask'**: tensor of size ``(batch_size, *tensor_size)`` with values in {0, 1}.
        :rtype: dict
        """
        self.rng_manual_seed(seed)

        if isinstance(input_mask, torch.Tensor) and len(input_mask.shape) > len(
            self.tensor_size
        ):
            input_mask = input_mask.to(self.device)
            if input_mask.shape[0] > 1:
                # Batch dim exists in input_mask and it's > 1
                batch_size = input_mask.shape[0]
            else:
                # Singular batch dim exists in input_mask so use batch_size
                input_mask = input_mask[0]

        if batch_size is not None:
            # Create each mask in batch independently
            outs = []
            for b in range(batch_size):
                inp = None
                if isinstance(input_mask, torch.Tensor) and len(input_mask.shape) > len(
                    self.tensor_size
                ):
                    inp = input_mask[b]
                elif isinstance(input_mask, torch.Tensor):
                    inp = input_mask
                outs.append(self.batch_step(input_mask=inp, **kwargs))
            mask = torch.stack(outs)
        else:
            mask = self.batch_step(input_mask=input_mask, **kwargs)

        return {"mask": mask}

    def check_pixelwise(self, input_mask=None) -> bool:
        r"""Check if pixelwise can be used given input_mask dimensions and tensor_size dimensions"""
        pixelwise = self.pixelwise

        if pixelwise and len(self.tensor_size) == 2:
            warn(
                "Generating pixelwise mask assumes channel in first dimension. For 2D images (i.e. of shape (H,W)) ensure tensor_size is at least 3D (i.e. C,H,W). However, for tensor_size of shape (C,M), this will work as expected."
            )
        elif pixelwise and len(self.tensor_size) == 1:
            warn("For 1D tensor_size, pixelwise must be False.")
            pixelwise = False

        if (
            isinstance(input_mask, torch.Tensor) and input_mask.numel() > 1
        ):  # Input mask is properly specified
            if pixelwise:
                if len(input_mask.shape) == 1:
                    warn("input_mask is only 1D so pixelwise cannot be used.")
                    return False
                elif len(input_mask.shape) == 2 and len(input_mask.shape) < len(
                    self.tensor_size
                ):
                    # When input_mask 2D, this can either be shape C,M or H,W.
                    # When input_mask C,M, tensor_size will also be C,M (as passed in from SplittingLoss) and pixelwise can be used safely.
                    # When input_mask H,W but tensor_size higher-dimensional e.g. C,H,W, then pixelwise should be set to False as it will happen anyway.
                    return False
                elif not all(
                    torch.equal(input_mask[i], input_mask[0])
                    for i in range(1, input_mask.shape[0])
                ):
                    warn("To use pixelwise, all channels must be same.")
                    return False

        return pixelwise

    def batch_step(self, input_mask: torch.Tensor = None) -> dict:
        r"""
        Create one batch of splitting mask.

        :param torch.Tensor, None input_mask: optional mask to be split. If ``None``, all pixels are considered. If not ``None``, only pixels where ``mask==1`` are considered. Batch dimension should not be included in shape.
        """
        pixelwise = self.check_pixelwise(input_mask)

        if self.random_split_ratio:
            self.split_ratio = (
                torch.rand(batch_size, generator=self.rng, **self.factory_kwargs)
                * (self.sigma_max - self.sigma_min)
                + self.sigma_min
            )

        if isinstance(input_mask, torch.Tensor) and input_mask.numel() > 1:
            input_mask = input_mask.to(self.device)
            # Sample indices from given input mask
            if pixelwise:
                idx = input_mask[0, ...].nonzero(as_tuple=False)
            else:
                idx = input_mask.nonzero(as_tuple=False)

            shuff = idx[
                torch.randperm(len(idx), generator=self.rng, device=self.device)
            ]
            idx_out = shuff[: int(self.split_ratio * len(idx))].t()

            mask = torch.zeros_like(input_mask)

            if pixelwise:
                mask = mask[0, ...]
                mask[tuple(idx_out)] = 1
                mask = torch.stack([mask] * input_mask.shape[0])
            else:
                mask[tuple(idx_out)] = 1

        else:
            # Sample pixels from a uniform distribution as input_mask is not given
            mask = torch.ones(self.tensor_size, device=self.device)
            aux = torch.rand(self.tensor_size, generator=self.rng, device=self.device)
            if not pixelwise:
                mask[aux > self.split_ratio] = 0
            else:
                mask[:, aux[0, ...] > self.split_ratio] = 0

        return mask


class GaussianSplittingMaskGenerator(BernoulliSplittingMaskGenerator):
    """Randomly generate Gaussian splitting/inpainting masks.

    Generates binary masks with an approximate given split ratio, where samples are weighted according to a spatial Gaussian distribution, where pixels near the center are less likely to be kept.
    This mask is used for measurement splitting for MRI in `SSDU <https://pubmed.ncbi.nlm.nih.gov/32614100/>`_.

    Can be used either for generating random inpainting masks for :class:`deepinv.physics.Inpainting`, or random splitting masks for :class:`deepinv.loss.SplittingLoss`.

    Optional pass in input_mask to subsample this mask given the split ratio.

    Handles both 2D mask (i.e. [C, H, W] from `SSDU <https://pubmed.ncbi.nlm.nih.gov/32614100/>`_) and 2D+time dynamic mask (i.e. [C, T, H, W] from `Acar et al. <https://link.springer.com/chapter/10.1007/978-3-030-88552-6_4>`_) generation. Does not handle 1D data (e.g. of shape [C, M])

    |sep|

    :Examples:

        Randomly split input mask using Gaussian weighting

        >>> from deepinv.physics.generator import GaussianSplittingMaskGenerator
        >>> from deepinv.physics import Inpainting
        >>> physics = Inpainting((1, 3, 3), 0.9)
        >>> gen = GaussianSplittingMaskGenerator((1, 3, 3), split_ratio=0.6, center_block=0)
        >>> gen.step(batch_size=2, input_mask=physics.mask)["mask"].shape
        torch.Size([2, 1, 3, 3])

    :param tuple[int] tensor_size: size of the tensor to be masked without batch dimension e.g. of shape (C, H, W) or (C, T, H, W)
    :param float split_ratio: ratio of values to be kept (i.e. ones).
    :param bool pixelwise: Apply the mask in a pixelwise fashion, i.e., zero all channels in a given pixel simultaneously.
    :param float std_scale: scale parameter of 2D Gaussian, in pixels.
    :param int, tuple[int] center_block: size of block in image center that is always kept for MRI autocalibration signal. Either int for square block or 2-tuple (h, w)
    :param str, torch.device device: device where the tensor is stored (default: 'cpu').
    :param torch.Generator rng: random number generator.
    """

    def __init__(
        self,
        tensor_size: Tuple[int],
        split_ratio: float,
        pixelwise: bool = True,
        std_scale: float = 4.0,
        center_block: Union[Tuple[int], int] = (8, 8),
        device: torch.device = torch.device("cpu"),
        rng: torch.Generator = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            tensor_size=tensor_size,
            split_ratio=split_ratio,
            pixelwise=pixelwise,
            device=device,
            rng=rng,
            **kwargs,
        )
        if len(tensor_size) < 3:
            raise ValueError(
                "tensor_size should be at least of shape (C, H, W). Gaussian splitting mask does not support signals of shape (C, M)."
            )
        self.std_scale = std_scale
        self.center_block = (
            (center_block, center_block)
            if isinstance(center_block, int)
            else center_block
        )

    def batch_step(self, input_mask: torch.Tensor = None) -> dict:
        r"""
        Create one batch of splitting mask using Gaussian distribution.

        Adapted from https://github.com/byaman14/SSDU/blob/main/masks/ssdu_masks.py from `SSDU <https://pubmed.ncbi.nlm.nih.gov/32614100/>`_.

        :param torch.Tensor, None input_mask: optional mask to be split. If None, all pixels are considered. If not None, only pixels where mask==1 are considered. No batch dim in shape.
        """
        pixelwise = self.check_pixelwise()
        _T = self.tensor_size[1] if len(self.tensor_size) > 3 else 1
        _C = self.tensor_size[0] if not pixelwise else 1

        # Create blank input mask if not specified. Create with time dim even if we only want static mask
        if not isinstance(input_mask, torch.Tensor) or input_mask.numel() <= 1:
            input_mask = torch.ones(_C, _T, *self.tensor_size[-2:], device=self.device)

        if len(input_mask.shape) < len(self.tensor_size):
            # Missing channel dim, so create it
            no_channel_dim = True
            input_mask = input_mask.unsqueeze(0)
            _C = 1
        else:
            no_channel_dim = False

        if len(input_mask.shape) == 3:
            # Create time dim even if we only want static mask
            input_mask = input_mask.unsqueeze(1)

        if pixelwise:
            # Only use one channel (they are all the same...)
            input_mask = input_mask[[0], ...]

        nx, ny = input_mask.shape[-2:]
        centerx, centery = nx // 2, ny // 2

        x, y = torch.meshgrid(
            torch.arange(0, nx, 1, device=self.device),
            torch.arange(0, ny, 1, device=self.device),
            indexing="ij",
        )

        # Create PDF
        gaussian = torch.exp(
            -(
                (x - centerx) ** 2 / (2 * (nx / self.std_scale) ** 2)
                + (y - centery) ** 2 / (2 * (ny / self.std_scale) ** 2)
            )
        )
        prob_mask = input_mask * gaussian[..., :, :]

        prob_mask[
            ...,
            centerx - self.center_block[0] // 2 : centerx + self.center_block[0] // 2,
            centery - self.center_block[1] // 2 : centery + self.center_block[1] // 2,
        ] = 0

        norm_prob = prob_mask / prob_mask.sum(dim=(-2, -1), keepdim=True)

        # Fill output mask
        mask_out = torch.zeros_like(input_mask).flatten(-2)

        for c in range(_C):
            for t in range(_T):
                ind = random_choice(
                    nx * ny,
                    size=(input_mask[c, t, :, :].sum() * (1 - self.split_ratio))
                    .ceil()
                    .int()
                    .item(),
                    p=norm_prob[c, t, :, :].flatten(),
                    replace=False,
                    rng=self.rng,
                )
                mask_out[c, t, ind] = 1

        # Invert mask for output and handle dimensions
        mask_out = input_mask - mask_out.unflatten(-1, (nx, ny))

        if len(self.tensor_size) == 3:
            mask_out = mask_out[:, 0, ...]  # no actual time dim

        if self.pixelwise and not no_channel_dim:
            mask_out = torch.cat([mask_out] * self.tensor_size[0], dim=0)

        return mask_out


class Phase2PhaseSplittingMaskGenerator(BernoulliSplittingMaskGenerator):
    """Phase2Phase splitting mask generator for dynamic data.

    To be exclusively used with :class:`deepinv.loss.Phase2PhaseLoss`.
    Splits dynamic data (i.e. data of shape (B, C, T, H, W)) into even and odd phases in the T dimension.

    Used in `Phase2Phase: Respiratory Motion-Resolved Reconstruction of Free-Breathing Magnetic Resonance Imaging Using Deep Learning Without a Ground Truth for Improved Liver Imaging <https://journals.lww.com/investigativeradiology/abstract/2021/12000/phase2phase__respiratory_motion_resolved.4.aspx>`_
    for free-breathing MRI.

    If input_mask not passed, a blank input mask is used instead.

    :param tuple[int] tensor_size: size of the tensor to be masked without batch dimension of shape (C, T, H, W)
    :param str, torch.device device: device where the tensor is stored (default: 'cpu').
    :param torch.Generator rng: unused.
    """

    def __init__(
        self,
        tensor_size: Tuple[int],
        device: torch.device = "cpu",
        rng: torch.Generator = None,
    ):
        super().__init__(
            tensor_size=tensor_size,
            split_ratio=None,
            pixelwise=None,
            device=device,
            rng=rng,
        )

    def batch_step(self, input_mask: torch.Tensor = None) -> dict:
        if len(self.tensor_size) != 4:
            raise ValueError("tensor_size must be of shape (C, T, H, W)")

        if not isinstance(input_mask, torch.Tensor) or input_mask.numel() <= 1:
            input_mask = torch.ones(self.tensor_size, device=self.device)

        if tuple(input_mask.shape) != self.tensor_size:
            raise ValueError("input_mask must be same shape as tensor_size")

        mask_out = torch.zeros_like(input_mask)
        mask_out[:, ::2] = input_mask[:, ::2]
        return mask_out


class Artifact2ArtifactSplittingMaskGenerator(Phase2PhaseSplittingMaskGenerator):
    """Artifact2Artifact splitting mask generator for dynamic data.

    To be exclusively used with :class:`deepinv.loss.Artifact2ArtifactLoss`.
    Randomly selects a chunk from dynamic data (i.e. data of shape (B, C, T, H, W)) in the T dimension and puts zeros in the rest of the mask.

    When ``step`` called with ``persist_prev``, the selected chunk will be different from the previous time it was called.
    This is used so input chunk is compared to a different output chunk.

    Artifact2Artifact was introduced in `RARE: Image Reconstruction using Deep Priors Learned without Ground Truth <https://arxiv.org/abs/1912.05854>`_
    for free-breathing MRI.

    If input_mask not passed, a blank input mask is used instead.

    :param tuple[int] tensor_size: size of the tensor to be masked without batch dimension of shape (C, T, H, W)
    :param int, tuple[int] split_size: time-length of chunk. Must divide ``tensor_size[1]`` exactly. If ``tuple``, one is randomly selected each time.
    :param str, torch.device device: device where the tensor is stored (default: 'cpu').
    :param torch.Generator rng: torch random number generator.
    """

    def __init__(
        self,
        tensor_size: Tuple[int],
        split_size: Union[int, Tuple[int]] = 2,
        device: torch.device = "cpu",
        rng: torch.Generator = None,
    ):
        super().__init__(tensor_size, device, rng=rng)
        self.split_size = split_size
        self.prev_idx = None
        self.prev_split_size = None

    def batch_step(
        self, input_mask: torch.Tensor = None, persist_prev: bool = False
    ) -> dict:
        def rand_select(arr):
            return arr[
                torch.randint(
                    len(arr), (1,), generator=self.rng, device=self.device
                ).item()
            ]

        # Do Phase2Phase step to check input dimensions
        _ = super().batch_step(input_mask=input_mask)

        # Choose split_size
        split_size = self.split_size
        if isinstance(self.split_size, (tuple, list)):
            if persist_prev:
                split_size = self.prev_split_size
            else:
                self.prev_split_size = split_size = rand_select(self.split_size)

        # Randomly select one chunk. Don't select previous chunk if leave_prev_idx is True
        idxs = list(range(input_mask.shape[1] // split_size))
        if persist_prev:
            idxs.remove(self.prev_idx)

        self.prev_idx = idx = rand_select(idxs)

        mask_out = torch.zeros_like(input_mask)
        mask_out[:, split_size * idx : split_size * (idx + 1)] = input_mask[
            :, split_size * idx : split_size * (idx + 1)
        ]
        return mask_out
