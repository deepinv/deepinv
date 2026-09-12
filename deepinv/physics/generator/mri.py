from __future__ import annotations
from abc import ABC, abstractmethod
import warnings

import torch

from deepinv.physics.generator import PhysicsGenerator
from deepinv.physics.functional.rand import random_choice


def ceildiv(a: float, b: float) -> float:
    return -(a // -b)


class BaseMaskGenerator(PhysicsGenerator, ABC):
    """Base generator for MRI acceleration masks.

    Generate a mask of vertical lines for MRI acceleration with fixed sampling in low frequencies (center of k-space) and undersampling in the high frequencies.

    The type of undersampling is determined by the child class. The mask is repeated across channels and randomly varies across batch dimension.

    :param tuple img_size: image size, either (H, W) or (C, H, W) or (C, T, H, W), where optional C is channels, and optional T is number of time-steps
    :param int acceleration: acceleration factor, defaults to 4
    :param float center_fraction: fraction of lines to sample in low frequencies (center of k-space). If 0, there is no fixed low-freq sampling. Defaults to None.
    :param torch.Generator rng: torch random generator. Defaults to None.
    """

    def __init__(
        self,
        img_size: tuple,
        acceleration: int = 4,
        center_fraction: float | None = None,
        rng: torch.Generator = None,
        device="cpu",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs, rng=rng, device=device)
        self.img_size = img_size
        self.acc = acceleration

        # Set default center_fraction if not provided
        if center_fraction is not None:
            self.center_fraction = center_fraction
        elif acceleration < 8:
            self.center_fraction = 0.08
        elif acceleration >= 8:
            self.center_fraction = 0.04

        if len(self.img_size) == 2:
            self.H, self.W = self.img_size
            self.C, self.T = 1, 0
        elif len(self.img_size) == 3:
            self.C, self.H, self.W = self.img_size
            self.T = 0
        elif len(self.img_size) == 4:
            self.C, self.T, self.H, self.W = self.img_size
        else:
            raise ValueError("img_size must be (H, W) or (C, H, W) or (C, T, H, W)")

        self.calculate_lines(self.W)

    def calculate_lines(self, W: int):
        """Calculate number of lines and center lines from total width.

        :param int W: total number of columns (width of mask)
        """
        self.n_center = int(self.center_fraction * W)
        self.n_lines = int(W // self.acc - self.n_center)

        if self.n_lines < 0:
            raise ValueError(
                "center_fraction is too high for this acceleration factor."
            )
        elif self.n_lines == 0:
            warnings.warn(
                "Number of high frequency lines to be sampled is 0. Reduce acceleration factor or reduce center_fraction."
            )

    @abstractmethod
    def sample_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Given empty mask, sample lines according to child class sampling strategy.

        This must be implemented in child classes. Time dimension is specified but can be ignored if needed.

        :param torch.Tensor mask: empty mask of shape (B, C, T, H, W)
        :return torch.Tensor: sampled mask of shape (B, C, T, H, W)
        """
        pass

    @abstractmethod
    def get_pdf(self) -> torch.Tensor:
        """Get mask probability density function (PDF).

        :return torch.Tensor: unnormalized 1D vector representing pdf evaluated across mask columns.
        """
        pass

    def step(
        self, batch_size=1, seed: int = None, img_size: tuple | None = None, **kwargs
    ) -> dict:
        r"""
        Create a mask of vertical lines.

        :param int batch_size: batch_size.
        :param int seed: optional: the seed for the random number generator, to reseed on-the-fly.
        :param tuple img_size: if not `None`, generate masks of this 2D image shape and override `img_size` attribute, must be of form `(H, W)`.

        :return: dictionary with key **'mask'**: tensor of size (batch_size, C, H, W) or (batch_size, C, T, H, W) with values in {0, 1}.
        :rtype: dict
        """
        self.rng_manual_seed(seed)

        _B = 1 if batch_size == 0 else batch_size
        _T = self.T if self.T > 0 else 1
        _H, _W = (self.H, self.W) if img_size is None else img_size

        self.calculate_lines(_W)

        mask = torch.zeros((_B, self.C, _T, _H, _W), **self.factory_kwargs)

        if self.n_lines + self.n_center >= _W:
            mask += 1
        else:
            mask = self.sample_mask(mask)

        if self.T == 0:
            mask = mask[:, :, 0, :, :]

        if batch_size == 0:
            mask = mask[0]

        return {"mask": mask}


class SequentialMaskGenerator(PhysicsGenerator):
    r"""Generates a sequential Cartesian mask.

    The wrapped generator first generates a mask, and then selects the set of non-zero mask columns.
    It then creates an indexed mask with one column per selected time stamp, ordered from
    left to right by default. The temporal union is exactly the wrapped static
    mask.

    The sequential Cartesian mask is constructed from a static mask as

    .. math::

        M^{\mathrm{seq}}_t
        =
        M^{\mathrm{static}} \odot L_{\ell_t},
        \qquad t=0,\ldots,T-1,

    where :math:`M^{\mathrm{static}}` is a static Cartesian mask,
    :math:`L_{\ell_t}` is a binary mask selecting the :math:`\ell_t`-th
    k-space column, and :math:`\odot` denotes elementwise multiplication.
    The sequence of selected columns is

    .. math::

        (\ell_0,\ldots,\ell_{T-1})
        =
        \operatorname{sort}\!\left(
            \operatorname{colsupp}(M^{\mathrm{static}})
        \right),

    where :math:`\operatorname{colsupp}` denotes the set of sampled columns.
    The ordering is reversed when ``reverse=True``.


    .. note::
        By construction, the sum of temporal masks is equal to the static mask, i.e.

        .. math::

            \sum_{t=0}^{T-1} M^{\mathrm{seq}}_t
            =
            M^{\mathrm{static}}.


    :param BaseMaskGenerator spatial_generator: static Cartesian mask generator,
        configured with image size ``(H,W)`` or ``(C,H,W)``.
    :param bool reverse: acquire selected columns from right to left, defaults
        to ``False``.

    |sep|

    :Example:

    >>> spatial = EquispacedMaskGenerator((2, 8, 16), acceleration=4)
    >>> generator = SequentialMaskGenerator(spatial)
    >>> mask = generator.step(batch_size=1, seed=0)["mask"]
    >>> mask.shape  # (B,C,T,H,W), with T = W // acceleration = 4
    torch.Size([1, 2, 4, 8, 16])
    >>> torch.equal(mask.amax(dim=2), spatial.step(batch_size=1, seed=0)["mask"])
    True
    """

    def __init__(self, spatial_generator: BaseMaskGenerator, reverse: bool = False):
        if spatial_generator.T != 0:
            raise ValueError(
                "spatial_generator must generate a static mask from img_size "
                "(H,W) or (C,H,W)."
            )
        super().__init__(
            rng=spatial_generator.rng,
            device=spatial_generator.device,
            dtype=spatial_generator.factory_kwargs["dtype"],
        )
        self.spatial_generator = spatial_generator
        self.reverse = reverse

    def step(
        self, batch_size: int = 1, seed: int = None, img_size: tuple = None, **kwargs
    ) -> dict:
        r"""
        Creates a mask of vertical lines.

        :param int batch_size: batch_size.
        :param int seed: optional: the seed for the random number generator, to reseed on-the-fly.
        :param tuple img_size: if not `None`, generate masks of this 2D image shape and override `img_size` attribute, must be of form `(H, W)`.

        :return: dictionary with key **'mask'**: tensor of size (batch_size, C, T, H, W).
        :rtype: dict
        """
        spatial_mask = self.spatial_generator.step(
            batch_size=batch_size, seed=seed, img_size=img_size, **kwargs
        )["mask"]
        squeeze_batch = batch_size == 0
        if squeeze_batch:
            spatial_mask = spatial_mask.unsqueeze(0)
        if spatial_mask.ndim != 4:
            raise ValueError(
                "spatial_generator must return a mask of shape (B,C,H,W), "
                f"but got {tuple(spatial_mask.shape)}."
            )

        selected_columns = [
            spatial_mask[b].bool().any(dim=(0, 1)).nonzero().flatten()
            for b in range(spatial_mask.shape[0])
        ]
        time_size = max(columns.numel() for columns in selected_columns)
        temporal_mask = spatial_mask.new_zeros(
            (*spatial_mask.shape[:2], time_size, *spatial_mask.shape[-2:])
        )
        for b, columns in enumerate(selected_columns):
            columns = columns.flip(0) if self.reverse else columns
            for t, column in enumerate(columns):
                temporal_mask[b, :, t, :, column] = spatial_mask[b, :, :, column]

        if squeeze_batch:
            temporal_mask = temporal_mask[0]
        return {"mask": temporal_mask}


class RandomMaskGenerator(BaseMaskGenerator):
    """Generator for MRI Cartesian acceleration masks using random uniform undersampling.

    Generate a mask of vertical lines for MRI acceleration with fixed sampling in low frequencies (center of k-space) and random uniform undersampling in the high frequencies.

    Supports k-t sampling, where the mask is selected randomly across time.

    The mask is repeated across channels and randomly varies across batch dimension.

    For parameter descriptions see :class:`deepinv.physics.generator.mri.BaseMaskGenerator`

    |sep|

    :Examples:

        Random k-t mask generator for a 8x64x64 video:

        >>> from deepinv.physics.generator import RandomMaskGenerator
        >>> generator = RandomMaskGenerator((2, 8, 64, 64), acceleration=8, center_fraction=0.04) # C, T, H, W
        >>> params = generator.step(batch_size=1)
        >>> mask = params["mask"]
        >>> mask.shape
        torch.Size([1, 2, 8, 64, 64])

    """

    def get_pdf(self, W: int) -> torch.Tensor:
        """Create one-dimensional uniform probability density function across columns, ignoring any fixed sampling columns.

        :param int W: total number of columns (width of mask)
        :return torch.Tensor: unnormalized 1D vector representing pdf evaluated across mask columns.
        """
        return torch.ones(W, device=self.device)

    def sample_mask(self, mask: torch.Tensor) -> torch.Tensor:
        pdf = self.get_pdf(_W := mask.shape[-1])

        # lines are never randomly sampled from the already sampled center
        pdf[_W // 2 - self.n_center // 2 : _W // 2 + ceildiv(self.n_center, 2)] = 0

        # normalize distribution
        pdf = pdf / torch.sum(pdf)
        # select low-frequency lines according to pdf
        if self.n_lines > 0:
            for b in range(mask.shape[0]):
                for t in range(mask.shape[2]):
                    idx = random_choice(
                        _W, self.n_lines, replace=False, p=pdf, rng=self.rng
                    )
                    mask[b, :, t, :, idx] = 1

        # central lines are always sampled
        mask[
            :,
            :,
            :,
            :,
            _W // 2 - self.n_center // 2 : _W // 2 + ceildiv(self.n_center, 2),
        ] = 1

        return mask


class PolyOrderMaskGenerator(BaseMaskGenerator):
    r"""
    Generator for MRI Cartesian acceleration masks using polynomial variable density.

    Generates a mask of vertical lines for MRI acceleration with fixed sampling in low frequencies (center of k-space) and polynomial order sampling in the high frequencies.

    This is achieved by the following:

    1. Create 1D polynomial function :math:`(1 - r)^{p}` where :math:`r` is the distance from the center and :math:`p` is the polynomial order.
    2. Scale so that its mean matches desired acceleration factor
    3. Use the function as a probability density function (pdf) to sample from a Bernoulli.

    The mask is repeated across channels and randomly varies across batch dimension.

    Algorithm taken from :footcite:t:`millard2023theoretical`.

    :Examples:

        >>> from deepinv.physics.generator.mri import PolyOrderMaskGenerator
        >>> generator = PolyOrderMaskGenerator((2, 128, 128), acceleration=8, center_fraction=0.04, poly_order=8)
        >>> params = generator.step(batch_size=1)
        >>> mask = params["mask"]
        >>> mask.shape
        torch.Size([1, 2, 128, 128])

    For other parameter descriptions see :class:`deepinv.physics.generator.mri.BaseMaskGenerator`

    :param int poly_order: polynomial order of the sampling pdf
    """

    def __init__(self, *args, poly_order: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.poly_order = poly_order
        self.pdf = self.get_pdf()

    def get_pdf(self, max_iter: int = 100, tol: float = 1e-3) -> torch.Tensor:
        # Distance from the center (normalized to [-1, 1])
        r = torch.linspace(-1, 1, self.W, device=self.device).abs()

        # Polynomial variable density function
        pdf = (1 - r) ** self.poly_order

        # Sample the central lines
        pdf[
            self.W // 2 - self.n_center // 2 : self.W // 2 + ceildiv(self.n_center, 2)
        ] = 1

        # Using binary search to scale the mask to the desired acceleration factor
        a, b = -1.0, 1.0  # Binary search bounds
        target_fraction = 1.0 / self.acc

        for i in range(max_iter):
            c = (a + b) / 2  # Midpoint of search range
            scaled_pdf = pdf + c  # Adjust probabilities
            scaled_pdf.clamp_(0, 1)  # Keep values in range [0, 1]
            scaled_pdf[
                self.W // 2
                - self.n_center // 2 : self.W // 2
                + ceildiv(self.n_center, 2)
            ] = 1  # Ensure center is fully sampled

            # Calculate the current sampling fraction
            current_fraction = scaled_pdf.mean().item()

            # Adjust binary search bounds
            if current_fraction < target_fraction - tol:
                a = c
            elif current_fraction > target_fraction + tol:
                b = c
            else:
                return scaled_pdf
        else:
            raise ValueError(f"get_pdf did not converge after {max_iter} iterations")

    def sample_mask(self, mask: torch.Tensor) -> torch.Tensor:
        for b in range(mask.shape[0]):
            for t in range(mask.shape[2]):
                mask[b, :, t, :, :] = (
                    torch.bernoulli(self.pdf, generator=self.rng)
                    .unsqueeze(0)
                    .expand(self.H, self.W)
                )
        return mask


class GaussianMaskGenerator(RandomMaskGenerator):
    """Generator for MRI Cartesian acceleration masks using Gaussian undersampling.

    Generate a mask of vertical lines for MRI acceleration with fixed sampling in low frequencies (center of k-space) and Gaussian undersampling in the high frequencies.

    The high frequencies are selected according to a tail-adjusted Gaussian pdf. This ensures that the expected number of rows selected is equal to (N / acceleration).

    Supports k-t sampling, where the Gaussian mask varies randomly across time.

    The mask is repeated across channels and randomly varies across batch dimension.

    Algorithm taken from :footcite:t:`schlemper2017deep`.

    For parameter descriptions see :class:`deepinv.physics.generator.mri.BaseMaskGenerator`

    |sep|

    :Examples:

        Gaussian random k-t mask generator for a 8x64x64 video:

        >>> from deepinv.physics.generator import GaussianMaskGenerator
        >>> generator = GaussianMaskGenerator((2, 8, 64, 64), acceleration=8, center_fraction=0.04) # C, T, H, W
        >>> params = generator.step(batch_size=1)
        >>> mask = params["mask"]
        >>> mask.shape
        torch.Size([1, 2, 8, 64, 64])

    """

    def get_pdf(self, W: int) -> torch.Tensor:
        """Create one-dimensional Gaussian probability density function across columns, ignoring any fixed sampling columns.

        Add uniform distribution so that the probability of sampling high-frequency lines is non-zero.

        :param int W: total number of columns (width of sampling mask)
        :return torch.Tensor: unnormalized 1D vector representing pdf evaluated across mask columns.
        """
        x = torch.arange(W, device=self.device)
        pdf = torch.exp(-(0.5 / (W / 10.0) ** 2) * (x - W / 2) ** 2)
        return pdf + (W / (2.0 * self.acc) * 1.0 / W)


class EquispacedMaskGenerator(BaseMaskGenerator):
    """Generator for MRI Cartesian acceleration masks using uniform (equispaced) non-random undersampling with random offset.

    Generate a mask of vertical lines for MRI acceleration with fixed sampling in low frequencies (center of k-space) and equispaced undersampling in the high frequencies.

    The number of lines selected with equal spacing are at a proportion that reaches the desired acceleration rate taking into consideration the number of low-freq lines, so that the total number of lines is (N / acceleration).

    Supports k-t sampling, where the uniform mask is sheared across time.

    The mask is repeated across channels and the offset varies randomly across batch dimension. Based off fastMRI code https://github.com/facebookresearch/fastMRI

    For parameter descriptions see :class:`deepinv.physics.generator.mri.BaseMaskGenerator`

    |sep|

    :Examples:

        Equispaced k-t mask generator for a 8x64x64 video:

        >>> from deepinv.physics.generator import EquispacedMaskGenerator
        >>> generator = EquispacedMaskGenerator((2, 8, 64, 64), acceleration=8, center_fraction=0.04) # C, T, H, W
        >>> params = generator.step(batch_size=1)
        >>> mask = params["mask"]
        >>> mask.shape
        torch.Size([1, 2, 8, 64, 64])

    """

    def get_pdf(self) -> torch.Tensor:
        raise NotImplementedError("get_pdf is undefined for this mask generator.")

    def sample_mask(self, mask: torch.Tensor) -> torch.Tensor:
        _W = mask.shape[-1]
        pad = (_W - self.n_center + 1) // 2
        mask[:, :, :, :, pad : pad + self.n_center] = 1

        # determine acceleration rate by adjusting for the number of low frequencies
        adjusted_accel = (self.acc * (self.n_center - _W)) / (
            self.n_center * self.acc - _W
        )
        offset = torch.randint(
            low=0,
            high=round(adjusted_accel),
            size=(mask.size(0),),
            device=self.device,
            generator=self.rng,
        )

        for b in range(mask.shape[0]):
            for t in range(mask.shape[2]):
                accel_samples = (
                    torch.arange(
                        (t + offset[b]) % adjusted_accel,
                        _W - 1,
                        adjusted_accel,
                        device=self.device,
                    )
                    .round()
                    .type(torch.int32)
                )
                mask[b, :, t, :, accel_samples] = 1

        return mask
