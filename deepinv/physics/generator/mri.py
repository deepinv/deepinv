from abc import ABC, abstractmethod
from typing import Optional, Tuple
import warnings

import torch

from deepinv.physics.generator import PhysicsGenerator
from deepinv.physics.functional import random_choice


def ceildiv(a: float, b: float) -> float:
    return -(a // -b)


class BaseMaskGenerator(PhysicsGenerator, ABC):
    """Base generator for MRI acceleration masks.

    Generate a mask of vertical lines for MRI acceleration with fixed sampling in low frequencies (center of k-space) and undersampling in the high frequencies.

    The type of undersampling is determined by the child class. The mask is repeated across channels and randomly varies across batch dimension.

    :param Tuple img_size: image size, either (H, W) or (C, H, W) or (C, T, H, W), where optional C is channels, and optional T is number of time-steps
    :param int acceleration: acceleration factor, defaults to 4
    :param float center_fraction: fraction of lines to sample in low frequencies (center of k-space). If 0, there is no fixed low-freq sampling. Defaults to None.
    :param torch.Generator rng: torch random generator. Defaults to None.
    """

    def __init__(
        self,
        img_size: Tuple,
        acceleration: int = 4,
        center_fraction: Optional[float] = None,
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

        self.n_center = int(self.center_fraction * self.W)
        self.n_lines = int(self.W // self.acc - self.n_center)

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
        """Get mask PDF.

        :return torch.Tensor: unnormalised 1D vector representing pdf evaluated across mask columns.
        """
        pass

    def step(
        self, batch_size=1, seed: int = None, img_size: Optional[Tuple] = None, **kwargs
    ) -> dict:
        r"""
        Create a mask of vertical lines.

        :param int batch_size: batch_size.
        :param int seed: optional: the seed for the random number generator, to reseed on-the-fly.
        :param tuple img_size: optionally reset the 2D image size on-the-fly, must be of form (H, W).

        :return: dictionary with key **'mask'**: tensor of size (batch_size, C, H, W) or (batch_size, C, T, H, W) with values in {0, 1}.
        :rtype: dict
        """
        self.rng_manual_seed(seed)

        _T = self.T if self.T > 0 else 1
        _H, _W = (self.H, self.W) if img_size is None else img_size

        mask = self.sample_mask(
            torch.zeros((batch_size, self.C, _T, _H, _W), **self.factory_kwargs)
        )

        if self.T == 0:
            mask = mask[:, :, 0, :, :]

        return {"mask": mask}


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

    def get_pdf(self) -> torch.Tensor:
        """Create one-dimensional uniform probability density function across columns, ignoring any fixed sampling columns.

        :return torch.Tensor: unnormalised 1D vector representing pdf evaluated across mask columns.
        """
        return torch.ones(self.W, device=self.device)

    def sample_mask(self, mask: torch.Tensor) -> torch.Tensor:
        pdf = self.get_pdf()

        # lines are never randomly sampled from the already sampled center
        pdf[
            self.W // 2 - self.n_center // 2 : self.W // 2 + ceildiv(self.n_center, 2)
        ] = 0

        # normalise distribution
        pdf = pdf / torch.sum(pdf)
        # select low-frequency lines according to pdf
        for b in range(mask.shape[0]):
            for t in range(mask.shape[2]):
                idx = random_choice(
                    self.W, self.n_lines, replace=False, p=pdf, rng=self.rng
                )
                mask[b, :, t, :, idx] = 1

        # central lines are always sampled
        mask[
            :,
            :,
            :,
            :,
            self.W // 2 - self.n_center // 2 : self.W // 2 + ceildiv(self.n_center, 2),
        ] = 1

        return mask


class PolyOrderMaskGenerator(BaseMaskGenerator):
    """
    Generator for MRI Cartesian acceleration masks using polynomial variable density.

    Generates a mask of vertical lines for MRI acceleration with fixed sampling in low frequencies (center of k-space) and polynomial order sampling in the high frequencies.
    Polynomial sampling varies as r^poly_order where r is the distance from the centre.

    The mask is repeated across channels and randomly varies across batch dimension.

    Algorithmn taken from Millard and Chiew `A Theoretical Framework for Self-Supervised MR Image Reconstruction Using Sub-Sampling via Variable Density Noisier2Noise`

    :Examples:
        Polynomial random sampling for 2x128x128 images

        >>> from deepinv.physics.generator.mri import PolyOrderMaskGenerator
        >>> generator = PolyOrderMaskGenerator((2, 128, 128), acceleration=8, center_fraction=0.04, poly_order=8)
        >>> params = generator.step(batch_size=1)
        >>> mask = params["mask"]
        >>> mask.shape
        torch.Size([1, 2, 128, 128])

    :param int poly_order: polynomial order of the sampling pdf (should be the same poly_order used for input_mask)
    :

    """

    def __init__(
        self,
        img_size: Tuple,
        acceleration: int = 4,
        center_fraction: Optional[float] = None,
        poly_order: int = 8,
        rng: torch.Generator = None,
        device="cpu",
        *args,
        **kwargs,
    ):
        super().__init__(
            img_size=img_size,
            acceleration=acceleration,
            center_fraction=center_fraction,
            rng=rng,
            device=device,
            *args,
            **kwargs,
        )
        self.poly_order = poly_order
        # generate a pdf that is cached
        self.pdf = self.get_pdf()

    def get_pdf(self) -> torch.Tensor:
        """
        Create a 1D polynomial variable probability density function across k-space columns that varies as (1-r)**poly_order
        Central lines are always sampled.
        : return torch.Tensor: unnormalised 1D vector representing pdf evaluated across mask columns.
        """
        # Distance from the center (normalized to [-1, 1])
        r = torch.linspace(-1, 1, self.W, device=self.device).abs()

        # Polynomial variable density function
        pdf = (1 - r) ** self.poly_order

        # Sample the central lines
        pdf[
            self.W // 2 - self.n_center // 2 : self.W // 2 + ceildiv(self.n_center, 2)
        ] = 1

        # Using binary search to scale the prob_mask to the desired acceleration factor
        eta = 1e-3  # Tolerance
        a, b = -1.0, 1.0  # Binary search bounds
        iterations = 0
        target_fraction = 1 / self.acc

        while iterations < 100:
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
            if current_fraction < target_fraction - eta:
                a = c
            elif current_fraction > target_fraction + eta:
                b = c
            else:
                break  # Exit loop when within tolerance

            iterations += 1

        if iterations == 100:
            warnings.warn("gen_pdf_columns did not converge after 100 iterations")

        return scaled_pdf

    def sample_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Samples from Bernoulli distribution from self.pdf to create a 1D mask
        :return torch.Tensor: 2D mask
        """
        for b in range(mask.shape[0]):  # Iterate over batch
            for t in range(mask.shape[2]):  # Iterate over time steps
                # Sample from Bernoulli and reshape to match (H, W)
                mask_1d = torch.bernoulli(self.pdf)  # Shape: [W]
                # assumes pixelwise which is not good
                mask[b, :, t, :, :] = mask_1d.unsqueeze(0).expand(
                    self.H, self.W
                )  # Match (H, W)
        return mask


class GaussianMaskGenerator(RandomMaskGenerator):
    """Generator for MRI Cartesian acceleration masks using Gaussian undersampling.

    Generate a mask of vertical lines for MRI acceleration with fixed sampling in low frequencies (center of k-space) and Gaussian undersampling in the high frequencies.

    The high frequences are selected according to a tail-adjusted Gaussian pdf. This ensures that the expected number of rows selected is equal to (N / acceleration).

    Supports k-t sampling, where the Gaussian mask varies randomly across time.

    The mask is repeated across channels and randomly varies across batch dimension.

    Algorithm taken from Schlemper et al. `A Deep Cascade of Convolutional Neural Networks for Dynamic MR Image Reconstruction <https://github.com/js3611/Deep-MRI-Reconstruction/blob/master/utils/compressed_sensing.py>`_.

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

    def get_pdf(self) -> torch.Tensor:
        """Create one-dimensional Gaussian probability density function across columns, ignoring any fixed sampling columns.

        Add uniform distribution so that the probability of sampling high-frequency lines is non-zero.

        :return torch.Tensor: unnormalised 1D vector representing pdf evaluated across mask columns.
        """

        def normal_pdf(length, sensitivity):
            return torch.exp(
                -sensitivity
                * (torch.arange(length, device=self.device) - length / 2) ** 2
            )

        pdf = normal_pdf(self.W, 0.5 / (self.W / 10.0) ** 2)
        lmda = self.W / (2.0 * self.acc)

        pdf += lmda * 1.0 / self.W

        return pdf


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
        pad = (self.W - self.n_center + 1) // 2
        mask[:, :, :, :, pad : pad + self.n_center] = 1

        # determine acceleration rate by adjusting for the number of low frequencies
        adjusted_accel = (self.acc * (self.n_center - self.W)) / (
            self.n_center * self.acc - self.W
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
                        self.W - 1,
                        adjusted_accel,
                        device=self.device,
                    )
                    .round()
                    .type(torch.int32)
                )
                mask[b, :, t, :, accel_samples] = 1

        return mask
