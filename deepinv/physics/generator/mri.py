import torch
from typing import Sequence, Optional, Union, Tuple
import numpy as np
from deepinv.physics.generator import PhysicsGenerator

#TODO inline examples
#TODO find common code across step() methods and put in Base

class BaseMaskGenerator(PhysicsGenerator):
    """Base generator for MRI acceleration masks.

    Generate a mask of vertical lines for MRI acceleration with fixed sampling in low frequencies (center of k-space) and undersampling in the high frequencies.

    The type of undersampling is determined by the child class. The mask is repeated across channels and randomly varies across batch dimension.

    :param Tuple img_size: image size, either (H, W) or (C, H, W) or (C, T, H, W), where optional C is channels, and optional T is number of time-steps
    :param int acceleration: acceleration factor, defaults to 4
    :param float center_fraction: fraction of lines to sample in low frequencies (center of k-space), defaults to 0.125
    """
    def __init__(self, img_size: Tuple, acceleration: int = 4, center_fraction: Optional[float] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_size = img_size
        self.acc = acceleration

        if center_fraction is not None:
            self.center_fraction = center_fraction
        elif acceleration == 4:
            self.center_fraction = 0.08
        elif acceleration == 8:
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
        self.n_lines = int(self.W / self.acc) - self.n_center
    
    def step(self, batch_size=1) -> dict:
        r"""
        Create a mask of vertical lines.

        :param int batch_size: batch_size.
        :return: dictionary with key **'mask'**: tensor of size (batch_size, C, H, W) or (batch_size, C, T, H, W) with values in {0, 1}.
        :rtype: dict
        """   
        raise NotImplementedError()

class RandomMaskGenerator(BaseMaskGenerator):
    """Generator for MRI Cartesian acceleration masks using random uniform undersampling.

    Generate a mask of vertical lines for MRI acceleration with fixed sampling in low frequencies (center of k-space) and random uniform undersampling in the high frequencies.

    Supports k-t sampling, where the mask is selected randomly across time. 
    
    The mask is repeated across channels and randomly varies across batch dimension.

    For parameter descriptions see :class:`deepinv.physics.generator.mri.BaseMaskGenerator`
    """
    def step(self, batch_size: int = 1) -> dict:
        _T = self.T if self.T > 0 else 1
        mask = torch.zeros((batch_size, self.C, _T, self.H, self.W), **self.factory_kwargs) #B, C, T, H, W

        # central lines are always sampled
        center_line_indices = torch.linspace(
            self.W // 2 - self.n_center // 2,
            self.W // 2 + self.n_center // 2 + 1,
            steps=50,
            dtype=torch.long,
        )
        mask[:, :, :, :, center_line_indices] = 1

        # select low-frequency lines according to pdf
        for b in range(batch_size):
            for t in range(_T):
                idx = np.random.choice(self.W, size=(self.n_lines // 2,), replace=False)
                mask[b, :, t, :, idx] = 1
        
        if self.T == 0:
            mask = mask[:, :, 0, :, :]

        return {"mask": mask}   

class GaussianMaskGenerator(BaseMaskGenerator):
    """Generator for MRI Cartesian acceleration masks using Gaussian undersampling.

    Generate a mask of vertical lines for MRI acceleration with fixed sampling in low frequencies (center of k-space) and Gaussian undersampling in the high frequencies.

    The high frequences are selected according to a tail-adjusted Gaussian pdf. This ensures that the expected number of rows selected is equal to (N / acceleration).

    Supports k-t sampling, where the Gaussian mask varies randomly across time. 
    
    The mask is repeated across channels and randomly varies across batch dimension.

    Algorithm taken from Schlemper et al. `A Deep Cascade of Convolutional Neural Networks for Dynamic MR Image Reconstruction <https://github.com/js3611/Deep-MRI-Reconstruction/blob/master/utils/compressed_sensing.py>`_.

    For parameter descriptions see :class:`deepinv.physics.generator.mri.BaseMaskGenerator`
    """
    def step(self, batch_size: int = 1) -> dict:
        _T = self.T if self.T > 0 else 1
        mask = torch.zeros((batch_size, self.C, _T, self.H, self.W), **self.factory_kwargs) #B, C, T, H, W

        # generate normal distribution
        normal_pdf = lambda length, sensitivity: np.exp(-sensitivity * (np.arange(length) - length / 2)**2)
        pdf_x = normal_pdf(self.W, 0.5/(self.W/10.)**2)
        lmda = self.W / (2. * self.acc)
    
        # add uniform distribution so that probability of sampling
        # high-frequency lines is non-zero
        pdf_x += lmda * 1./self.W
    
        # lines are never randomly sampled from the already sampled center
        pdf_x[self.W//2 - self.n_center//2 : self.W//2 + self.n_center//2] = 0
        pdf_x /= np.sum(pdf_x)  # normalise distribution

        # select low-frequency lines according to pdf
        for b in range(batch_size):
            for t in range(_T):
                idx = np.random.choice(self.W, self.n_lines, replace=False, p=pdf_x)
                mask[b, :, t, :, idx] = 1
    
        # central lines are always sampled
        mask[:, :, :, :, self.W//2-self.n_center//2:self.W//2+self.n_center//2] = 1

        if self.T == 0:
            mask = mask[:, :, 0, :, :]

        return {"mask": mask}
    
class UniformMaskGenerator(BaseMaskGenerator):
    pass