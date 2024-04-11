import numpy as np
import torch
from deepinv.physics.generator import PhysicsGenerator


class AccelerationMaskGenerator(PhysicsGenerator):
    r"""
    Generator for MRI cartesian acceleration masks.

    It generates a mask of vertical lines for MRI acceleration using fixed sampling in the low frequencies (center of k-space),
    and random uniform sampling in the high frequencies.

    :param tuple img_size: image size.
    :param int acceleration: acceleration factor.
    :param str device: cpu or gpu.

    |sep|

    :Examples:

    >>> from deepinv.physics.generator import AccelerationMaskGenerator
    >>> import deepinv
    >>> mask_generator = AccelerationMaskGenerator((16, 16))
    >>> mask_dict = mask_generator.step() # dict_keys(['mask'])
    >>> deepinv.utils.plot(mask_dict['mask'].squeeze(1))
    >>> print(mask_dict['mask'].shape)
    torch.Size([1, 2, 16, 16])

    """

    def __init__(self, img_size: tuple, acceleration=4, device: str = "cpu"):
        super().__init__(shape=img_size, device=device)
        self.device = device
        self.img_size = img_size
        self.acceleration = acceleration

    def step(self, batch_size=1):
        r"""
        Create a mask of vertical lines.

        :param int batch_size: batch_size.
        :return: dictionary with key **'mask'**: tensor of size (batch_size, 1, H, W) with values in {0, 1}.
        :rtype: dict
        """
        img_size = self.img_size
        acceleration_factor = self.acceleration

        if acceleration_factor == 4:
            central_lines_percent = 0.08
            num_lines_center = int(central_lines_percent * img_size[-1])
            side_lines_percent = 0.25 - central_lines_percent
            num_lines_side = int(side_lines_percent * img_size[-1])
        if acceleration_factor == 8:
            central_lines_percent = 0.04
            num_lines_center = int(central_lines_percent * img_size[-1])
            side_lines_percent = 0.125 - central_lines_percent
            num_lines_side = int(side_lines_percent * img_size[-1])
        mask = torch.zeros((batch_size,) + img_size, **self.factory_kwargs)
        center_line_indices = torch.linspace(
            img_size[0] // 2 - num_lines_center // 2,
            img_size[0] // 2 + num_lines_center // 2 + 1,
            steps=50,
            dtype=torch.long,
        )
        mask[:, :, center_line_indices] = 1

        for i in range(batch_size):
            random_line_indices = np.random.choice(
                img_size[0], size=(num_lines_side // 2,), replace=False
            )
            mask[i, :, random_line_indices] = 1

        return {"mask": torch.cat([mask.float().unsqueeze(1)] * 2, dim=1)}
