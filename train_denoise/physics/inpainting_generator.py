import torch
from deepinv.physics.generator import PhysicsGenerator

class InpaintingMaskGenerator(PhysicsGenerator):
    
    def __init__(
        self,
        mask_shape: tuple,
        num_channels: int = 1,
        device: str = "cpu",
        dtype: type = torch.float32,
        block_size_ratio=0.1,
        num_blocks=5,
    ) -> None:
        kwargs = {"mask_shape": mask_shape, "block_size_ratio": block_size_ratio, "num_blocks": num_blocks}
        if len(mask_shape) != 2:
            raise ValueError(
                "mask_shape must 2D. Add channels via num_channels parameter"
            )
        super().__init__(
            num_channels=num_channels,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    def generate_mask(self, image_shape, block_size_ratio, num_blocks):
        # Create an all-ones tensor which will serve as the initial mask
        mask = torch.ones(image_shape)
        batch_size = mask.shape[0]
        
        # Calculate block size based on the image dimensions and block_size_ratio
        block_width = int(image_shape[-2] * block_size_ratio)
        block_height = int(image_shape[-1] * block_size_ratio)
        
        # Generate random coordinates for each block in each batch
        x_coords = torch.randint(0, image_shape[-1] - block_width, (batch_size, num_blocks))
        y_coords = torch.randint(0, image_shape[-2] - block_height, (batch_size, num_blocks))
        
        # Create grids of indices for the block dimensions
        x_range = torch.arange(block_width).view(1, 1, -1)
        y_range = torch.arange(block_height).view(1, 1, -1)
        
        # Expand ranges to match the batch and num_blocks dimensions
        x_indices = x_coords.unsqueeze(-1) + x_range
        y_indices = y_coords.unsqueeze(-1) + y_range
        
        # Expand and flatten the indices for advanced indexing
        x_indices = x_indices.unsqueeze(2).expand(-1, -1, block_height, -1).reshape(-1)
        y_indices = y_indices.unsqueeze(3).expand(-1, -1, -1, block_width).reshape(-1)
        
        # Create batch indices for advanced indexing
        batch_indices = torch.arange(batch_size).view(-1, 1, 1).expand(-1, num_blocks, block_width * block_height).reshape(-1)
        channel_indices = torch.arange(3).view(1, 1, 1, -1).expand(batch_size, num_blocks, block_width * block_height, -1).reshape(-1)
        
        # Apply the blocks using advanced indexing
        mask[batch_indices, :, y_indices, x_indices] = 0

        return mask

    def step(self, batch_size: int = 1, block_size_ratio: float = None, num_blocks = None):
        r"""
        Generate a random motion blur PSF with parameters :math:`\sigma` and :math:`l`

        :param int batch_size: batch_size.
        :param float sigma: the standard deviation of the Gaussian Process
        :param float l: the length scale of the trajectory

        :return: dictionary with key **'filter'**: the generated PSF of shape `(batch_size, 1, psf_size[0], psf_size[1])`
        """

        # TODO: add randomness
        block_size_ratio = self.block_size_ratio if block_size_ratio is None else block_size_ratio
        num_blocks = self.num_blocks if num_blocks is None else num_blocks
        batch_shape = (batch_size, self.num_channels, self.mask_shape[-2], self.mask_shape[-1])
        
        mask = self.generate_mask(batch_shape, block_size_ratio, num_blocks)
        

        return {
            "mask": mask.to(self.factory_kwargs["device"])
        }

