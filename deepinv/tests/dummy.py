import torch
import numpy as np

from deepinv.datasets.base import ImageDataset


def create_circular_spherical_mask(imsize, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = tuple(int(s / 2) for s in imsize)
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(*center, *tuple(s - c for s, c in zip(imsize, center)))

    coords = np.ogrid[tuple(slice(0, s) for s in imsize)]
    dist_from_center = np.sqrt(
        sum((coords[i] - center[i]) ** 2 for i in range(len(imsize)))
    )
    mask = dist_from_center <= radius
    return mask


class DummyCircles(ImageDataset):
    def __init__(self, samples, imsize=(3, 32, 28), max_circles=10, seed=1):
        super().__init__()

        self.x = torch.zeros((samples,) + imsize, dtype=torch.float32)

        rng = np.random.default_rng(seed)

        max_rad = max(imsize[0], imsize[1]) / 2
        for i in range(samples):
            circles = rng.integers(low=1, high=max_circles)

            for c in range(circles):
                pos = rng.uniform(high=imsize[1:])
                color = rng.random((imsize[0], 1), dtype=np.float32)
                r = rng.uniform(high=max_rad)
                mask = torch.from_numpy(
                    create_circular_spherical_mask(imsize[1:], center=pos, radius=r)
                )
                self.x[i, :, mask] = torch.from_numpy(color)
        print(self.x.shape)

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.x.shape[0]


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x
