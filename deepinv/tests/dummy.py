import torch
from torch.utils.data import Dataset
import numpy as np


def create_circular_mask(imsize, center=None, radius=None):
    h, w = imsize
    if center is None:  # use the middle of the image
        center = (int(h / 2), int(w / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], h - center[0], w - center[1])

    X, Y = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask


class DummyCircles(Dataset):
    def __init__(self, samples, imsize=(3, 32, 28), max_circles=10, seed=1):
        super().__init__()

        self.x = torch.zeros((samples,) + imsize, dtype=torch.float32)

        rng = np.random.default_rng(seed)

        max_rad = max(imsize[0], imsize[1]) / 2
        for i in range(samples):
            circles = rng.integers(low=1, high=max_circles)

            for c in range(circles):
                pos = rng.uniform(high=imsize[1:])
                colour = rng.random((imsize[0], 1), dtype=np.float32)
                r = rng.uniform(high=max_rad)
                mask = torch.from_numpy(
                    create_circular_mask(imsize[1:], center=pos, radius=r)
                )
                self.x[i, :, mask] = torch.from_numpy(colour)

    def __getitem__(self, index):
        return self.x[index, :, :, :]

    def __len__(self):
        return self.x.shape[0]


if __name__ == "__main__":
    device = "cuda:0"
    imsize = (3, 23, 100)
    dataset = DummyCircles(10, imsize=imsize)

    x = dataset[0]

    import matplotlib.pyplot as plt
    from deepinv.utils.plotting import config_matplotlib

    config_matplotlib()

    plt.imshow(x.permute(1, 2, 0).cpu().numpy())
    plt.show()


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x
