from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepinv.models import Denoiser


class MMSE(Denoiser):
    r"""
    Closed-form MMSE denoiser for a Dirac-mixture prior based on a given dataset of images :math:`a_k`.

    .. math::

        p(x) = (1/N) \sum_{k=1}^N \delta(x - a_k)

    Given a noisy observation :math:`y = x + \sigma n` with :math:`n \sim \mathcal{N}(0, I)`, the MMSE estimate is given by:

    .. math::

        \mathbb{E}[x | y] = \sum_{k=1}^N \alpha_k(y) a_k \quad \text{with} \quad \alpha_k(y) = softmax\left(- \frac{\| y - a_k\|^2}{\sigma^2 \right).
    """

    def __init__(self, dataset: torch.Tensor):
        super().__init__()
        # Store dataset atoms as buffer so they follow .to(device)
        self.register_buffer("atoms", dataset.clone().detach())

    def forward(self, x: torch.Tensor, sigma: torch.tensor | float) -> torch.Tensor:
        if isinstance(sigma, torch.Tensor):
            assert len(sigma) == x.shape[0]
            sigma = sigma[:, None]
        # distance: ||x - a_k||
        dist = torch.norm(
            x.view(x.shape[0], 1, -1) - self.atoms.view(1, self.atoms.shape[0], -1), dim=2
        ) # (B, N)
        # Posterior weights
        alpha = F.softmax(-dist**2 / sigma**2, dim=1) 
        # MMSE estimate
        denoised_flat = alpha @ self.atoms.view(self.atoms.shape[0], -1)
        return denoised_flat.view_as(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Example usage
    from torchvision import datasets
    import torchvision.transforms as T
    dataset = datasets.MNIST(root='/lustre/fswork/projects/rech/kkn/uuw19zo/datasets', train=False, download=True)
    dataset = torch.stack([T.ToTensor()(img) for img, _ in dataset])
    #dataset = torch.randn(100, 1, 28, 28).to(device)  # Example dataset of 100 images
    print(dataset.shape)
    denoiser = MMSE(dataset).to(device)

    noisy_images = torch.randn(10, 1, 28, 28).to(device)  # Batch of 10 noisy images
    sigma = 0.1 * torch.ones(10).to(device)

    denoised_images = denoiser(noisy_images, sigma)
    from deepinv.utils import plot
    plot(denoised_images[0], save_dir="test.png")

