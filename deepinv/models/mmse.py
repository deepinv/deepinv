import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Denoiser


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
        B = x.shape[0]
        atoms = self.atoms
        sigma2 = self.sigma ** 2

        # Squared distance: ||x - a_k||²
        dist = torch.norm(
            x.view(B, 1, -1) - atoms.view(1, atoms.shape[0], -1), dim=2
        ) # (B, N)
    
        # Posterior weights
        alpha = F.softmax(-dist2 / sigma**2, dim=1) 

        # MMSE estimate
        denoised_flat = weights @ atoms_flat     # (B, D)
        return denoised_flat.view_as(x)


if __name__ == "__main__":
    devie = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Example usage
    dataset = torch.randn(100, 1, 28, 28).to(device)  # Example dataset of 100 images
    denoiser = MMSE(dataset).to(device)

    noisy_images = torch.randn(10, 1, 28, 28).to(device)  # Batch of 10 noisy images
    sigma = 0.1

    denoised_images = denoiser(noisy_images, sigma)
    print(denoised_images.shape)  # Should be (10, 1, 28, 28)
