# tests/test_diffusion_discrete.py
from __future__ import annotations
import pytest
import sys

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="Stable Diffusion (diffusers/transformers) requires Python >= 3.10",
)
pytest.importorskip(
    "transformers", reason="Install the 'latent' extra to run this test."
)
pytest.importorskip("diffusers", reason="Install the 'latent' extra to run this test.")

import torch
from torch import nn, Tensor


# Import the samplers under test
from deepinv.sampling import DDIMDiffusion, PSLDDiffusionPosterior

import deepinv as dinv


# ----------------------------------------------------------------------------- #
# A tiny stand-in for LatentDiffusion with the same public interface:
# - forward(x, t, prompt): predicts noise ε(x_t, t)  (here: simple linear op)
# - encode(x): image -> "latent"  (identity + clamp to [-1,1])
# - decode(z): "latent" -> image  (identity + clamp to [-1,1])
#
# This keeps tests fast and avoids external downloads while checking algorithmic
# plumbing (shapes, determinism, gradients, basic data-consistency improvement).
# ----------------------------------------------------------------------------- #
class _TinyLatentDiffusion(nn.Module):
    def __init__(
        self,
        scale: float = 0.0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.conv = nn.Conv2d(3, 3, kernel_size=1, bias=False).to(
            self.device, dtype=self.dtype
        )
        with torch.no_grad():
            self.conv.weight.fill_(scale)

    def forward(self, x: Tensor, t: Tensor, prompt: str | None = None) -> Tensor:
        return self.conv(x.to(self.device, self.dtype))

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        return x.to(self.device, self.dtype).clamp_(-1, 1)

    @torch.no_grad()
    def decode(self, z: Tensor) -> Tensor:
        return z.to(self.device, self.dtype).clamp_(-1, 1)


# ----------------------------- Fixtures ------------------------------------- #
@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def imshape():
    # Use a tiny spatial size to keep tests snappy
    return (1, 3, 32, 32)


# --------------------------- DDIM Tests ------------------------------------- #
def test_ddim_forward_shape_and_determinism(device, imshape):
    """
    DDIM should:
      - return a tensor with the same shape as the input latent
      - be deterministic (no randomness in the implementation)
    """
    torch.manual_seed(0)

    model = _TinyLatentDiffusion(scale=0.05, device=device, dtype=torch.float16)  # ε≈0
    ddim = DDIMDiffusion(
        model=model,
        beta_min=0.00085,
        beta_max=0.012,
        num_train_timesteps=20,
        num_inference_steps=10,
        prompt="",
        dtype=torch.float32,
        device=torch.device(device),
    )

    zT = torch.randn(imshape, device=device, dtype=torch.float32)
    z0_a = ddim.forward(zT.clone())
    z0_b = ddim.forward(zT.clone())

    assert z0_a.shape == zT.shape
    assert torch.allclose(z0_a, z0_b, atol=0.0, rtol=0.0)


# --------------------------- PSLD Tests ------------------------------------- #
def test_psld_reduces_data_residual(device, imshape):
    """
    PSLD should move the decoded image closer to the measurement y in the
    data-consistency sense (||A(x_hat) - y||). We compare against the
    *initial* sample's residual as a baseline.
    """
    torch.manual_seed(0)

    # Tiny model; ε is tiny but not zero
    model = _TinyLatentDiffusion(scale=0.05, device=device)

    psld = PSLDDiffusionPosterior(
        model=model,
        beta_min=0.00085,
        beta_max=0.012,
        num_train_timesteps=40,
        num_inference_steps=20,
        dtype=torch.float32,
        device=torch.device(device),
    )

    # Ground-truth image in [-1,1] so that decode/physics are consistent
    x_gt = torch.tanh(torch.randn(imshape, device=device, dtype=torch.float32))
    physics = dinv.physics.Inpainting(
        img_size=x_gt.shape[1:],  # (C,H,W)
        mask=0.5,  # random 50% mask for speed/coverage
        device=device,
        noise_model=dinv.physics.GaussianNoise(sigma=0.0),
    )
    y = physics(x_gt)  # y in [-1,1] (physics here is linear masking)

    # Initial latent z_T
    zT = torch.randn(imshape, device=device, dtype=torch.float32)

    # Baseline residual from the initial decode (no sampling)
    x_init = model.decode(zT.clone())
    baseline_res = torch.linalg.norm(
        (physics.A(x_init) - y).reshape(x_init.size(0), -1), dim=1
    ).mean()

    # PSLD run with a conservative step to avoid overshoot in the toy setup
    z0 = psld.forward(
        sample=zT.clone(),
        y=y,
        forward_model=physics,
        dps_eta=0.1,  # small step
        gamma=0.5,
        omega=1.0,
    )
    x_hat = model.decode(z0)

    # New residual
    new_res = torch.linalg.norm(
        (physics.A(x_hat) - y).reshape(x_hat.size(0), -1), dim=1
    ).mean()

    # PSLD should not increase residual; usually it decreases it.
    assert new_res <= baseline_res + 1e-6


# ------------------------ SD latent model test -------------------------- #
@pytest.mark.slow
def test_ddim_with_real_latentdiffusion(device):
    """
    Optional test with the real LatentDiffusion (loads SD weights).
    Disabled by default to keep CI fast/lightweight.
    """
    from deepinv.models import LatentDiffusion  # uses diffusers/CLIP under the hood

    torch.manual_seed(0)
    model = LatentDiffusion(device=device)
    ddim = DDIMDiffusion(
        model=model,
        beta_min=0.00085,
        beta_max=0.012,
        num_train_timesteps=20,
        num_inference_steps=10,
        prompt="",  # unconditional
        dtype=torch.float16,
        device=torch.device(device),
    )
    zT = torch.randn((1, 4, 32, 32), device=device, dtype=torch.float16)
    z0 = ddim.forward(zT)
    assert z0.shape == zT.shape
