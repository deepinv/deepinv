import sys
import types

import torch


class DummyPhysics:
    """Very small 'physics' object with call and A_adjoint."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Forward operator H: multiply by 2
        return 2.0 * x

    def A_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        # Adjoint operator H^T: divide by 2
        return 0.5 * y


def _install_dummy_deal(tmp_path):
    """
    Install a minimal 'deal' module in sys.modules so that
    DEALReconstructor can import `deal.DEAL` without requiring
    the real repository.
    """

    class DummyDEAL(torch.nn.Module):
        def __init__(self, color: bool = False):
            super().__init__()
            # one parameter just so state_dict is not empty
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.max_iter = 0

        def solve_inverse_problem(
            self,
            y,
            H,
            Ht,
            sigma,
            lmbda,
            eps_in,
            eps_out,
            path,
            x_init,
            verbose,
        ):
            # Stand-in behaviour: just return H^T(y)
            return Ht(y)

    # Put a fake 'deal' module in sys.modules
    dummy_module = types.SimpleNamespace(DEAL=DummyDEAL)
    sys.modules["deal"] = dummy_module

    # Create a small checkpoint compatible with DummyDEAL
    model = DummyDEAL()
    ckpt_path = tmp_path / "dummy_deal.pth"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)

    return ckpt_path


def test_deal_reconstructor_and_helper(tmp_path):
    # Prepare fake deal module and checkpoint *before* importing the wrapper
    ckpt_path = _install_dummy_deal(tmp_path)

    from deepinv.reconstructors import DEALReconstructor, deal_reconstruct

    physics = DummyPhysics()
    y = torch.ones(1, 1, 4, 4)  # dummy "sinogram"

    # Test the class interface
    reconstructor = DEALReconstructor(
        checkpoint_path=str(ckpt_path),
        sigma=25.0,
        lam=10.0,
        max_iters=3,
        device="cpu",
        auto_scale=False,
        clamp_output=True,
    )

    x_hat = reconstructor.reconstruct(y, physics)

    assert x_hat.shape == y.shape
    # For our DummyPhysics: H(x)=2x, H^T(y)=0.5y
    assert torch.allclose(x_hat, 0.5 * y, atol=1e-6)

    # Test the convenience function
    x_hat2 = deal_reconstruct(
        y,
        physics,
        checkpoint_path=str(ckpt_path),
        sigma=25.0,
        lam=10.0,
        max_iters=3,
        device="cpu",
        auto_scale=False,
        clamp_output=True,
    )

    assert torch.allclose(x_hat2, x_hat, atol=1e-6)
