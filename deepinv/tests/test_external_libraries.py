import deepinv as dinv
import torch
import pytest

import unittest.mock as mock


class TestTomographyWithAstra:
    def dummy_compute_norm(cls, x0: torch.Tensor) -> torch.Tensor:
        return torch.tensor(1.0).to(x0)

    def dummy_projection(cls, x: torch.Tensor, out: torch.Tensor) -> None:
        out[:] = 1.0

    @pytest.mark.parametrize(
        "is_2d,geometry_type,normalize",
        [
            (True, "parallel", False),
            (True, "parallel", True),
            (True, "fanbeam", False),
            (True, "fanbeam", True),
            (False, "parallel", False),
            (False, "parallel", True),
            (False, "conebeam", False),
            (False, "conebeam", True),
        ],
    )
    def test_tomography_with_astra_logic(self, is_2d, geometry_type, normalize):
        r"""
        Tests tomography operator with astra backend which does not have a numerically precise adjoint.

        :param bool is_2d: Runs the test with 2D geometry, else 3D.
        :param str geometry_type: In 2D, expects ``parallel`` or ``fanbeam``. In 3D expects ``parallel`` or ``conebeam``.
        :param bool normalize: Initializes the operator with ``normalize=normalize``.
        """

        pytest.importorskip(
            "astra",
            reason="This test requires astra-toolbox. It should be "
            "installed with `conda install -c astra-toolbox -c nvidia astra-toolbox`",
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        ## Test 2d transforms
        if is_2d:
            img_size = (16, 16)
            num_detectors = 2 * img_size[0]
            num_angles = 2 * img_size[0]
            physics = dinv.physics.TomographyWithAstra(
                img_size=img_size,
                num_detectors=num_detectors,
                num_angles=num_angles,
                angular_range=(
                    (0, torch.pi) if geometry_type == "parallel" else (0, 2 * torch.pi)
                ),
                geometry_type=geometry_type,
                normalize=normalize,
                device=device,
            )

        else:
            ## Test 3d transforms
            img_size = (16, 16, 16)
            num_detectors = (32, 32)
            num_angles = 2 * img_size[0]
            physics = dinv.physics.TomographyWithAstra(
                img_size=img_size,
                num_angles=num_angles,
                angular_range=(
                    (0, torch.pi) if geometry_type == "parallel" else (0, 2 * torch.pi)
                ),
                num_detectors=num_detectors,
                geometry_type=geometry_type,
                detector_spacing=(1.0, 1.0),
                object_spacing=(1.0, 1.0, 1.0),
                normalize=normalize,
                device=device,
            )

        x = torch.rand(1, 1, *img_size, device=device)

        if device != "cuda":
            with (
                mock.patch.object(
                    dinv.physics.functional.XrayTransform,
                    "_forward_projection",
                    new=self.dummy_projection,
                ),
                mock.patch.object(
                    dinv.physics.functional.XrayTransform,
                    "_backprojection",
                    new=self.dummy_projection,
                ),
                mock.patch.object(
                    dinv.physics.TomographyWithAstra,
                    "compute_norm",
                    new=self.dummy_compute_norm,
                ),
            ):
                ## -------- Test forward --------
                Ax = physics.A(x)
                assert Ax.shape == (1, 1, *physics.measurement_shape)

                ## ------- Test backward --------
                y = torch.rand_like(Ax)
                At_y = physics.A_adjoint(y)
                assert At_y.shape == (1, 1, *img_size)

                ## ---- Test pseudo-inverse -----
                x_hat = physics.A_dagger(y)
                assert x_hat.shape == (1, 1, *img_size)

                ## --- Test autograd.Function ---
                pred = torch.zeros_like(x, requires_grad=True)
                loss = torch.linalg.norm(physics.A(pred) - Ax)
                loss.backward()
                assert pred.grad is not None

        else:
            ## --- Test adjointness ---
            Ax = physics.A(x)
            y = torch.rand_like(Ax)
            At_y = physics.A_adjoint(y)

            Ax_y = torch.sum(Ax * y).item()
            At_y_x = torch.sum(At_y * x).item()

            relative_error = abs(Ax_y - At_y_x) / At_y_x
            assert relative_error < 0.01  # at least 99% adjoint

            ## --- Test pseudoinverse ---
            r_tol = 0.05 if geometry_type == "parallel" else 0.1
            r = physics.A_adjoint(physics.A(x))
            y = physics.A(r)
            error = torch.linalg.norm(physics.A_dagger(y) - r) / torch.linalg.norm(r)
            assert error < r_tol

            ## --- Test autograd.Function ---
            pred = torch.zeros_like(x, requires_grad=True)
            loss = torch.linalg.norm(physics.A(pred) - Ax)
            loss.backward()
            assert pred.grad is not None
