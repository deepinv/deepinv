import deepinv as dinv
import torch
import pytest


class TestTomographyWithAstra:
    def dummy_compute_norm(
        self,
        x0: torch.Tensor,
        max_iter: int = 100,
        tol: float = 1e-3,
        verbose: bool = True,
        squared: bool = True,
    ) -> torch.Tensor:
        return torch.tensor(1.0).to(x0)

    def dummy_projection(self, x: torch.Tensor, out: torch.Tensor) -> None:
        out[:] = 1.0

    @pytest.mark.parametrize("normalize", [True, False, None])
    @pytest.mark.parametrize("fbp", [True, False])
    @pytest.mark.parametrize(
        "is_2d,geometry_type",
        [
            (True, "parallel"),
            (True, "fanbeam"),
            (False, "parallel"),
            (False, "conebeam"),
        ],
    )
    def test_tomography_with_astra_logic(
        self, is_2d, geometry_type, normalize, fbp, monkeypatch
    ):
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
        if device != "cuda":
            monkeypatch.setattr(
                target=dinv.physics.functional.XrayTransform,
                name="_forward_projection",
                value=self.dummy_projection,
            )
            monkeypatch.setattr(
                target=dinv.physics.functional.XrayTransform,
                name="_backprojection",
                value=self.dummy_projection,
            )
            monkeypatch.setattr(
                target=dinv.physics.TomographyWithAstra,
                name="compute_norm",
                value=self.dummy_compute_norm,
            )

        ## Test 2d transforms
        if is_2d:
            img_size = (16, 16)
            n_detector_pixels = 2 * img_size[0]
            num_angles = 2 * img_size[0]
            physics = dinv.physics.TomographyWithAstra(
                img_size=img_size,
                n_detector_pixels=n_detector_pixels,
                angles=num_angles,
                angular_range=((0, 180) if geometry_type == "parallel" else (0, 360)),
                geometry_type=geometry_type,
                normalize=normalize,
                device=device,
            )

        else:
            ## Test 3d transforms
            img_size = (16, 16, 16)
            n_detector_pixels = (32, 32)
            num_angles = 2 * img_size[0]
            physics = dinv.physics.TomographyWithAstra(
                img_size=img_size,
                angles=num_angles,
                angular_range=((0, 180) if geometry_type == "parallel" else (0, 360)),
                n_detector_pixels=n_detector_pixels,
                geometry_type=geometry_type,
                detector_spacing=(1.0, 1.0),
                pixel_spacing=(1.0, 1.0, 1.0),
                normalize=normalize,
                device=device,
            )

        x = torch.rand(1, 1, *img_size, device=device)

        if device != "cuda":
            ## -------- Test forward --------
            Ax = physics.A(x)
            assert Ax.shape == (1, 1, *physics.measurement_shape)

            ## ------- Test backward --------
            y = torch.rand_like(Ax)
            At_y = physics.A_adjoint(y)
            assert At_y.shape == (1, 1, *img_size)

            ## ---- Test pseudo-inverse -----
            x_hat = physics.A_dagger(y, fbp=fbp)
            assert x_hat.shape == (1, 1, *img_size)

            ## --- Test autograd.Function ---
            pred = torch.zeros_like(x, requires_grad=True)
            loss = torch.linalg.norm(physics.A(pred) - Ax)
            loss.backward()
            assert pred.grad is not None

            if normalize is None:
                # when normalize is not set by the user, it should default to True
                assert physics.normalize is True

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
            r_tol = 0.1 if geometry_type != "parallel" and fbp else 0.05
            r = physics.A_adjoint(physics.A(x))
            y = physics.A(r)
            error = torch.linalg.norm(
                physics.A_dagger(y, fbp=fbp) - r
            ) / torch.linalg.norm(r)
            assert error < r_tol

            ## --- Test autograd.Function ---
            pred = torch.zeros_like(x, requires_grad=True)
            loss = torch.linalg.norm(physics.A(pred) - Ax)
            loss.backward()
            assert pred.grad is not None

            threshold = 1e-3 if geometry_type != "conebeam" else 5e-2
            if normalize:
                assert abs(physics.compute_norm(x, squared=False) - 1.0) < threshold

            if normalize is None:
                # when normalize is not set by the user, it should default to True
                assert physics.normalize is True
                assert abs(physics.compute_norm(x, squared=False) - 1.0) < threshold

        ## --- Test geometry properties ---
        if is_2d:
            assert physics.measurement_shape == (32, 32)
            assert physics.xray_transform.domain_shape == (1, 16, 16)
            assert physics.xray_transform.range_shape == (1, 32, 32)
        else:
            assert physics.measurement_shape == (32, 32, 32)
            assert physics.xray_transform.domain_shape == (16, 16, 16)
            assert physics.xray_transform.range_shape == (32, 32, 32)
        assert physics.num_angles == 32
        assert physics.xray_transform.object_cell_volume == pytest.approx(1.0)
        assert physics.xray_transform.detector_cell_u_length == pytest.approx(1.0)
        assert physics.xray_transform.detector_cell_v_length == pytest.approx(1.0)
        assert physics.xray_transform.detector_cell_area == pytest.approx(1.0)
        if geometry_type in ["fanbeam", "conebeam"]:
            assert physics.xray_transform.source_radius == pytest.approx(80.0)
            assert physics.xray_transform.detector_radius == pytest.approx(20.0)
            assert physics.xray_transform.magnification_factor == pytest.approx(1.25)
        else:
            assert physics.xray_transform.magnification_factor == pytest.approx(1.0)
