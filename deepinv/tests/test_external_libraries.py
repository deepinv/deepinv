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

    def dummy_create_projector(
        self,
        type: str,
        projection_geometry: dict[str, float],
        object_geometry: dict[str, float],
    ) -> int:
        return 1

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
    @pytest.mark.parametrize("cubic", [True, False])
    @pytest.mark.parametrize("channels", [1, 3])
    def test_tomography_with_astra_logic(
        self, is_2d, geometry_type, normalize, fbp, monkeypatch, cubic, channels, device
    ):
        r"""
        Tests tomography operator with astra backend which does not have a numerically precise adjoint.

        :param bool is_2d: Runs the test with 2D geometry, else 3D.
        :param str geometry_type: In 2D, expects ``parallel`` or ``fanbeam``. In 3D expects ``parallel`` or ``conebeam``.
        :param bool normalize: Initializes the operator with ``normalize=normalize``.
        :param bool fbp: Whether or not to approximate the pseudo-inverse with filtered back-projection.
        :param bool cubic: Whether or not the input image is cubic (i.e. has the same size in all dimensions).
        :param int channels: Number of input channels. The tomography operator is applied per channel.
        :param str device: The device to run the test on.
        """

        astra = pytest.importorskip(
            "astra",
            reason="This test requires astra-toolbox. It should be "
            "installed with `conda install -c astra-toolbox -c nvidia astra-toolbox`",
        )

        if "cuda" not in str(device):
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
            monkeypatch.setattr(
                target=astra,
                name="create_projector",
                value=self.dummy_create_projector,
            )

        ## Test 2d transforms
        if is_2d:
            img_size = (16, 16) if cubic else (32, 16)
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
            img_size = (16, 16, 16) if cubic else (32, 24, 16)
            n_detector_pixels = (32, 32) if cubic else (64, 48)
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

        x = torch.rand(1, channels, *img_size, device=device)

        if device != "cuda":
            ## -------- Test forward --------
            Ax = physics.A(x)
            assert Ax.shape == (1, channels, *physics.measurement_shape)

            ## ------- Test backward --------
            y = torch.rand_like(Ax)
            At_y = physics.A_adjoint(y)
            assert At_y.shape == (1, channels, *img_size)

            ## ---- Test pseudo-inverse -----
            x_hat = physics.A_dagger(y, fbp=fbp)
            assert x_hat.shape == (1, channels, *img_size)

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
            assert physics.measurement_shape == (32, 32) if cubic else (32, 16)
            assert (
                physics.xray_transform.domain_shape == (1, 16, 16)
                if cubic
                else (1, 32, 16)
            )
            assert (
                physics.xray_transform.range_shape == (1, 32, 32)
                if cubic
                else (1, 32, 16)
            )
        else:
            assert physics.measurement_shape == (32, 32, 32) if cubic else (64, 48, 32)
            assert (
                physics.xray_transform.domain_shape == (16, 16, 16)
                if cubic
                else (32, 24, 16)
            )
            assert (
                physics.xray_transform.range_shape == (32, 32, 32)
                if cubic
                else (64, 48, 32)
            )
        assert physics.num_angles == 32 if cubic else 64
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


class TestTomographyWithRTK:
    """Integration tests for TomographyWithRTK using the RTK CUDA backend."""

    @pytest.fixture(autouse=True)
    def rtk_env(self):
        """Skip the test if itk-rtk or a CUDA GPU is unavailable"""
        itk = pytest.importorskip("itk", reason="itk is not installed")

        try:
            from itk import RTK as rtk
        except ImportError:
            pytest.skip("itk-rtk is not installed")

        if not torch.cuda.is_available():
            pytest.skip("TomographyWithRTK requires a CUDA GPU")

        self.rtk = rtk

    @pytest.fixture
    def device(self):
        return "cuda:0"

    def _make_geometry(
        self,
        n_angles: int,
        source_to_isocenter: float = 500.0,
        source_to_detector: float = 1000.0,
    ):
        """Build a circular geometry."""
        geometry = self.rtk.ThreeDCircularProjectionGeometry.New()
        angular_step = 360.0 / n_angles
        for i in range(n_angles):
            geometry.AddProjection(
                source_to_isocenter,
                source_to_detector,
                i * angular_step,
            )
        return geometry

    @staticmethod
    def _make_fanbeam_setup(img_size_2d, n_detector_pixels, n_angles):
        """Return (proj_info, vol_info) for a 2D fan-beam operator."""
        W, H = img_size_2d
        D = n_detector_pixels
        proj_info = {
            "size": [D, n_angles],
            "spacing": [1.0, 1.0],
            "origin": [-0.5 * (D - 1), -0.5 * (n_angles - 1)],
        }
        vol_info = {
            "size": [W, H],
            "spacing": [1.0, 1.0],
            "origin": [-0.5 * (W - 1), -0.5 * (H - 1)],
        }
        return proj_info, vol_info

    @staticmethod
    def _make_conebeam_setup(img_size_3d, proj_shape_2d, n_angles):
        """Return (proj_info, vol_info) for a 3D cone-beam operator."""
        W, D_y, H = img_size_3d
        Du, Dv = proj_shape_2d
        proj_info = {
            "size": [Du, Dv, n_angles],
            "spacing": [1.0, 1.0, 1.0],
            "origin": [-0.5 * (Du - 1), -0.5 * (Dv - 1), -0.5 * (n_angles - 1)],
        }
        vol_info = {
            "size": [W, D_y, H],
            "spacing": [1.0, 1.0, 1.0],
            "origin": [-0.5 * (W - 1), -0.5 * (D_y - 1), -0.5 * (H - 1)],
        }
        return proj_info, vol_info

    # Shapes sanity
    # -----------------------------------------------

    @pytest.mark.parametrize("normalize", [True, False])
    @pytest.mark.parametrize(
        "mode,img_size,proj_size",
        [
            ("fanbeam", (32, 16), 32),
            ("conebeam", (32, 24, 16), (64, 48)),
        ],
    )
    def test_shapes_and_sanity(self, mode, img_size, proj_size, normalize, device):
        """
        Verifies that A, A_adjoint, and fbp return tensors of the correct shape.
        """
        n_angles = 64

        if mode == "fanbeam":
            W, H = img_size
            D = proj_size
            geometry = self._make_geometry(n_angles)
            proj_info, vol_info = self._make_fanbeam_setup(img_size, D, n_angles)
            expected_proj_shape = (1, 1, n_angles, D)
            expected_vol_shape = (1, 1, H, W)
        else:
            Du, Dv = proj_size
            geometry = self._make_geometry(n_angles)
            proj_info, vol_info = self._make_conebeam_setup(
                img_size, proj_size, n_angles
            )
            expected_proj_shape = (1, 1, n_angles, Dv, Du)
            expected_vol_shape = (1, 1, *img_size)

        physics = dinv.physics.TomographyWithRTK(
            geometry=geometry,
            projection_stack_information=proj_info,
            volume_information=vol_info,
            mode=mode,
            normalize=normalize,
            ray_step_size=1.0,
            verbose=False,
        )

        x = torch.rand(1, 1, *img_size, device=device)
        Ax = physics.A(x)
        assert Ax.shape == expected_proj_shape, (
            f"[{mode}] A output shape mismatch: got {Ax.shape}, "
            f"expected {expected_proj_shape}"
        )

        y = torch.rand_like(Ax)
        Aty = physics.A_adjoint(y)
        assert Aty.shape == expected_vol_shape, (
            f"[{mode}] A_adjoint shape mismatch: got {Aty.shape}, "
            f"expected {expected_vol_shape}"
        )

        reco = physics.fbp(y)
        assert reco.shape == expected_vol_shape, (
            f"[{mode}] fbp shape mismatch: got {reco.shape}, "
            f"expected {expected_vol_shape}"
        )

    # Adjointness
    # -----------------------------------------------

    @pytest.mark.parametrize("mode", ["fanbeam", "conebeam"])
    def test_adjointness(self, mode, device):
        """
        Checks adjointness with the RTK projectors.
        """
        n_angles = 32

        if mode == "fanbeam":
            img_size = (32, 32)
            D = 64
            geometry = self._make_geometry(n_angles)
            proj_info, vol_info = self._make_fanbeam_setup(img_size, D, n_angles)
            x_shape = (1, 1, *img_size)
            y_shape = (1, 1, n_angles, D)
        else:
            img_size = (32, 32, 32)
            proj_size = (64, 64)
            geometry = self._make_geometry(n_angles)
            proj_info, vol_info = self._make_conebeam_setup(
                img_size, proj_size, n_angles
            )
            x_shape = (1, 1, *img_size)
            y_shape = (1, 1, n_angles, *proj_size)

        physics = dinv.physics.TomographyWithRTK(
            geometry=geometry,
            projection_stack_information=proj_info,
            volume_information=vol_info,
            mode=mode,
            normalize=False,
            ray_step_size=1.0,
        )

        x = torch.rand(*x_shape, device=device)
        y = torch.rand(*y_shape, device=device)
        Ax = physics.A(x)
        Aty = physics.A_adjoint(y)

        Ax_y = torch.sum(Ax * y).item()
        x_Aty = torch.sum(x * Aty).item()

        relative_error = abs(Ax_y - x_Aty) / (abs(x_Aty) + 1e-12)
        assert relative_error < 0.05, (
            f"[{mode}] Adjointness failed: ⟨Ax,y⟩={Ax_y:.6f}, "
            f"⟨x,Aty⟩={x_Aty:.6f}, rel_err={relative_error:.2e}"
        )

    # normalization
    # -----------------------------------------------

    @pytest.mark.parametrize("mode", ["fanbeam", "conebeam"])
    def test_normalize_stores_norm_mat(self, mode, device):

        n_angles = 16

        if mode == "fanbeam":
            geometry = self._make_geometry(n_angles)
            proj_info, vol_info = self._make_fanbeam_setup((16, 16), 32, n_angles)
        else:
            geometry = self._make_geometry(n_angles)
            proj_info, vol_info = self._make_conebeam_setup(
                (16, 16, 16), (32, 32), n_angles
            )

        physics = dinv.physics.TomographyWithRTK(
            geometry=geometry,
            projection_stack_information=proj_info,
            volume_information=vol_info,
            mode=mode,
            normalize=True,
            ray_step_size=1.0,
        )

        assert physics.norm_mat is not None
