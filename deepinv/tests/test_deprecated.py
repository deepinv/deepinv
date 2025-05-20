import pytest
import deepinv as dinv
import torch


def test_deprecated_physics_image_size():
    img_size = (3, 16, 32)
    m = 30
    rng = torch.Generator("cpu").manual_seed(0)
    device = "cpu"

    # Inpainting: tensor_size is changed to img_size
    with pytest.warns(DeprecationWarning, match="tensor_size.*deprecated"):
        p = dinv.physics.Inpainting(
            tensor_size=img_size, mask=0.5, device=device, rng=rng
        )
        assert p.img_size == img_size

    # CS: img_shape is changed to img_size
    with pytest.warns(DeprecationWarning, match="img_shape.*deprecated"):
        p = dinv.physics.CompressedSensing(
            m=m, img_shape=img_size, device="cpu", compute_inverse=True, rng=rng
        )
        assert p.img_size == img_size

    # SinglePixelCamera: img_shape is changed to img_size
    with pytest.warns(DeprecationWarning, match="img_shape.*deprecated"):
        p = dinv.physics.SinglePixelCamera(
            m=m, fast=True, img_shape=img_size, device=device, rng=rng
        )
        assert p.img_size == img_size

    # RandomPhaseRetrieval: img_shape is changed to img_size
    with pytest.warns(DeprecationWarning, match="img_shape.*deprecated"):
        p = dinv.physics.RandomPhaseRetrieval(m=m, img_shape=img_size, device=device)
        assert p.img_size == img_size

    # Ptychography: in_shape is changed to img_size
    with pytest.warns(DeprecationWarning, match="in_shape.*deprecated"):
        p = dinv.physics.Ptychography(
            in_shape=img_size, probe=None, shifts=None, device=device
        )
        assert p.img_size == img_size

    # StructuredRandomPhaseRetrieval: input_shape is changed to img_size
    with pytest.warns(DeprecationWarning, match="input_shape.*deprecated"):
        p = dinv.physics.StructuredRandomPhaseRetrieval(
            input_shape=img_size, output_shape=img_size, n_layers=2, device=device
        )
        assert p.img_size == img_size

    # RandomPhaseRetrieval: img_shape is changed to img_size
    with pytest.warns(DeprecationWarning, match="img_shape.*deprecated"):
        p = dinv.physics.RandomPhaseRetrieval(m=500, img_shape=img_size, device=device)
        assert p.img_size == img_size

    # StructuredRandom: input_shape is changed to img_size
    with pytest.warns(DeprecationWarning, match="input_shape.*deprecated"):
        p = dinv.physics.StructuredRandom(
            input_shape=img_size, output_shape=img_size, device=device
        )
        assert p.img_size == img_size

    # Inpainting mask generators: tensor_size is changed to img_size
    with pytest.warns(DeprecationWarning, match="tensor_size.*deprecated"):
        p = dinv.physics.generator.inpainting.Artifact2ArtifactSplittingMaskGenerator(
            tensor_size=img_size, device=device
        )
        assert p.img_size == img_size

        p = dinv.physics.generator.inpainting.BernoulliSplittingMaskGenerator(
            tensor_size=img_size, split_ratio=0.5, device=device
        )
        assert p.img_size == img_size

        p = dinv.physics.generator.inpainting.GaussianSplittingMaskGenerator(
            tensor_size=img_size, split_ratio=0.5, device=device
        )
        assert p.img_size == img_size

        p = dinv.physics.generator.inpainting.Phase2PhaseSplittingMaskGenerator(
            tensor_size=img_size, device=device
        )
        assert p.img_size == img_size
