import pytest
import deepinv as dinv
import torch
import warnings


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

    # test_no_warning_with_correct_parameter
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        p = dinv.physics.Inpainting(img_size=img_size, mask=0.5, device=device, rng=rng)
        assert p.img_size == img_size
        assert len(record) == 0

    # CS: img_shape is changed to img_size
    with pytest.warns(DeprecationWarning, match="img_shape.*deprecated"):
        p = dinv.physics.CompressedSensing(
            m=m, img_shape=img_size, device="cpu", rng=rng
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
        with pytest.warns(DeprecationWarning, match="output_shape.*deprecated"):
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
        with pytest.warns(DeprecationWarning, match="output_shape.*deprecated"):
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

        p = dinv.physics.generator.inpainting.MultiplicativeSplittingMaskGenerator(
            tensor_size=img_size,
            split_generator=dinv.physics.generator.GaussianMaskGenerator(
                img_size=img_size, acceleration=2
            ),
        )
        assert p.img_size == img_size

    # GAN Discriminator
    with pytest.warns(DeprecationWarning, match="input_shape.*deprecated"):
        model = dinv.models.gan.ESRGANDiscriminator(input_shape=img_size)
        assert model.img_size == img_size

    # Loss
    with pytest.warns(DeprecationWarning, match="tensor_size.*deprecated"):
        loss = dinv.loss.mri.Phase2PhaseLoss(
            tensor_size=(2, 4, 4, 4), dynamic_model=False
        )
        assert loss.img_size == (2, 4, 4, 4)
        loss = dinv.loss.mri.Artifact2ArtifactLoss(
            tensor_size=(2, 4, 4, 4), dynamic_model=False
        )
        assert loss.img_size == (2, 4, 4, 4)

    # DIP
    with pytest.warns(DeprecationWarning, match="img_shape.*deprecated"):
        generator = dinv.models.ConvDecoder(img_shape=(3, 4, 4))
        with pytest.warns(DeprecationWarning, match="input_size.*deprecated"):
            model = dinv.models.DeepImagePrior(
                generator=generator, input_size=(3, 4, 4)
            )

    # DEPRECATED FUNCTION ARGUMENT TESTS
    # single_pixel_camera
    with pytest.warns(DeprecationWarning, match="img_shape.*deprecated"):
        mask = dinv.physics.singlepixel.sequency_mask(img_shape=img_size, m=4)
        assert mask.shape[1:] == img_size

        mask = dinv.physics.singlepixel.old_sequency_mask(img_shape=img_size, m=4)
        assert mask.shape[1:] == img_size

        mask = dinv.physics.singlepixel.zig_zag_mask(img_shape=img_size, m=4)
        assert mask.shape[1:] == img_size

        mask = dinv.physics.singlepixel.xy_mask(img_shape=img_size, m=4)
        assert mask.shape[1:] == img_size

    # structured random
    with pytest.warns(DeprecationWarning, match="input_shape.*deprecated"):
        with pytest.warns(DeprecationWarning, match="output_shape.*deprecated"):
            dinv.physics.structured_random.compare(
                input_shape=img_size, output_shape=img_size
            )

            dinv.physics.structured_random.padding(
                torch.randn((1,) + img_size),
                input_shape=img_size,
                output_shape=img_size,
            )

            dinv.physics.structured_random.trimming(
                torch.randn((1,) + img_size),
                input_shape=img_size,
                output_shape=img_size,
            )


def test_deprecated_functions():
    with pytest.warns(DeprecationWarning):
        dinv.utils.rescale_img(torch.randn(3, 16, 32), rescale_mode="min_max")
