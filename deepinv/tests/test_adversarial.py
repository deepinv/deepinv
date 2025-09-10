import pytest

import torch
from torch.utils.data import DataLoader

import deepinv as dinv
from deepinv.loss import adversarial
from deepinv.physics.generator import BernoulliSplittingMaskGenerator
from dummy import DummyCircles

# NOTE: They're injected in tests as fixtures.
from test_loss import dataset, physics  # noqa: F401

DISCRIMS = [
    "PatchGANDiscriminator",
    "ESRGANDiscriminator",
    "DCGANDiscriminator",
    "SkipConvDiscriminator",
]

ADVERSARIAL_COMBOS = [
    "DeblurGAN",
    "CSGM",
    "AmbientGAN",
    "UAIR",
    "MultiOperatorAdversarial",
]


@pytest.fixture
def imsize():
    return (3, 64, 64)


def choose_discriminator(discrim_name, imsize):
    if discrim_name == "PatchGANDiscriminator":
        return dinv.models.gan.PatchGANDiscriminator(
            input_nc=1, ndf=2, n_layers=5, batch_norm=False, original=False
        )
    elif discrim_name == "ESRGANDiscriminator":
        return dinv.models.gan.ESRGANDiscriminator(
            imsize, hidden_dims=[64] * 6, batch_norm=False
        )
    elif discrim_name == "DCGANDiscriminator":
        return dinv.models.gan.DCGANDiscriminator(ndf=2, nc=1)
    elif discrim_name == "SkipConvDiscriminator":
        return dinv.models.gan.SkipConvDiscriminator(
            imsize[1:], d_dim=2, d_blocks=1, in_channels=1
        )


@pytest.mark.parametrize("discrim_name", DISCRIMS)
def test_discrim_training(discrim_name, imsize, device):
    # Test discriminator training with frozen generator
    imsize = (1, *imsize[1:])
    D = choose_discriminator(discrim_name, imsize).to(device)

    x = DummyCircles(1, imsize).x
    physics = dinv.physics.Inpainting(imsize, mask=0.7, device=device)
    y = physics(x)
    dataloader = torch.utils.data.DataLoader(dinv.datasets.TensorDataset(x=x, y=y))

    model = dinv.models.MedianFilter()
    optimizer = torch.optim.SGD([torch.tensor(0.0, requires_grad=False)])
    optimizer_D = torch.optim.Adam(D.parameters(), lr=1e-3)
    loss = adversarial.SupAdversarialLoss(D=D, optimizer_D=optimizer_D, device=device)

    with torch.no_grad():
        x_net = model(y, physics)
        Dx0 = D(x)
        Dx_net0 = D(x_net)

    # Test discriminator metric
    m = adversarial.DiscriminatorMetric(device=device)
    assert m(Dx_net0, real=False) == (Dx_net0 - 0.0) ** 2
    assert m(Dx0, real=True) == (1.0 - Dx0) ** 2

    trainer = dinv.Trainer(
        model=model,
        physics=physics,
        optimizer=optimizer,
        train_dataloader=dataloader,
        epochs=10,
        losses=loss,
        save_path=None,
        verbose=False,
        show_progress_bar=False,
    )

    model = trainer.train()

    with torch.no_grad():
        x_net = model(y, physics)
        # print(f"x {Dx0} -> {D(x)}, x_net {Dx_net0} -> {D(x_net)}")

        # Real should become more real
        assert 1.0 >= D(x) > Dx0

        # Fake should be more fake than real
        # Note we don't test that fake becomes more fake, as the discrim
        # training is not necessarily monotonic
        assert D(x_net) < D(x)


def choose_adversarial_combo(combo_name, imsize, device, dataset, domain):
    unet = dinv.models.UNet(
        in_channels=imsize[0],
        out_channels=imsize[0],
        scales=2,
        circular_padding=True,
        batch_norm=False,
    ).to(device)

    csgm_generator = dinv.models.CSGMGenerator(
        dinv.models.DCGANGenerator(nz=10, ngf=8, nc=imsize[0]),
        inf_max_iter=100,
        inf_tol=0.001,
    ).to(device)

    # For multi operator losses
    dataloader = DataLoader(dataset[0], batch_size=1, shuffle=True)
    physics_generator = BernoulliSplittingMaskGenerator(
        imsize, 0.5, device=device, rng=torch.Generator(device).manual_seed(42)
    )

    if combo_name == "DeblurGAN":
        model = unet
        D = dinv.models.PatchGANDiscriminator(
            n_layers=1,
            ndf=8,
            input_nc=imsize[0],
            batch_norm=False,
            original=False,
        ).to(device)
        loss = [
            dinv.loss.SupLoss(),
            adversarial.SupAdversarialLoss(D=D, device=device),
        ]
    elif combo_name == "UAIR":
        model = unet
        D = dinv.models.ESRGANDiscriminator(img_size=imsize).to(device)
        loss = adversarial.UAIRLoss(
            D=D,
            device=device,
            domain=domain,
            dataloader=dataloader,
            physics_generator=physics_generator,
        )
    elif combo_name == "CSGM":
        model = csgm_generator
        D = dinv.models.DCGANDiscriminator(ndf=8, nc=imsize[0]).to(device)
        loss = adversarial.SupAdversarialLoss(device=device, D=D)
    elif combo_name == "AmbientGAN":
        model = csgm_generator
        D = dinv.models.DCGANDiscriminator(ndf=8, nc=imsize[0]).to(device)
        loss = adversarial.UnsupAdversarialLoss(D=D, device=device, domain=domain)
    elif combo_name == "MultiOperatorAdversarial":
        model = unet
        D = dinv.models.SkipConvDiscriminator(
            imsize[-2:], d_dim=32, d_blocks=1, use_sigmoid=False, in_channels=3
        )
        loss = adversarial.MultiOperatorUnsupAdversarialLoss(
            D=D,
            dataloader=dataloader,
            physics_generator=physics_generator,
            device=device,
            domain=domain,
        )

    return model, loss


@pytest.mark.parametrize("combo_name", ADVERSARIAL_COMBOS)
@pytest.mark.parametrize("domain", [None, "A_adjoint"])
def test_adversarial_losses(combo_name, imsize, device, physics, dataset, domain):
    # Test generator training with frozen discriminator
    model, loss = choose_adversarial_combo(combo_name, imsize, device, dataset, domain)
    if domain is not None and combo_name in ["DeblurGAN", "CSGM"]:
        pytest.skip("domain does not apply for supervised loss.")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    train_dataloader = DataLoader(dataset[0], batch_size=1, shuffle=True)
    test_dataloader = DataLoader(dataset[1], batch_size=1, shuffle=False)

    trainer = dinv.Trainer(
        model=model,
        physics=physics,
        train_dataloader=train_dataloader,
        epochs=1,
        losses=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        verbose=False,
        show_progress_bar=False,
        save_path=None,
        device=device,
        optimizer_step_multi_dataset=False,
    )

    initial_psnr = trainer.test(test_dataloader)["PSNR"]

    trainer.train()
    final_psnr = trainer.test(test_dataloader)["PSNR"]

    # PSNR won't necessarily increase
    # so this just tests that the loss can be trained
    # and it does something in 1 epoch
    assert final_psnr.item() != initial_psnr.item()


@pytest.mark.parametrize("discrim_name", DISCRIMS)
def test_discriminators(discrim_name, imsize):
    # 1 channel for speed
    D = choose_discriminator(discrim_name, (1, *imsize[1:]))
    x = torch.rand(1, *imsize[1:]).unsqueeze(0)
    y = D(x)
    assert len(y.flatten()) == 1
