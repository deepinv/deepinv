import pytest

import torch
from torch.utils.data import DataLoader

import deepinv as dinv
from deepinv.loss import adversarial
from deepinv.physics.generator import BernoulliSplittingMaskGenerator
from test_loss import dataset, physics

ADVERSARIAL_COMBOS = [
    "DeblurGAN",
    "CSGM",
    "AmbientGAN",
    "UAIR",
    "MultiOperatorAdversarial",
]
DISCRIMS = [
    "PatchGANDiscriminator",
    "ESRGANDiscriminator",
    "DCGANDiscriminator",
    "SkipConvDiscriminator",
]


@pytest.fixture
def imsize():
    return (3, 64, 64)


def choose_adversarial_combo(combo_name, imsize, device, dataset):
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

    if combo_name == "DeblurGAN":
        generator = unet
        discrimin = dinv.models.PatchGANDiscriminator(
            n_layers=1,
            ndf=8,
            input_nc=imsize[0],
            batch_norm=False,
            original=False,
        ).to(device)
        gen_loss = [
            dinv.loss.SupLoss(),
            adversarial.SupAdversarialGeneratorLoss(device=device),
        ]
        dis_loss = adversarial.SupAdversarialDiscriminatorLoss(device=device)
    elif combo_name == "UAIR":
        generator = unet
        discrimin = dinv.models.ESRGANDiscriminator(input_shape=imsize).to(device)
        gen_loss = adversarial.UAIRGeneratorLoss(device=device)
        dis_loss = adversarial.UAIRDiscriminatorLoss(device=device)
    elif combo_name == "CSGM":
        generator = csgm_generator
        discrimin = dinv.models.DCGANDiscriminator(ndf=8, nc=imsize[0]).to(device)
        gen_loss = adversarial.SupAdversarialGeneratorLoss(device=device)
        dis_loss = adversarial.SupAdversarialDiscriminatorLoss(device=device)
    elif combo_name == "AmbientGAN":
        generator = csgm_generator
        discrimin = dinv.models.DCGANDiscriminator(ndf=8, nc=imsize[0]).to(device)
        gen_loss = adversarial.UnsupAdversarialGeneratorLoss(device=device)
        dis_loss = adversarial.UnsupAdversarialDiscriminatorLoss(device=device)
    elif combo_name == "MultiOperatorAdversarial":
        generator = unet
        discrimin = dinv.models.SkipConvDiscriminator(
            imsize[-2:], d_dim=32, d_blocks=1, use_sigmoid=False, in_channels=3
        )
        dataloader_factory = lambda: DataLoader(dataset[0], batch_size=1, shuffle=True)
        physics_generator_factory = lambda: BernoulliSplittingMaskGenerator(
            imsize, 0.5, device=device, rng=torch.Generator(device).manual_seed(42)
        )
        gen_loss = adversarial.MultiOperatorUnsupAdversarialGeneratorLoss(
            dataloader_factory=dataloader_factory,
            physics_generator_factory=physics_generator_factory,
            device=device,
        )
        dis_loss = adversarial.MultiOperatorUnsupAdversarialDiscriminatorLoss(
            dataloader_factory=dataloader_factory,
            physics_generator_factory=physics_generator_factory,
            device=device,
        )

    return generator, discrimin, gen_loss, dis_loss


@pytest.mark.parametrize("combo_name", ADVERSARIAL_COMBOS)
def test_adversarial_training(combo_name, imsize, device, physics, dataset):
    model, D, gen_loss, dis_loss = choose_adversarial_combo(
        combo_name, imsize, device, dataset
    )

    optimizer = dinv.training.adversarial.AdversarialOptimizer(
        torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-8),
        torch.optim.Adam(D.parameters(), lr=1e-4, weight_decay=1e-8),
    )
    scheduler = dinv.training.adversarial.AdversarialScheduler(
        torch.optim.lr_scheduler.StepLR(optimizer.G, step_size=5, gamma=0.9),
        torch.optim.lr_scheduler.StepLR(optimizer.D, step_size=5, gamma=0.9),
    )

    train_dataloader = DataLoader(dataset[0], batch_size=1, shuffle=True)
    test_dataloader = DataLoader(dataset[1], batch_size=1, shuffle=False)

    trainer = dinv.training.AdversarialTrainer(
        model=model,
        D=D,
        physics=physics,
        train_dataloader=train_dataloader,
        epochs=1,
        losses=gen_loss,
        losses_d=dis_loss,
        optimizer=optimizer,
        scheduler=scheduler,
        verbose=False,
        show_progress_bar=False,
        save_path=None,
        device=device,
        optimizer_step_multi_dataset=False,
    )

    # initial_psnr = trainer.test(test_dataloader)[0]

    trainer.train()
    final_test = trainer.test(test_dataloader)

    # if not isinstance(gen_loss, adversarial.MultiOperatorUnsupAdversarialGeneratorLoss):
    # Multi-operator won't train in dummy case
    assert final_test["PSNR"] > 0


def choose_discriminator(discrim_name, imsize):
    if discrim_name == "PatchGANDiscriminator":
        return dinv.models.gan.PatchGANDiscriminator(
            input_nc=1, ndf=2, n_layers=5, batch_norm=False, original=False
        )
    elif discrim_name == "ESRGANDiscriminator":
        return dinv.models.gan.ESRGANDiscriminator(
            imsize, hidden_dims=[64] * 6, bn=False
        )
    elif discrim_name == "DCGANDiscriminator":
        return dinv.models.gan.DCGANDiscriminator(ndf=2, nc=1)
    elif discrim_name == "SkipConvDiscriminator":
        return dinv.models.gan.SkipConvDiscriminator(
            imsize[1:], d_dim=2, d_blocks=1, in_channels=1
        )


@pytest.mark.parametrize("discrim_name", DISCRIMS)
def test_discriminators(discrim_name, imsize):
    # 1 channel for speed
    D = choose_discriminator(discrim_name, (1, *imsize[1:]))
    x = torch.rand(1, *imsize[1:]).unsqueeze(0)
    y = D(x)
    assert len(y.flatten()) == 1
