import pytest

import torch
from torch.utils.data import DataLoader

import deepinv as dinv
from deepinv.loss import adversarial

# NOTE: They're injected in tests as fixtures.
from test_loss import dataset, physics  # noqa: F401

ADVERSARIAL_COMBOS = ["DeblurGAN", "CSGM", "AmbientGAN", "UAIR"]


@pytest.fixture
def imsize():
    return (3, 64, 64)


def build_unet(imsize, device):
    return dinv.models.UNet(
        in_channels=imsize[0],
        out_channels=imsize[0],
        scales=2,
        circular_padding=True,
        batch_norm=False,
        dim=len(imsize) - 1,
    ).to(device)


def build_csgm_generator(imsize, device):
    return dinv.models.CSGMGenerator(
        dinv.models.DCGANGenerator(nz=10, ngf=4, nc=imsize[0], dim=len(imsize) - 1),
        inf_max_iter=100,
        inf_tol=0.001,
    ).to(device)


def choose_adversarial_combo(combo_name, imsize, device):
    if combo_name == "DeblurGAN":
        generator = build_unet(imsize, device)
        discrimin = dinv.models.PatchGANDiscriminator(
            n_layers=1, ndf=8, input_nc=imsize[0], batch_norm=False
        ).to(device)
        gen_loss = [
            dinv.loss.SupLoss(),
            adversarial.SupAdversarialGeneratorLoss(device=device),
        ]
        dis_loss = adversarial.SupAdversarialDiscriminatorLoss(device=device)
    elif combo_name == "UAIR":
        generator = build_unet(imsize, device)
        discrimin = dinv.models.ESRGANDiscriminator(img_size=imsize).to(device)
        gen_loss = adversarial.UAIRGeneratorLoss(device=device)
        dis_loss = adversarial.UnsupAdversarialDiscriminatorLoss(device=device)
    elif combo_name == "CSGM":
        generator = build_csgm_generator(imsize, device)
        discrimin = dinv.models.DCGANDiscriminator(ndf=8, nc=imsize[0]).to(device)
        gen_loss = adversarial.SupAdversarialGeneratorLoss(device=device)
        dis_loss = adversarial.SupAdversarialDiscriminatorLoss(device=device)
    elif combo_name == "AmbientGAN":
        generator = build_csgm_generator(imsize, device)
        discrimin = dinv.models.DCGANDiscriminator(ndf=8, nc=imsize[0]).to(device)
        gen_loss = adversarial.UnsupAdversarialGeneratorLoss(device=device)
        dis_loss = adversarial.UnsupAdversarialDiscriminatorLoss(device=device)

    return generator, discrimin, gen_loss, dis_loss


@pytest.mark.parametrize("combo_name", ADVERSARIAL_COMBOS)
def test_adversarial_training(combo_name, imsize, device, physics, dataset):
    model, D, gen_loss, dis_loss = choose_adversarial_combo(combo_name, imsize, device)

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

    assert final_test["PSNR"] > 0


@pytest.mark.parametrize("test_shape", [(3, 64, 64), (3, 64, 64, 64)])
def test_adversarial_nets(test_shape):
    # We simply test if all nets function on the forward pass and produce correct shapes

    dim = len(test_shape) - 1

    generator = dinv.models.DCGANGenerator(output_size=64, nz=4, ngf=16, dim=str(dim))
    latent = torch.randn([1, 4, *tuple(1 for i in range(dim))])
    out = generator(latent)
    assert out.shape[1:] == test_shape

    discriminator = dinv.models.DCGANDiscriminator(ndf=8, dim=dim)
    batch = torch.randn([1] + list(test_shape))
    out = discriminator(batch)
    assert out.shape == tuple([1, 1] + [1] * dim)

    discriminator = dinv.models.ESRGANDiscriminator(
        img_size=test_shape, filters=(8, 8, 8), dim=dim
    )
    batch = torch.randn([1] + list(test_shape))
    out = discriminator(batch)
    assert out.shape == tuple([1] + list(discriminator.output_shape))

    discriminator = dinv.models.PatchGANDiscriminator(ndf=2, dim=dim)
    batch = torch.randn([1] + list(test_shape))
    out = discriminator(batch)
    assert out.shape == tuple([1, 1] + [11 for i in range(dim)])
