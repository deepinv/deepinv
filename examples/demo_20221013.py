import torch
import deepinv as dinv

dataloader = dinv.datasets.mnist_dataloader()

physics = dinv.physics.inpainting(img_width=28,
                                  img_heigth=28,
                                  mask_rate=0.3,
                                  device=dinv.device)

model = dinv.models.unet(in_channels=1,
                         out_channels=1,
                         circular_padding=True,
                         compact=3).to(dinv.device)

# model = dinv.data_parallel(model, ngpu=4)

loss = dinv.loss.EILoss(transform=dinv.transform.Shift(n_trans=2),
                        physics=physics,
                        ei_loss_weight=0.1,
                        metric=torch.nn.MSELoss().to(dinv.device))


optimizer = torch.optim.Adam(model.parameters(),
                             lr=5e-4,
                             weight_decay=1e-8)

dinv.train(model=model,
           train_dataloader=dataloader,
           learning_rate=5e-4,
           physics=physics,
           epochs=100,
           schedule=[50],
           loss_closure=loss,
           optimizer=optimizer,
           device=dinv.device)