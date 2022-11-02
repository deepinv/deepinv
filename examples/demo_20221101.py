import deepinv as dinv

dataloader = dinv.datasets.mnist_dataloader(mode='test', batch_size=128, num_workers=4, shuffle=True)

physics = dinv.physics.inpainting(img_width=28,
                                  img_heigth=28,
                                  mask_rate=0.3,
                                  device=dinv.device)

model = dinv.models.unet(in_channels=1,
                         out_channels=1,
                         circular_padding=True,
                         compact=3).to(dinv.device)


loss_mc = dinv.loss.MCLoss(physics=physics,
                           metric=dinv.metric.mse(dinv.device))

loss_ei = dinv.loss.EILoss(transform=dinv.transform.Shift(n_trans=2),
                           physics=physics,
                           metric=dinv.metric.mse(dinv.device))

loss_ms = dinv.loss.MeaSplitLoss(physics=physics,
                                 metric=dinv.metric.mse(dinv.device),
                                 division_mask_rate=0.5)


noise = dinv.noise.Poisson(gamma=0.1,
                           mask=physics.mask)

loss_sure = dinv.loss.SURE_Poisson_Loss(gamma=0.1,
                                        tau=1e-2,
                                        physics=physics)

loss_rei = dinv.loss.RobustEILoss(transform=dinv.transform.Shift(n_trans=2),
                                  physics=physics,
                                  noise=noise)


optimizer = dinv.optim.Adam(model.parameters(),
                            lr=1e-4,
                            weight_decay=1e-8)


dinv.train(model=model,
           train_dataloader=dataloader,
           learning_rate=5e-4,
           physics=physics,
           epochs=20,
           schedule=[10],
           noise=noise,
           loss_closure=[loss_mc, loss_ei],
           # loss_closure=[loss_sure, loss_rei],
           # loss_closure=[loss_ms, loss_ei],
           loss_weight=[1,1],
           optimizer=optimizer,
           device=dinv.device,
           ckp_interval=10,
           save_path='dinv_mc_ei',
           verbos=True)

