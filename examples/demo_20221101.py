import sys
sys.path.append('../deepinv')
import deepinv as dinv


physics = dinv.physics.compressed_sensing(m=100, img_shape=(1,28,28), save=True).to(dinv.device)


# todo: dinv.generate_dataset(dataset, physics, pairs=True) if pairs == True we get (x,y, physics), otherwise (y, physics)
# todo: here dataset contains images (mnist_dataset)
# todo: keep in mind how we add MOI
# todo: dinv.generate_dataset(dataset, [physics1,...,physics10], pairs=True) if pairs == True we get (x,y, physicsx), otherwise (y, physicsx)

ataloader = dinv.datasets.mnist_dataloader(mode='train', batch_size=128, num_workers=4, shuffle=True)

backbone = dinv.models.unet(in_channels=1,
                         out_channels=1,
                         circular_padding=True,
                         scales=3).to(dinv.device)

model = dinv.models.FBPNet(backbone)

loss_mc = dinv.loss.MCLoss(metric=dinv.metric.mse(dinv.device))
loss_sup = dinv.loss.SupLoss(metric=dinv.metric.mse(dinv.device))

loss_ei = dinv.loss.EILoss(transform=dinv.transform.Shift(n_trans=2),
                           metric=dinv.metric.mse(dinv.device))

loss_mcsure = dinv.loss.SureMCLoss(sigma=.2)

loss_ms = dinv.loss.MeaSplitLoss(metric=dinv.metric.mse(dinv.device),
                                 split_ratio=0.9)

optimizer = dinv.optim.Adam(model.parameters(),
                            lr=5e-4,
                            weight_decay=1e-8)


dinv.train(model=model,
           train_dataloader=dataloader,
           learning_rate=5e-4,
           physics=physics,
           epochs=500,
           schedule=[400],
           loss_closure=[loss_mcsure,loss_ei],
           loss_weight=[1],
           optimizer=optimizer,
           device=dinv.device,
           ckp_interval=250,
           save_path='dinv_mcsure_ei',
           verbose=True)
